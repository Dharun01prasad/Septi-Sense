"""
SeptiSense backend - runs on the laptop.

  ESP32 node --HTTP POST /api/v1/telemetry (1 record/s, acknowledged)--> this service
  this service --WebSocket /ws (live waveforms, vitals, predictions, alerts, status)--> dashboard
  dashboard   --REST /api/v1/...--> accounts, history, uploads, alerts, config, replay

Patients sign up / sign in. "Start monitoring" links the wearable to the signed-in patient: everything the device
sends from then on is stored as that patient's monitoring session. Uploaded recordings and replays of held-out
dataset patients run through the same model.  Keys: 'sess:<id>' (monitoring session), 'up:<id>' (upload),
'replay:<pid>' (held-out hospital-B patient), 'device' (device events with no session).

Start:  python app.py          (then open http://localhost:8000)
Educational / research prototype - NOT a medical device.
"""
import asyncio, collections, csv, hashlib, hmac, io, json, os, secrets, socket, sqlite3, threading, time
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

import explain
import notify
import respiration

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB_PATH = os.path.join(HERE, 'septisense.db')
DASH_DIR = os.path.join(ROOT, '05_Dashboard')
REPLAY_DIR = os.path.join(ROOT, '02_ML', 'replay')
RESULTS_DIR = os.path.join(ROOT, '02_ML', 'results')
DEVICE_KEY = 'device'
COOKIE = 'ss_session'
SESSION_DAYS = 7

DEFAULT_CONFIG = {
    'seconds_per_model_hour': 3600,   # 3600 = real time. Demo mode compresses one model hour into N seconds.
    'alarm_threshold': None,          # None -> the model's validated threshold (90 % specificity on hospital A)
    'warn_fraction': 0.6,             # warn when risk >= warn_fraction x alarm threshold
    'alarm_persist_hours': 2,
    'hr_high': 120, 'hr_low': 45, 'spo2_low': 92, 'temp_high': 38.3, 'temp_low': 36.0, 'rr_high': 24,
    'vital_alert_seconds': 30,
    'device_offline_seconds': 10,
    'temp_offset_c': 3.5,             # wrist skin -> core estimate; set by calibration against a thermometer
    'replay_seconds_per_hour': 2.0,
    'sms_enabled': False,             # phone alerts through an Android SMS gateway (see notify.py)
    'sms_gateway_url': '',
    'sms_user': '',
    'sms_pass': '',
    'sms_country_code': '+91',
    'sms_min_interval_min': 15,       # at most one SMS per patient per N minutes
    'sms_for_replays': False,         # also text the person who started a replay (clearly marked DEMO)
}
SECRET = ('sms_pass',)
DEVICE_KINDS = ('device_offline', 'ecg_leads_off', 'no_finger', 'hr', 'spo2', 'temp', 'rr')

app = FastAPI(title='SeptiSense backend', version='2.0')

# ============================================================ storage
_db_lock = threading.Lock()
db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.execute('PRAGMA journal_mode=WAL')
db.executescript('''
CREATE TABLE IF NOT EXISTS telemetry (
  device_id TEXT, boot_id INTEGER, seq INTEGER, t_ms INTEGER, recv_ms INTEGER, late INTEGER,
  hr_ecg REAL, hr_ppg REAL, spo2 REAL, skin_temp REAL, amb_temp REAL, rmssd REAL, sdnn REAL,
  sqi_ecg REAL, sqi_ppg REAL, sqi_temp REAL, leads_off INTEGER, finger INTEGER, hr_agree INTEGER,
  PRIMARY KEY (device_id, boot_id, seq));
CREATE INDEX IF NOT EXISTS tel_t ON telemetry(t_ms);
CREATE TABLE IF NOT EXISTS resp (patient TEXT, t_ms INTEGER, rr REAL, quality REAL, parts TEXT);
CREATE INDEX IF NOT EXISTS resp_t ON resp(t_ms);
CREATE TABLE IF NOT EXISTS hourly (patient TEXT, hour INTEGER, t_ms INTEGER, HR REAL, O2Sat REAL, Temp REAL,
  Resp REAL, SBP REAL, coverage TEXT, label INTEGER, PRIMARY KEY (patient, hour));
CREATE TABLE IF NOT EXISTS predictions (patient TEXT, hour INTEGER, t_ms INTEGER, risk REAL, risk_lo REAL,
  risk_hi REAL, quality TEXT, reasons TEXT, PRIMARY KEY (patient, hour));
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, patient TEXT, kind TEXT, severity TEXT,
  message TEXT, value REAL, start_ms INTEGER, end_ms INTEGER, acked INTEGER DEFAULT 0, acked_ms INTEGER);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, phone TEXT UNIQUE,
  pw_salt TEXT, pw_hash TEXT, age REAL, gender INTEGER, sms INTEGER DEFAULT 1, created_ms INTEGER);
CREATE TABLE IF NOT EXISTS auth_sessions (token TEXT PRIMARY KEY, user_id INTEGER, created_ms INTEGER, expires_ms INTEGER);
CREATE TABLE IF NOT EXISTS monitor_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
  start_ms INTEGER, end_ms INTEGER);
CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, name TEXT,
  created_ms INTEGER, n_hours INTEGER, age REAL, gender INTEGER, label_start_hour INTEGER, onset_hour INTEGER);
CREATE TABLE IF NOT EXISTS notifications (id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER, user_id INTEGER,
  phone TEXT, sent_ms INTEGER, ok INTEGER, detail TEXT);
''')
if 'horizons' not in [r[1] for r in db.execute('PRAGMA table_info(predictions)')]:   # older database
    db.execute('ALTER TABLE predictions ADD COLUMN horizons TEXT')
db.commit()


def q(sql, args=(), one=False):
    with _db_lock:
        cur = db.execute(sql, args)
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description] if cur.description else []
    rows = [{c: (None if isinstance(v, float) and v != v else v) for c, v in zip(cols, r)} for r in rows]   # NaN -> null
    return (rows[0] if rows else None) if one else rows


def x(sql, args=()):
    with _db_lock:
        cur = db.execute(sql, args)
        db.commit()
        return cur.lastrowid


def numeric_frame(rows):
    """DataFrame with every column except device_id numeric (an all-NULL column would otherwise be 'object')."""
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c != 'device_id':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def kv_get(k, default=None):
    r = q('SELECT v FROM kv WHERE k=?', (k,), one=True)
    return json.loads(r['v']) if r else default


def kv_set(k, v):
    x('INSERT OR REPLACE INTO kv VALUES (?,?)', (k, json.dumps(v)))


CONFIG = dict(DEFAULT_CONFIG, **{k: v for k, v in kv_get('config', {}).items() if k in DEFAULT_CONFIG})


def save_config():
    kv_set('config', {k: CONFIG[k] for k in DEFAULT_CONFIG})

# ============================================================ model
MODEL = None
try:
    from inference import RiskModel
    MODEL = RiskModel()
    print(f'model loaded: {MODEL.meta["feature_set"]}, horizons {sorted(MODEL.horizons)} h, '
          f'6 h threshold {MODEL.threshold:.3f}')
except Exception as e:  # backend still runs (telemetry, dashboard) without a model
    print('MODEL NOT LOADED:', e)


def alarm_threshold():
    t = CONFIG.get('alarm_threshold')
    return float(t) if t is not None else (MODEL.threshold if MODEL else 0.5)


def horizon_thresholds():
    """Alarm threshold per horizon. A manual 6 h threshold scales the other horizons by the same factor."""
    if not MODEL:
        return {}
    f = alarm_threshold() / MODEL.threshold
    return {str(h): t * f for h, t in MODEL.thresholds().items()}

# ============================================================ accounts
def now_ms():
    return int(time.time() * 1000)


def _hash(pw, salt):
    return hashlib.pbkdf2_hmac('sha256', pw.encode(), bytes.fromhex(salt), 200_000).hex()


def public_user(u):
    return {k: u[k] for k in ('id', 'name', 'phone', 'age', 'gender', 'sms', 'created_ms')} if u else None


def user_from_token(tok):
    if not tok:
        return None
    return q('SELECT u.* FROM auth_sessions s JOIN users u ON u.id = s.user_id WHERE s.token=? AND s.expires_ms>?',
             (tok, now_ms()), one=True)


def _token(req):
    a = req.headers.get('authorization', '')
    return req.cookies.get(COOKIE) or (a[7:] if a.lower().startswith('bearer ') else None)


def current_user(req: Request):
    u = user_from_token(_token(req))
    if not u:
        raise HTTPException(401, 'please sign in')
    return u


def _new_session(response: Response, uid):
    tok = secrets.token_urlsafe(32)
    x('INSERT INTO auth_sessions VALUES (?,?,?,?)', (tok, uid, now_ms(), now_ms() + SESSION_DAYS * 86400000))
    response.set_cookie(COOKIE, tok, max_age=SESSION_DAYS * 86400, httponly=True, samesite='lax', path='/')


def _profile_fields(b, partial=False):
    out = {}
    if 'name' in b or not partial:
        name = str(b.get('name') or '').strip()
        if not name:
            raise HTTPException(400, 'please enter your name')
        out['name'] = name[:80]
    if 'phone' in b or not partial:
        phone = notify.normalize_phone(str(b.get('phone') or ''), CONFIG['sms_country_code'])
        if len(phone) < 9:
            raise HTTPException(400, 'please enter a valid phone number')
        out['phone'] = phone
    if 'age' in b or not partial:
        try:
            age = float(b.get('age'))
        except (TypeError, ValueError):
            raise HTTPException(400, 'please enter your age')
        if not 1 <= age <= 120:
            raise HTTPException(400, 'age must be between 1 and 120')
        out['age'] = age
    if 'gender' in b or not partial:
        out['gender'] = 1 if int(b.get('gender', 1)) == 1 else 0
    if 'sms' in b:
        out['sms'] = 1 if b['sms'] else 0
    return out


@app.post('/api/v1/auth/signup')
async def api_signup(req: Request, response: Response):
    b = await req.json()
    f = _profile_fields(b)
    pw = str(b.get('password') or '')
    if len(pw) < 6:
        raise HTTPException(400, 'password must have at least 6 characters')
    if q('SELECT id FROM users WHERE phone=?', (f['phone'],), one=True):
        raise HTTPException(409, 'an account with this phone number already exists - please sign in')
    salt = secrets.token_hex(16)
    uid = x('INSERT INTO users (name, phone, pw_salt, pw_hash, age, gender, sms, created_ms) VALUES (?,?,?,?,?,?,?,?)',
            (f['name'], f['phone'], salt, _hash(pw, salt), f['age'], f['gender'], 1, now_ms()))
    _new_session(response, uid)
    return public_user(q('SELECT * FROM users WHERE id=?', (uid,), one=True))


@app.post('/api/v1/auth/login')
async def api_login(req: Request, response: Response):
    b = await req.json()
    u = q('SELECT * FROM users WHERE phone=?', (notify.normalize_phone(str(b.get('phone') or ''), CONFIG['sms_country_code']),), one=True)
    if not u or not hmac.compare_digest(u['pw_hash'], _hash(str(b.get('password') or ''), u['pw_salt'])):
        raise HTTPException(401, 'wrong phone number or password')
    _new_session(response, u['id'])
    return public_user(u)


@app.post('/api/v1/auth/logout')
def api_logout(req: Request, response: Response):
    x('DELETE FROM auth_sessions WHERE token=?', (_token(req),))
    response.delete_cookie(COOKIE, path='/')
    return {'ok': True}


@app.get('/api/v1/auth/me')
def api_me(req: Request):
    return public_user(current_user(req))


@app.put('/api/v1/auth/me')
async def api_me_update(req: Request):
    u = current_user(req)
    f = _profile_fields(await req.json(), partial=True)
    if 'phone' in f and q('SELECT id FROM users WHERE phone=? AND id<>?', (f['phone'], u['id']), one=True):
        raise HTTPException(409, 'another account uses this phone number')
    for k, v in f.items():
        x(f'UPDATE users SET {k}=? WHERE id=?', (v, u['id']))
    return public_user(q('SELECT * FROM users WHERE id=?', (u['id'],), one=True))

# ============================================================ patients: sessions, uploads, replays
ACTIVE_SESSION = {'id': kv_get('active_session')}
REPLAY_INFO = {}


def active_session():
    sid = ACTIVE_SESSION['id']
    return q('SELECT * FROM monitor_sessions WHERE id=? AND end_ms IS NULL', (sid,), one=True) if sid else None


def live_key():
    return f"sess:{ACTIVE_SESSION['id']}" if ACTIVE_SESSION['id'] else None


def replay_meta(pid):
    if pid not in REPLAY_INFO:
        path = os.path.join(REPLAY_DIR, f'{pid}.json')
        if not os.path.exists(path):
            return None
        d = json.load(open(path))
        REPLAY_INFO[pid] = dict(name=f'Recorded patient {pid} (hospital B, held out)', age=d['age'], gender=d['gender'],
                                source='replay', onset_hour=d['onset_hour'], label_start_hour=d['label_start_hour'],
                                total_hours=len(d['hours']), started_by=None)
    return REPLAY_INFO[pid]


def patient_info(key):
    kind, _, ident = (key or '').partition(':')
    try:
        if kind == 'sess':
            r = q('SELECT m.id, m.user_id, m.start_ms, m.end_ms, u.name, u.age, u.gender FROM monitor_sessions m '
                  'JOIN users u ON u.id = m.user_id WHERE m.id=?', (int(ident),), one=True)
            return dict(r, source='device', session_start_ms=r['start_ms']) if r else None
        if kind == 'up':
            r = q('SELECT * FROM uploads WHERE id=?', (int(ident),), one=True)
            return dict(r, source='upload', total_hours=r['n_hours']) if r else None
        if kind == 'replay':
            return replay_meta(ident)
    except ValueError:
        return None
    return None


def owner(key):
    info = patient_info(key) if key and not key.startswith('replay:') else None
    return info.get('user_id') if info else None


def require_access(u, key):
    if key.startswith('replay:') and patient_info(key):
        return patient_info(key)
    info = patient_info(key)
    if not info or info.get('user_id') != u['id']:
        raise HTTPException(404, 'not found')
    return info


def user_keys(uid):
    return ([f"sess:{r['id']}" for r in q('SELECT id FROM monitor_sessions WHERE user_id=?', (uid,))] +
            [f"up:{r['id']}" for r in q('SELECT id FROM uploads WHERE user_id=?', (uid,))])

# ============================================================ live state (memory)
RING_S = 120
ecg_ring = collections.deque(maxlen=250 * RING_S)        # (t_s, value)
beats = collections.deque(maxlen=400)                     # (t_s, amplitude)
ppg_ring = collections.deque(maxlen=25 * RING_S)          # (t_s, ir)
DEVICE = {'device_id': None, 'boot_id': None, 'fw': None, 'last_seen_ms': None, 'rssi': None, 'backlog': 0,
          'received': 0, 'late': 0, 'duplicates': 0, 'first_seq': None, 'max_seq': None, 'gaps': 0,
          'latest': None, 'rr': None, 'rr_quality': 0.0, 'outages': []}
clients: dict = {}          # websocket -> user id
LAST_SMS = {}


async def broadcast(msg, to=None):
    """Send to every signed-in dashboard (to=None) or only to one user's dashboards."""
    data = json.dumps(clean(msg), default=float)
    for ws, uid in list(clients.items()):
        if to is not None and uid != to:
            continue
        try:
            await ws.send_text(data)
        except Exception:
            clients.pop(ws, None)

# ============================================================ API: device side
@app.get('/api/v1/time')
def api_time():
    return {'epoch_ms': now_ms()}


@app.post('/api/v1/telemetry')
async def telemetry(req: Request):
    body = await req.json()
    recv = now_ms()
    dev, boot = body.get('device_id', 'unknown'), int(body.get('boot_id') or 0)
    recs = body.get('records') or []
    if DEVICE['boot_id'] != boot or DEVICE['device_id'] != dev:      # new device or reboot
        DEVICE.update(device_id=dev, boot_id=boot, first_seq=None, max_seq=None, received=0, gaps=0)
    if DEVICE['last_seen_ms'] and recv - DEVICE['last_seen_ms'] > CONFIG['device_offline_seconds'] * 1000:
        DEVICE['outages'] = (DEVICE['outages'] + [[DEVICE['last_seen_ms'], recv]])[-20:]
    DEVICE.update(fw=body.get('fw'), rssi=body.get('rssi'), backlog=body.get('backlog', 0), last_seen_ms=recv)
    newest_up = max((r.get('uptime_ms') or 0) for r in recs) if recs else 0
    ack = None
    key = live_key()
    for r in recs:
        seq = int(r['seq'])
        t = r.get('t_ms') or (recv - (newest_up - (r.get('uptime_ms') or newest_up)))
        late = 1 if recv - t > 5000 else 0
        v, s = r.get('vitals', {}), r.get('sqi', {})
        with _db_lock:
            cur = db.execute('INSERT OR IGNORE INTO telemetry VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                             (dev, boot, seq, int(t), recv, late, v.get('hr_ecg'), v.get('hr_ppg'), v.get('spo2'),
                              v.get('skin_temp'), v.get('amb_temp'), v.get('rmssd'), v.get('sdnn'), s.get('ecg'),
                              s.get('ppg'), s.get('temp'), int(bool(s.get('leads_off'))), int(bool(s.get('finger'))),
                              int(bool(s.get('hr_agree')))))
            db.commit()
            inserted = cur.rowcount == 1
        if inserted:
            DEVICE['received'] += 1
            DEVICE['late'] += late
            DEVICE['first_seq'] = seq if DEVICE['first_seq'] is None else min(DEVICE['first_seq'], seq)
            DEVICE['max_seq'] = seq if DEVICE['max_seq'] is None else max(DEVICE['max_seq'], seq)
        else:
            DEVICE['duplicates'] += 1
        ack = seq if ack is None else max(ack, seq)
        if 'ecg' in r:                                              # live waveforms (newest record only)
            ts = t / 1000.0
            fs = r['ecg'].get('fs', 250)
            samples = r['ecg'].get('samples', [])
            t0 = ts - len(samples) / fs
            for i, val in enumerate(samples):
                ecg_ring.append((t0 + i / fs, val))
            for idx, amp in r['ecg'].get('rpeaks', []):
                beats.append((t0 + idx / fs, amp))
            ir = r.get('ppg', {}).get('ir', [])
            p0 = ts - len(ir) / 25.0
            for i, val in enumerate(ir):
                ppg_ring.append((p0 + i / 25.0, val))
            DEVICE['latest'] = dict(t_ms=t, vitals=v, sqi=s)
            await broadcast({'type': 'live', 'patient': key, 't_ms': t, 'ecg': {'fs': fs, 'samples': samples},
                             'ppg': {'fs': 25, 'ir': ir}, 'vitals': v, 'sqi': s,
                             'rr': DEVICE['rr'], 'rr_quality': DEVICE['rr_quality'],
                             'temp_core_est': (v.get('skin_temp') + CONFIG['temp_offset_c']) if v.get('skin_temp') else None},
                            to=owner(key))
    if DEVICE['max_seq'] is not None:
        DEVICE['gaps'] = (DEVICE['max_seq'] - DEVICE['first_seq'] + 1) - DEVICE['received']
    last = q('SELECT risk FROM predictions WHERE patient=? ORDER BY hour DESC LIMIT 1', (key,), one=True) if key else None
    act = q("SELECT severity FROM alerts WHERE patient=? AND end_ms IS NULL", (key or DEVICE_KEY,))
    sev = 'alarm' if any(a['severity'] == 'alarm' for a in act) else ('warn' if act else 'none')
    return {'ack_seq': ack, 'server_time_ms': recv, 'risk': last['risk'] if last else None, 'alert': sev}

# ============================================================ hourly aggregation + inference
def live_hourly():
    """Builds the model's hourly rows for the active monitoring session from per-second telemetry.
    Each second counts for a vital only if its signal-quality index is good enough."""
    s = active_session()
    if not s:
        return None
    H = CONFIG['seconds_per_model_hour'] * 1000
    t0 = s['start_ms']
    rows = q('SELECT t_ms, hr_ecg, hr_ppg, spo2, skin_temp, sqi_ecg, sqi_ppg, sqi_temp FROM telemetry WHERE t_ms >= ?', (t0,))
    if not rows:
        return None
    df = numeric_frame(rows)
    df['hour'] = ((df.t_ms - t0) // H).astype(int)
    df['HR'] = np.where(df.sqi_ecg.fillna(0) >= 0.5, df.hr_ecg, np.where(df.sqi_ppg.fillna(0) >= 0.5, df.hr_ppg, np.nan))
    df['O2Sat'] = np.where(df.sqi_ppg.fillna(0) >= 0.5, df.spo2, np.nan)
    df['Temp'] = np.where(df.sqi_temp.fillna(0) >= 0.5, df.skin_temp + CONFIG['temp_offset_c'], np.nan)
    rr = pd.DataFrame(q('SELECT t_ms, rr, quality FROM resp WHERE patient=? AND t_ms >= ?', (DEVICE_KEY, t0)))
    out = []
    for h in range(int(df.hour.max()) + 1):
        g = df[df.hour == h]
        cov = {k: float(g[k].notna().mean()) if len(g) else 0.0 for k in ('HR', 'O2Sat', 'Temp')}
        row = {'hour': h, 't_ms': int(t0 + h * H)}
        for k in ('HR', 'O2Sat', 'Temp'):
            row[k] = float(g[k].median()) if cov[k] >= 0.3 else np.nan      # too little good data -> missing
        if len(rr):
            rg = rr[((rr.t_ms - t0) // H == h) & (rr.quality >= 0.3)]
            row['Resp'] = float(rg.rr.median()) if len(rg) else np.nan
            cov['Resp'] = float(len(rg) > 0)
        else:
            row['Resp'] = np.nan; cov['Resp'] = 0.0
        row['coverage'] = cov
        out.append(row)
    return out


def predict_patient(patient, hourly_rows):
    if MODEL is None or not hourly_rows:
        return None
    p = patient_info(patient) or {}
    df = pd.DataFrame([{k: r.get(k) for k in ('HR', 'O2Sat', 'Temp', 'Resp')} for r in hourly_rows], dtype=float)
    df['Age'] = float(p['age']) if p.get('age') is not None else np.nan
    df['Gender'] = float(p['gender']) if p.get('gender') is not None else np.nan
    preds = MODEL.predict_hours(df)
    for r, pr in zip(hourly_rows, preds):
        cov = r.get('coverage') or {}
        missing = [k for k in ('HR', 'O2Sat', 'Temp', 'Resp') if r.get(k) is None or (isinstance(r.get(k), float) and np.isnan(r.get(k)))]
        quality = 'good' if not missing else ('partial' if len(missing) <= 2 else 'poor')
        x('INSERT OR REPLACE INTO predictions (patient, hour, t_ms, risk, risk_lo, risk_hi, quality, reasons, horizons) '
          'VALUES (?,?,?,?,?,?,?,?,?)',
          (patient, r['hour'], r['t_ms'], pr['risk'], pr['risk_lo'], pr['risk_hi'],
           json.dumps({'level': quality, 'missing': missing, 'coverage': cov}),
           json.dumps({'raising': pr['raising'], 'lowering': pr['lowering']}), json.dumps(pr['horizons'])))
        x('INSERT OR REPLACE INTO hourly VALUES (?,?,?,?,?,?,?,?,?,?)',
          (patient, r['hour'], r['t_ms'], _n(r.get('HR')), _n(r.get('O2Sat')), _n(r.get('Temp')), _n(r.get('Resp')),
           _n(r.get('SBP')), json.dumps(cov), r.get('label')))
    last = preds[-1]
    return dict(last, hour=hourly_rows[-1]['hour'], t_ms=hourly_rows[-1]['t_ms'])


def _n(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def clean(o):
    """Recursively replace NaN / inf with None so every API response is valid JSON."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def outlook_for(horizons):
    """Adds the current thresholds + base rates to stored per-horizon risks and words them for patients."""
    if not horizons:
        return []
    thr = horizon_thresholds()
    hz = {h: dict(v, threshold=thr.get(h, v.get('threshold')),
                  prevalence=MODEL.horizons[int(h)].prevalence if MODEL and int(h) in MODEL.horizons else None)
          for h, v in horizons.items()}
    return explain.outlook(hz, CONFIG['warn_fraction'])

# ============================================================ alerts
ACTIVE = {}   # (patient, kind) -> alert id


def set_alert(patient, kind, active, severity='warn', message='', value=None, t=None):
    key = (patient, kind)
    t = t or now_ms()
    if active and key not in ACTIVE:
        aid = x('INSERT INTO alerts (patient, kind, severity, message, value, start_ms) VALUES (?,?,?,?,?,?)',
                (patient, kind, severity, message, value, t))
        ACTIVE[key] = aid
        return {'type': 'alert', 'event': 'raised', 'alert': q('SELECT * FROM alerts WHERE id=?', (aid,), one=True)}
    if active and key in ACTIVE:
        prev = q('SELECT severity FROM alerts WHERE id=?', (ACTIVE[key],), one=True)
        x('UPDATE alerts SET severity=?, message=?, value=? WHERE id=?', (severity, message, value, ACTIVE[key]))
        if prev and prev['severity'] != 'alarm' and severity == 'alarm':
            return {'type': 'alert', 'event': 'escalated', 'alert': q('SELECT * FROM alerts WHERE id=?', (ACTIVE[key],), one=True)}
    if not active and key in ACTIVE:
        aid = ACTIVE.pop(key)
        x('UPDATE alerts SET end_ms=? WHERE id=?', (t, aid))
        return {'type': 'alert', 'event': 'cleared', 'alert': q('SELECT * FROM alerts WHERE id=?', (aid,), one=True)}
    return None


def risk_alerts(patient, t=None):
    preds = q('SELECT hour, risk FROM predictions WHERE patient=? ORDER BY hour DESC LIMIT ?',
              (patient, int(CONFIG['alarm_persist_hours'])))
    thr = alarm_threshold()
    events = []
    if not preds:
        return events
    latest = preds[0]['risk']
    alarm = len(preds) >= CONFIG['alarm_persist_hours'] and all(p['risk'] >= thr for p in preds)
    warn = not alarm and latest >= CONFIG['warn_fraction'] * thr
    e = set_alert(patient, 'sepsis_risk', alarm or warn, 'alarm' if alarm else 'warn',
                  (f'Sepsis risk {latest*100:.1f}% (≥ {thr*100:.1f}% for {CONFIG["alarm_persist_hours"]} h)' if alarm
                   else f'Sepsis risk rising: {latest*100:.1f}%'), latest, t)
    if e:
        events.append(e)
    return events


def recent_vitals(seconds):
    """Medians of good-quality seconds over the last N seconds (what the alerts and the patient view use)."""
    win = q('SELECT * FROM telemetry WHERE t_ms >= ?', (now_ms() - seconds * 1000,))
    if not win:
        return None
    df = numeric_frame(win)
    hr = df.hr_ecg.where(df.sqi_ecg >= 0.5).median()
    if np.isnan(hr):
        hr = df.hr_ppg.where(df.sqi_ppg >= 0.5).median()
    return dict(hr=_n(hr), spo2=_n(df.spo2.where(df.sqi_ppg >= 0.5).median()),
                temp_core=_n((df.skin_temp.where(df.sqi_temp >= 0.5) + CONFIG['temp_offset_c']).median()),
                skin=_n(df.skin_temp.where(df.sqi_temp >= 0.5).median()),
                hrv=_n(df.rmssd.where(df.sqi_ecg >= 0.5).median()),
                rr=DEVICE['rr'], rr_quality=DEVICE['rr_quality'],
                leads_off=float(df.leads_off.mean()), no_finger=float((df.finger == 0).mean()), n=len(df))


LAST_ALERT_KEY = {'k': None}


def vital_and_device_alerts():
    events = []
    key = live_key() or DEVICE_KEY
    old = LAST_ALERT_KEY['k']
    if old and old != key:                     # monitoring session changed: close the old session's device/vital alerts
        for (p, kind) in [k for k in ACTIVE if k[0] == old and k[1] in DEVICE_KINDS]:
            events.append(set_alert(p, kind, False))
    LAST_ALERT_KEY['k'] = key
    now = now_ms()
    offline = DEVICE['last_seen_ms'] is None or now - DEVICE['last_seen_ms'] > CONFIG['device_offline_seconds'] * 1000
    if DEVICE['last_seen_ms'] is not None:
        events.append(set_alert(key, 'device_offline', offline, 'warn', 'Monitoring node offline: data is buffered on the device'))
    rv = recent_vitals(CONFIG['vital_alert_seconds'])
    if rv and not offline:
        c = CONFIG
        events.append(set_alert(key, 'ecg_leads_off', rv['leads_off'] > 0.8, 'warn', 'ECG electrode off: check RA / LA / RL stickers'))
        events.append(set_alert(key, 'no_finger', rv['no_finger'] > 0.8, 'warn', 'No finger on the SpO₂ sensor'))
        # severity 'alarm' = far outside the normal range (NEWS2 sub-score 3); these can trigger a phone alert
        sev = lambda kind, v: 'alarm' if explain.score(kind, v) == 3 else 'warn'
        hr, spo2, tc = rv['hr'], rv['spo2'], rv['temp_core']
        events.append(set_alert(key, 'hr', hr is not None and (hr > c['hr_high'] or hr < c['hr_low'] or explain.score('hr', hr) == 3),
                                sev('hr', hr), f'Heart rate {hr or 0:.0f} bpm outside {c["hr_low"]}-{c["hr_high"]}', hr))
        events.append(set_alert(key, 'spo2', spo2 is not None and (spo2 < c['spo2_low'] or explain.score('spo2', spo2) == 3),
                                sev('spo2', spo2), f'SpO₂ {spo2 or 0:.0f}% below {c["spo2_low"]}%', spo2))
        events.append(set_alert(key, 'temp', tc is not None and (tc > c['temp_high'] or tc < c['temp_low']), sev('temp', tc),
                                f'Estimated temperature {tc or 0:.1f} °C outside {c["temp_low"]}-{c["temp_high"]}', tc))
        rr = rv['rr'] if rv['rr_quality'] >= 0.3 else None
        events.append(set_alert(key, 'rr', rr is not None and (rr > c['rr_high'] or explain.score('rr', rr) == 3), sev('rr', rr),
                                f'Breathing rate {rr or 0:.0f}/min above {c["rr_high"]}', rr))
    return [e for e in events if e]


async def emit_alerts(events):
    for e in events:
        a = e['alert']
        await broadcast(e, to=owner(a['patient']))
        if e['event'] in ('raised', 'escalated') and a['severity'] == 'alarm':
            asyncio.create_task(notify_alarm(a))


async def notify_alarm(a):
    """Text the patient's phone (number from sign-up) for alarm-level alerts. Logged in 'notifications'."""
    c = CONFIG
    if not c['sms_enabled'] or not c['sms_gateway_url']:
        return
    key = a['patient']
    demo = key.startswith('replay:')
    if demo:
        uid = (replay_meta(key[7:]) or {}).get('started_by') if c['sms_for_replays'] else None
        name = (replay_meta(key[7:]) or {}).get('name', key)
    else:
        uid = owner(key)
        name = None
    u = q('SELECT * FROM users WHERE id=?', (uid,), one=True) if uid else None
    if not u or not u['sms'] or not u['phone']:
        return
    if now_ms() - LAST_SMS.get(uid, 0) < float(c['sms_min_interval_min']) * 60000:
        x('INSERT INTO notifications (alert_id, user_id, phone, sent_ms, ok, detail) VALUES (?,?,?,?,?,?)',
          (a['id'], uid, u['phone'], now_ms(), 0, 'not sent: rate limit (one SMS per patient per '
                                                  f'{c["sms_min_interval_min"]} min)'))
        return
    LAST_SMS[uid] = now_ms()
    ok, detail = await asyncio.to_thread(notify.send_sms, c, u['phone'], notify.alarm_text(name or u['name'], a['message'], demo))
    x('INSERT INTO notifications (alert_id, user_id, phone, sent_ms, ok, detail) VALUES (?,?,?,?,?,?)',
      (a['id'], uid, u['phone'], now_ms(), int(ok), detail))
    print(f'SMS to {u["phone"]}: {"sent" if ok else "FAILED"} ({detail})')
    await broadcast({'type': 'notification', 'alert_id': a['id'], 'ok': ok, 'detail': detail}, to=uid)


def device_status():
    now = now_ms()
    d = dict(DEVICE)
    d.pop('latest', None)
    d['online'] = bool(d['last_seen_ms']) and now - d['last_seen_ms'] <= CONFIG['device_offline_seconds'] * 1000
    d['seconds_since_seen'] = None if not d['last_seen_ms'] else (now - d['last_seen_ms']) / 1000
    recent = q('SELECT AVG(sqi_ecg) e, AVG(sqi_ppg) p, AVG(sqi_temp) t, AVG(leads_off) lo, AVG(finger) f, AVG(hr_agree) ag, '
               'COUNT(*) n FROM telemetry WHERE t_ms >= ?', (now - 60000,), one=True)
    d['quality_60s'] = recent
    d['loss_pct'] = (100.0 * d['gaps'] / max(1, d['gaps'] + d['received'])) if d['received'] else None
    s = active_session()
    d['monitoring'] = dict(session=s['id'], user_id=s['user_id'], start_ms=s['start_ms']) if s else None
    return d

# ============================================================ background loop
async def worker():
    last_model = 0
    while True:
        await asyncio.sleep(2)
        try:
            events = []
            # breathing rate from the last 60 s of ECG beats + PPG
            if beats or ppg_ring:
                bt = list(beats); pr = list(ppg_ring)
                est = respiration.estimate([b[0] for b in bt], [b[1] for b in bt], [p[0] for p in pr], [p[1] for p in pr])
                DEVICE['rr'], DEVICE['rr_quality'] = est['rr'], est['quality']
                if est['rr'] is not None and DEVICE['latest']:
                    x('INSERT INTO resp VALUES (?,?,?,?,?)', (DEVICE_KEY, now_ms(), est['rr'], est['quality'], json.dumps(est['parts'])))
            key = live_key()
            if key and time.time() - last_model > 10:
                last_model = time.time()
                rows = await asyncio.to_thread(live_hourly)
                pred = await asyncio.to_thread(predict_patient, key, rows) if rows else None
                if pred:
                    events += risk_alerts(key)
                    await broadcast({'type': 'prediction', 'patient': key, **pred, 'threshold': alarm_threshold()}, to=owner(key))
            events += vital_and_device_alerts()
            await emit_alerts(events)
            await broadcast({'type': 'status', 'device': device_status()})
        except Exception as e:  # keep the service alive whatever happens
            print('worker error:', repr(e))


def _quiet_resets(loop, ctx):
    """Windows' asyncio prints a ConnectionResetError traceback whenever a browser drops a socket; it is harmless."""
    if isinstance(ctx.get('exception'), ConnectionResetError):
        return
    loop.default_exception_handler(ctx)


@app.on_event('startup')
async def startup():
    asyncio.get_running_loop().set_exception_handler(_quiet_resets)
    current = live_key() or DEVICE_KEY
    for a in q('SELECT id, patient, kind FROM alerts WHERE end_ms IS NULL'):
        if a['kind'] in DEVICE_KINDS and a['patient'] != current:   # left over from an earlier session
            x('UPDATE alerts SET end_ms=? WHERE id=?', (now_ms(), a['id']))
        else:
            ACTIVE[(a['patient'], a['kind'])] = a['id']
    LAST_ALERT_KEY['k'] = current
    asyncio.create_task(worker())

# ============================================================ monitoring sessions
@app.post('/api/v1/monitor/start')
def api_monitor_start(req: Request):
    """Link the wearable to the signed-in patient from now on (ends any other running session)."""
    u = current_user(req)
    x('UPDATE monitor_sessions SET end_ms=? WHERE end_ms IS NULL', (now_ms(),))
    sid = x('INSERT INTO monitor_sessions (user_id, start_ms) VALUES (?,?)', (u['id'], now_ms()))
    ACTIVE_SESSION['id'] = sid
    kv_set('active_session', sid)
    return q('SELECT * FROM monitor_sessions WHERE id=?', (sid,), one=True)


@app.post('/api/v1/monitor/stop')
def api_monitor_stop(req: Request):
    u = current_user(req)
    s = active_session()
    if s and s['user_id'] == u['id']:
        x('UPDATE monitor_sessions SET end_ms=? WHERE id=?', (now_ms(), s['id']))
        ACTIVE_SESSION['id'] = None
        kv_set('active_session', None)
    return {'ok': True}


@app.get('/api/v1/me/now')
def api_me_now(req: Request):
    """Everything the patient home screen needs: plain-language status, current vitals, 6/12/24 h outlook."""
    u = current_user(req)
    s = active_session()
    mine = bool(s and s['user_id'] == u['id'])
    dev = device_status()
    rv = recent_vitals(CONFIG['vital_alert_seconds']) if dev['online'] else None
    hints = []
    if rv:
        if rv['leads_off'] > 0.5:
            hints.append('The ECG stickers are not making contact. Press them firmly onto the skin.')
        if rv['no_finger'] > 0.5:
            hints.append('Place a finger gently on the clip sensor and keep it still.')
    elif dev['device_id']:
        hints.append('The sensor box is not sending data. Check that it is switched on and on the same Wi-Fi.')
    else:
        hints.append('No sensor box has connected yet. Switch it on; it connects automatically.')
    out = dict(monitoring=mine, someone_else_monitoring=bool(s and not mine),
               session=dict(s, key=f"sess:{s['id']}") if mine else None,
               device=dict(online=dev['online'], seen=bool(dev['device_id']), simulated=dev['device_id'] == 'simulator'),
               now=explain.now(rv if mine or not s else None, {'hints': hints}),
               vitals=rv if (mine or not s) else None, outlook=[], model_loaded=MODEL is not None,
               seconds_per_model_hour=CONFIG['seconds_per_model_hour'], prediction=None)
    if mine:
        last = q('SELECT * FROM predictions WHERE patient=? ORDER BY hour DESC LIMIT 1', (f"sess:{s['id']}",), one=True)
        if last:
            hz = json.loads(last['horizons']) if last.get('horizons') else {'6': dict(risk=last['risk'], lo=last['risk_lo'], hi=last['risk_hi'])}
            out['outlook'] = outlook_for(hz)
            out['prediction'] = dict(hour=last['hour'], quality=json.loads(last['quality']), reasons=json.loads(last['reasons']),
                                     risk=last['risk'], risk_lo=last['risk_lo'], risk_hi=last['risk_hi'])
    return clean(out)


@app.get('/api/v1/me/history')
def api_me_history(req: Request):
    u = current_user(req)
    sess = q('SELECT * FROM monitor_sessions WHERE user_id=? ORDER BY id DESC', (u['id'],))
    ups = q('SELECT * FROM uploads WHERE user_id=? ORDER BY id DESC', (u['id'],))
    for items, kind in ((sess, 'sess'), (ups, 'up')):
        for r in items:
            r['key'] = f"{kind}:{r['id']}"
            last = q('SELECT hour, risk, horizons FROM predictions WHERE patient=? ORDER BY hour DESC LIMIT 1', (r['key'],), one=True)
            r['hours'] = (last['hour'] + 1) if last else 0
            r['outlook'] = outlook_for(json.loads(last['horizons'])) if last and last.get('horizons') else []
    return clean({'sessions': sess, 'uploads': ups})

# ============================================================ uploads (recorded data)
ALIASES = {'hr': 'HR', 'heart_rate': 'HR', 'heartrate': 'HR', 'pulse': 'HR', 'o2sat': 'O2Sat', 'spo2': 'O2Sat',
           'sao2': 'O2Sat', 'temp': 'Temp', 'temperature': 'Temp', 'resp': 'Resp', 'rr': 'Resp',
           'respiratory_rate': 'Resp', 'breathing_rate': 'Resp', 'sbp': 'SBP', 'age': 'Age', 'gender': 'Gender',
           'sex': 'Gender', 'sepsislabel': 'SepsisLabel', 'label': 'SepsisLabel'}


@app.post('/api/v1/uploads')
async def api_upload(req: Request):
    """Recorded hourly vitals -> the same feature builder + models as live monitoring.
    Body: {name, content (CSV or PhysioNet .psv text, one row per hour, oldest first), age?, gender?}"""
    u = current_user(req)
    b = await req.json()
    try:
        df = pd.read_csv(io.StringIO(b.get('content') or ''), sep=None, engine='python')
    except Exception as e:
        raise HTTPException(400, f'could not read the file: {e}')
    df = df.rename(columns={c: ALIASES.get(str(c).strip().lower().replace(' ', '_'), c) for c in df.columns})
    vit = [k for k in ('HR', 'O2Sat', 'Temp', 'Resp') if k in df]
    if not vit:
        raise HTTPException(400, 'no vital-sign columns found: need at least one of HR, O2Sat (SpO2), Temp, Resp')
    if len(df) > 5000:
        raise HTTPException(400, 'file too long (max 5000 hourly rows)')
    for k in vit + [c for c in ('SBP', 'SepsisLabel') if c in df]:
        df[k] = pd.to_numeric(df[k], errors='coerce')
    if df[vit].notna().sum().sum() == 0:
        raise HTTPException(400, 'the vital-sign columns are empty')
    # age / sex: from the file if it has them, else from the form, else from the profile
    from_file = lambda c: df[c].dropna().iloc[0] if c in df and df[c].notna().any() else None
    pick = lambda *vs: next(v for v in vs if v not in (None, ''))
    age = float(pick(from_file('Age'), b.get('age'), u['age']))
    g = pick(from_file('Gender'), b.get('gender'), u['gender'])
    if isinstance(g, str):
        g = 1 if g.strip().lower() in ('1', 'm', 'male') else 0
    lab = df['SepsisLabel'].fillna(0).astype(int).values if 'SepsisLabel' in df else None
    ls = int(np.argmax(lab)) if lab is not None and lab.max() == 1 else None
    uid = x('INSERT INTO uploads (user_id, name, created_ms, n_hours, age, gender, label_start_hour, onset_hour) VALUES (?,?,?,?,?,?,?,?)',
            (u['id'], str(b.get('name') or 'recording')[:120], now_ms(), len(df), float(age), int(g), ls, (ls + 6) if ls is not None else None))
    key = f'up:{uid}'
    t0 = now_ms()
    rows = [{'hour': i, 't_ms': t0 + i * 3600000, **{k: (float(df[k].iloc[i]) if k in df and pd.notna(df[k].iloc[i]) else np.nan)
                                                   for k in ('HR', 'O2Sat', 'Temp', 'Resp', 'SBP')},
             'label': (int(lab[i]) if lab is not None else None), 'coverage': {}} for i in range(len(df))]
    if MODEL is None:
        raise HTTPException(503, 'no trained model loaded - run 02_ML/train.py')
    await asyncio.to_thread(predict_patient, key, rows)
    return {'key': key, 'hours': len(rows)}


@app.delete('/api/v1/uploads/{uid}')
def api_upload_delete(uid: int, req: Request):
    u = current_user(req)
    key = f'up:{uid}'
    require_access(u, key)
    for t in ('predictions', 'hourly'):
        x(f'DELETE FROM {t} WHERE patient=?', (key,))
    x('DELETE FROM uploads WHERE id=?', (uid,))
    return {'ok': True}

# ============================================================ replay of held-out dataset patients
REPLAYS = {}


def replay_cases():
    out = []
    if os.path.isdir(REPLAY_DIR):
        for f in sorted(os.listdir(REPLAY_DIR)):
            if f.endswith('.json'):
                d = json.load(open(os.path.join(REPLAY_DIR, f)))
                out.append(dict(pid=d['pid'], hours=len(d['hours']), age=d['age'], gender=d['gender'],
                                septic=d['onset_hour'] is not None, onset_hour=d['onset_hour'],
                                label_start_hour=d['label_start_hour'], source=d['source'],
                                running=REPLAYS.get(d['pid']) == 'running'))
    return out


async def run_replay(pid, speed, user_id):
    d = json.load(open(os.path.join(REPLAY_DIR, f'{pid}.json')))
    key = f'replay:{pid}'
    replay_meta(pid)['started_by'] = user_id
    x('DELETE FROM predictions WHERE patient=?', (key,)); x('DELETE FROM hourly WHERE patient=?', (key,))
    x('UPDATE alerts SET end_ms=? WHERE patient=? AND end_ms IS NULL', (now_ms(), key))
    for k in [k for k in ACTIVE if k[0] == key]:
        ACTIVE.pop(k)
    rows = []
    t0 = now_ms()
    for h in d['hours']:
        if REPLAYS.get(pid) != 'running':
            break
        rows.append({'hour': h['hour'], 't_ms': t0 + h['hour'] * 3600 * 1000, 'HR': _nan(h.get('HR')),
                     'O2Sat': _nan(h.get('O2Sat')), 'Temp': _nan(h.get('Temp')), 'Resp': _nan(h.get('Resp')),
                     'SBP': _nan(h.get('SBP')), 'label': h.get('label'), 'coverage': {}})
        pred = await asyncio.to_thread(predict_patient, key, rows)
        events = risk_alerts(key, rows[-1]['t_ms'])
        await broadcast({'type': 'prediction', 'patient': key, **pred, 'threshold': alarm_threshold(),
                         'vitals': {k: _n(rows[-1][k]) for k in ('HR', 'O2Sat', 'Temp', 'Resp', 'SBP')},
                         'onset_hour': d['onset_hour'], 'label_start_hour': d['label_start_hour']})
        await emit_alerts(events)
        await asyncio.sleep(speed)
    REPLAYS[pid] = 'done'


def _nan(v):
    return np.nan if v is None else float(v)


@app.get('/api/v1/replay/cases')
def api_replay_cases(req: Request):
    current_user(req)
    return replay_cases()


@app.post('/api/v1/replay/{pid}/start')
async def api_replay_start(pid: str, req: Request, speed: Optional[float] = None):
    u = current_user(req)
    if not replay_meta(pid):
        raise HTTPException(404, 'unknown replay case')
    if MODEL is None:
        raise HTTPException(503, 'no trained model loaded')
    REPLAYS[pid] = 'running'
    asyncio.create_task(run_replay(pid, speed or CONFIG['replay_seconds_per_hour'], u['id']))
    return {'patient': f'replay:{pid}'}


@app.post('/api/v1/replay/{pid}/stop')
def api_replay_stop(pid: str, req: Request):
    current_user(req)
    REPLAYS[pid] = 'stopped'
    return {'ok': True}


@app.get('/api/v1/replay/{pid}.csv', response_class=PlainTextResponse)
def api_replay_csv(pid: str, req: Request):
    """A held-out recorded patient as an upload-ready CSV (real PhysioNet data, for trying the upload feature)."""
    current_user(req)
    path = os.path.join(REPLAY_DIR, f'{pid}.json')
    if not os.path.exists(path):
        raise HTTPException(404, 'unknown replay case')
    d = json.load(open(path))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['hour', 'HR', 'O2Sat', 'Temp', 'Resp', 'SBP', 'Age', 'Gender', 'SepsisLabel'])
    for h in d['hours']:
        w.writerow([h['hour']] + ['' if h.get(k) is None else h[k] for k in ('HR', 'O2Sat', 'Temp', 'Resp', 'SBP')] +
                   [d['age'], d['gender'], h.get('label', '')])
    return PlainTextResponse(buf.getvalue(), headers={'Content-Disposition': f'attachment; filename="physionet_B_{pid}.csv"'})

# ============================================================ API: dashboard side
@app.get('/api/v1/patients/{pid}/series')
def api_series(pid: str, req: Request, minutes: int = 30):
    """Per-second vitals of a monitoring session for the trend charts, downsampled to <= 600 points."""
    u = current_user(req)
    if not pid.startswith('sess:'):
        return {'vitals': [], 'resp': []}
    info = require_access(u, pid)
    end = info['end_ms'] or now_ms()
    since = max(info['start_ms'], end - minutes * 60000)
    rows = q('SELECT t_ms, hr_ecg, hr_ppg, spo2, skin_temp, sqi_ecg, sqi_ppg, sqi_temp, rmssd, leads_off, finger, late '
             'FROM telemetry WHERE t_ms >= ? AND t_ms <= ? ORDER BY t_ms', (since, end))
    step = max(1, len(rows) // 600)
    rr = q('SELECT t_ms, rr, quality FROM resp WHERE patient=? AND t_ms >= ? AND t_ms <= ? ORDER BY t_ms', (DEVICE_KEY, since, end))
    for r in rows:
        r['temp_core_est'] = r['skin_temp'] + CONFIG['temp_offset_c'] if r['skin_temp'] else None
    return {'vitals': rows[::step], 'resp': rr[::max(1, len(rr) // 300)], 'start_ms': since, 'end_ms': end}


@app.get('/api/v1/patients/{pid}/hourly')
def api_hourly(pid: str, req: Request):
    u = current_user(req)
    info = require_access(u, pid)
    h = q('SELECT * FROM hourly WHERE patient=? ORDER BY hour', (pid,))
    p = q('SELECT * FROM predictions WHERE patient=? ORDER BY hour', (pid,))
    for r in p:
        r['reasons'] = json.loads(r['reasons']); r['quality'] = json.loads(r['quality'])
        r['horizons'] = json.loads(r['horizons']) if r.get('horizons') else None
    for r in h:
        r['coverage'] = json.loads(r['coverage']) if r['coverage'] else {}
    last = p[-1] if p else None
    return clean({'hourly': h, 'predictions': p, 'threshold': alarm_threshold(), 'thresholds': horizon_thresholds(),
                  'patient': info, 'outlook': outlook_for(last['horizons']) if last and last['horizons'] else [],
                  'seconds_per_model_hour': CONFIG['seconds_per_model_hour'] if pid.startswith('sess:') else 3600})


@app.get('/api/v1/alerts')
def api_alerts(req: Request, limit: int = 100):
    u = current_user(req)
    keys = user_keys(u['id']) + [DEVICE_KEY]
    marks = ','.join('?' * len(keys))
    rows = q(f"SELECT * FROM alerts WHERE patient IN ({marks}) OR patient LIKE 'replay:%' ORDER BY id DESC LIMIT ?",
             (*keys, limit))
    sms = {r['alert_id']: r for r in q('SELECT * FROM notifications WHERE user_id=? ORDER BY id', (u['id'],))}
    for r in rows:
        r['sms'] = sms.get(r['id'])
    return rows


@app.post('/api/v1/alerts/{aid}/ack')
async def api_ack(aid: int, req: Request):
    current_user(req)
    x('UPDATE alerts SET acked=1, acked_ms=? WHERE id=?', (now_ms(), aid))
    a = q('SELECT * FROM alerts WHERE id=?', (aid,), one=True)
    await broadcast({'type': 'alert', 'event': 'acked', 'alert': a}, to=owner(a['patient']) if a else None)
    return a


@app.get('/api/v1/config')
def api_config(req: Request):
    current_user(req)
    c = {k: ('••••••' if k in SECRET and v else v) for k, v in CONFIG.items()}
    return dict(c, model_threshold=MODEL.threshold if MODEL else None, effective_alarm_threshold=alarm_threshold(),
                horizon_thresholds=horizon_thresholds())


@app.put('/api/v1/config')
async def api_config_put(req: Request):
    current_user(req)
    b = await req.json()
    for k, v in b.items():
        if k in DEFAULT_CONFIG and not (k in SECRET and v == '••••••'):
            CONFIG[k] = v
    save_config()
    return api_config(req)


@app.post('/api/v1/sms/test')
async def api_sms_test(req: Request):
    u = current_user(req)
    ok, detail = await asyncio.to_thread(notify.send_sms, CONFIG, u['phone'],
                                         f'SeptiSense test message for {u["name"]}: phone alerts are working.')
    x('INSERT INTO notifications (alert_id, user_id, phone, sent_ms, ok, detail) VALUES (?,?,?,?,?,?)',
      (None, u['id'], u['phone'], now_ms(), int(ok), 'test: ' + detail))
    return {'ok': ok, 'detail': detail, 'phone': u['phone']}


@app.post('/api/v1/temp/calibrate')
async def api_temp_cal(req: Request):
    """Enter a reference thermometer reading; the offset becomes reference - current wrist skin temperature."""
    current_user(req)
    b = await req.json()
    rows = q('SELECT skin_temp FROM telemetry WHERE t_ms >= ? AND sqi_temp >= 0.5', (now_ms() - 60000,))
    skins = [r['skin_temp'] for r in rows if r['skin_temp']]
    if not skins:
        raise HTTPException(400, 'no stable skin temperature in the last minute')
    CONFIG['temp_offset_c'] = round(float(b['reference_c']) - float(np.median(skins)), 2)
    save_config()
    return {'temp_offset_c': CONFIG['temp_offset_c'], 'skin_median': float(np.median(skins))}


@app.get('/api/v1/device')
def api_device(req: Request):
    current_user(req)
    return device_status()


@app.get('/api/v1/model')
def api_model(req: Request):
    """Deployed model meta + the full held-out evaluation (results/metrics.json written by train.py)."""
    current_user(req)
    path = os.path.join(RESULTS_DIR, 'metrics.json')
    evaluation = clean(json.load(open(path))) if os.path.exists(path) else None
    if MODEL is None:
        return {'loaded': False, 'evaluation': evaluation}
    m = dict(MODEL.meta)
    m.pop('calibration', None)
    m['horizons'] = {h: {k: v for k, v in d.items() if k != 'calibration'} for h, d in (m.get('horizons') or {}).items()}
    return clean(dict(loaded=True, evaluation=evaluation, **m))


@app.get('/api/v1/export/{pid}.csv', response_class=PlainTextResponse)
def api_export(pid: str, req: Request):
    u = current_user(req)
    require_access(u, pid)
    rows = q('SELECT h.*, p.risk, p.risk_lo, p.risk_hi, p.horizons FROM hourly h LEFT JOIN predictions p ON h.patient=p.patient '
             'AND h.hour=p.hour WHERE h.patient=? ORDER BY h.hour', (pid,))
    for r in rows:
        hz = json.loads(r.pop('horizons') or '{}')
        for h, v in hz.items():
            r[f'risk_{h}h'] = v.get('risk')
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return PlainTextResponse(buf.getvalue(), headers={'Content-Disposition': f'attachment; filename="septisense_{pid.replace(":", "_")}.csv"'})


@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    u = user_from_token(ws.cookies.get(COOKIE))
    await ws.accept()
    if not u:
        await ws.close(code=4401)     # dashboard shows the sign-in screen
        return
    clients[ws] = u['id']
    try:
        await ws.send_text(json.dumps(clean({'type': 'status', 'device': device_status()}), default=float))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.pop(ws, None)


if os.path.isdir(DASH_DIR):
    app.mount('/static', StaticFiles(directory=DASH_DIR), name='static')
# evaluation plots (roc_pr.png, calibration_alerts_features.png); check_dir=False so it works before train.py has run
app.mount('/results', StaticFiles(directory=RESULTS_DIR, check_dir=False), name='results')


@app.get('/')
def index():
    return FileResponse(os.path.join(DASH_DIR, 'index.html'))


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()


if __name__ == '__main__':
    import uvicorn
    ip = lan_ip()
    print('=' * 64)
    print(f'  SeptiSense backend\n  Dashboard:        http://localhost:8000\n  ESP32 SERVER_HOST: "{ip}"  (put this in config.h)')
    print('  Stop: Ctrl+C')
    print('=' * 64)
    # the dashboard keeps a WebSocket open; without a timeout uvicorn waits for it forever on Ctrl+C
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='warning', timeout_graceful_shutdown=2)

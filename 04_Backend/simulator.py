"""
SeptiSense device simulator - sends EXACTLY the JSON the ESP32 node sends (1 record/s, ECG + PPG waveforms,
sequence numbers, offline buffering). Used to test the backend and dashboard without hardware.
All signals are SYNTHETIC and labelled as such in the dashboard device panel (device_id 'simulator').

  python3 simulator.py                         # healthy adult, steady
  python3 simulator.py --outage 30:45          # Wi-Fi drop from t=30 s to t=45 s (records buffered, then re-sent)
  python3 simulator.py --leadoff 60:70         # an ECG sticker comes off
  python3 simulator.py --deteriorate 120       # vitals drift toward a septic picture over 120 s (scenario test)
"""
import argparse, json, math, random, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument('--host', default='http://localhost:8000')
p.add_argument('--hr', type=float, default=74)
p.add_argument('--rr', type=float, default=15)
p.add_argument('--spo2', type=float, default=98)
p.add_argument('--skin', type=float, default=33.6)
p.add_argument('--outage', default='')
p.add_argument('--leadoff', default='')
p.add_argument('--nofinger', default='')
p.add_argument('--deteriorate', type=float, default=0, help='seconds to drift to HR 118, RR 26, SpO2 92, skin +1.6 C')
p.add_argument('--seconds', type=int, default=0, help='stop after N seconds (0 = forever)')
a = p.parse_args()


def window(spec):
    if not spec:
        return None
    s, e = spec.split(':')
    return float(s), float(e)


OUT, LO, NF = window(a.outage), window(a.leadoff), window(a.nofinger)
inside = lambda w, t: w is not None and w[0] <= t < w[1]
boot = random.randint(1, 2**31)
seq, backlog, start = 0, [], time.time()
ecg_phase = 0.0
beat_times = []


def ecg_wave(phase):
    """One heartbeat as a sum of Gaussians (P, Q, R, S, T), phase in [0, 1)."""
    comp = [(0.15, 0.025, 120), (0.245, 0.010, -150), (0.27, 0.012, 1200), (0.295, 0.010, -250), (0.55, 0.05, 260)]
    return sum(A * math.exp(-((phase - m) ** 2) / (2 * s * s)) for m, s, A in comp)


def post(body):
    req = urllib.request.Request(a.host + '/api/v1/telemetry', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.loads(r.read())


print('simulating SeptiSense node (synthetic signals) ->', a.host)
while True:
    t_rel = time.time() - start
    if a.seconds and t_rel > a.seconds:
        break
    k = min(1.0, t_rel / a.deteriorate) if a.deteriorate else 0.0
    hr = a.hr + (118 - a.hr) * k + random.gauss(0, 1.0)
    rr_br = a.rr + (26 - a.rr) * k
    spo2 = a.spo2 + (92 - a.spo2) * k + random.gauss(0, 0.4)
    skin = a.skin + 1.6 * k + random.gauss(0, 0.05)
    leads_off, finger = inside(LO, t_rel), not inside(NF, t_rel)
    # ---- one second of ECG at 250 Hz with breathing modulation (amplitude + RSA)
    samples, peaks = [], []
    for i in range(250):
        ts = t_rel + i / 250
        resp = math.sin(2 * math.pi * rr_br / 60 * ts)
        inst_hr = hr * (1 + 0.04 * resp)
        prev = ecg_phase
        ecg_phase = (ecg_phase + inst_hr / 60 / 250) % 1.0
        if prev < 0.27 <= ecg_phase or (prev > ecg_phase and ecg_phase >= 0.27):
            peaks.append([i, int(1200 * (1 + 0.12 * resp))])
        v = ecg_wave(ecg_phase) * (1 + 0.12 * resp) + random.gauss(0, 15)
        samples.append(0 if leads_off else int(v))
    if leads_off:
        peaks = []
    beat_times += [t_rel + pk[0] / 250 for pk in peaks]
    beat_times = [b for b in beat_times if b > t_rel - 60]
    rr_s = [beat_times[j + 1] - beat_times[j] for j in range(len(beat_times) - 1)]
    rmssd = (sum((rr_s[j + 1] - rr_s[j]) ** 2 for j in range(len(rr_s) - 1)) / max(1, len(rr_s) - 1)) ** 0.5 * 1000 if len(rr_s) > 8 else None
    # ---- PPG at 25 Hz: pulse + respiratory baseline wander
    ir, red = [], []
    for i in range(25):
        ts = t_rel + i / 25
        pulse = math.sin(2 * math.pi * hr / 60 * ts) ** 2
        base = 110000 + 1500 * math.sin(2 * math.pi * rr_br / 60 * ts)
        ir.append(int(base + 900 * pulse + random.gauss(0, 60)) if finger else 3000)
        red.append(int(0.8 * base + 500 * pulse) if finger else 2500)
    seq += 1
    rec = {'seq': seq, 't_ms': int(time.time() * 1000), 'uptime_ms': int(t_rel * 1000),
           'vitals': {'hr_ecg': None if leads_off else round(hr, 1), 'hr_ppg': round(hr + random.gauss(0, 1.5), 1) if finger else None,
                      'spo2': round(spo2) if finger else None, 'skin_temp': round(skin, 2), 'amb_temp': 26.5,
                      'rmssd': None if leads_off or rmssd is None else round(rmssd, 1), 'sdnn': None},
           'sqi': {'ecg': 0.0 if leads_off else 0.95, 'ppg': 0.9 if finger else 0.0, 'temp': 1.0,
                   'leads_off': leads_off, 'finger': finger, 'hr_agree': (not leads_off) and finger}}
    backlog.append(rec)
    live = dict(rec, ecg={'fs': 250, 'samples': samples, 'rpeaks': peaks}, ppg={'fs': 25, 'ir': ir, 'red': red})
    if inside(OUT, t_rel):
        print(f'#{seq} OFFLINE (buffered {len(backlog)})')
    else:
        try:
            while len(backlog) > 1:
                chunk = backlog[:min(30, len(backlog) - 1)]
                r = post({'device_id': 'simulator', 'boot_id': boot, 'fw': 'sim', 'rssi': -50, 'backlog': len(backlog), 'records': chunk})
                backlog = [b for b in backlog if b['seq'] > r['ack_seq']]
            r = post({'device_id': 'simulator', 'boot_id': boot, 'fw': 'sim', 'rssi': -50, 'backlog': len(backlog), 'records': [live]})
            backlog = [b for b in backlog if b['seq'] > r['ack_seq']]
            print(f"#{seq} HR {hr:.0f} SpO2 {spo2:.0f} skin {skin:.1f} -> risk {r['risk']} alert {r['alert']}")
        except Exception as e:
            print(f'#{seq} send failed ({e}); buffered {len(backlog)}')
    time.sleep(max(0, (start + seq) - time.time()))

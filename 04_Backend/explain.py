"""
Plain-language explanation of the current readings and of the risk outlook, for the patient view.

Normal / abnormal ranges are the NEWS2 bands (Royal College of Physicians, National Early Warning Score 2, 2017),
the same bands used for the news2_partial feature in 02_ML/features.py:
  score 0 = normal range, 1-2 = outside the normal range, 3 = far outside (NEWS2 asks for an urgent review).
This is a research prototype: the texts describe readings and estimated risk; they never diagnose.
"""
import math

INF = math.inf
# (inclusive upper bound, NEWS2 sub-score)
BANDS = {
    'hr':   [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (INF, 3)],
    'rr':   [(8, 3), (11, 1), (20, 0), (24, 2), (INF, 3)],
    'spo2': [(91, 3), (93, 2), (95, 1), (INF, 0)],
    'temp': [(35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1), (INF, 2)],
}
NORMAL = {'hr': '51–90 beats/min', 'rr': '12–20 breaths/min', 'spo2': '96 % or more', 'temp': '36.1–38.0 °C'}
NAMES = {'hr': 'Heart rate', 'rr': 'Breathing rate', 'spo2': 'Blood oxygen', 'temp': 'Temperature'}
TEXT = {
    ('hr', 'normal'): 'Your heart rate is in the normal resting range.',
    ('hr', 'high'): 'Your heart is beating faster than usual for someone at rest. Fever, infection, dehydration, '
                    'pain, anxiety or recent activity can all do this.',
    ('hr', 'low'): 'Your heart rate is slower than usual. This can be normal for very fit people; '
                   'tell a doctor if you feel dizzy or faint.',
    ('rr', 'normal'): 'Your breathing rate is in the normal range.',
    ('rr', 'high'): 'You are breathing faster than usual. Fever, infection, lung problems or anxiety can do this.',
    ('rr', 'low'): 'You are breathing more slowly than usual.',
    ('spo2', 'normal'): 'The oxygen level in your blood is normal.',
    ('spo2', 'low'): 'The oxygen level in your blood is lower than usual. Sit upright and keep your finger still on the sensor.',
    ('temp', 'normal'): 'Your estimated temperature is normal.',
    ('temp', 'high'): 'Your estimated temperature is in the fever range.',
    ('temp', 'low'): 'Your estimated temperature is lower than usual.',
}


def score(kind, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    for hi, s in BANDS[kind]:
        if v <= hi:
            return s
    return 0


def _level(s):
    return None if s is None else ('normal' if s == 0 else ('urgent' if s >= 3 else 'attention'))


def _direction(kind, v):
    mid = {'hr': 70, 'rr': 16, 'spo2': 100, 'temp': 37.0}[kind]
    return 'low' if v < mid else 'high'


def vital(kind, v, note=None):
    s = score(kind, v)
    lv = _level(s)
    if lv is None:
        return dict(kind=kind, name=NAMES[kind], value=None, level='unknown', score=None, normal=NORMAL[kind],
                    text=note or 'No reliable reading right now.')
    d = 'normal' if lv == 'normal' else _direction(kind, v)
    text = TEXT[(kind, d)]
    if lv == 'urgent':
        text += ' This is far outside the normal range.'
    if kind == 'temp':
        text += ' (Estimated from wrist skin temperature.)'
    if kind == 'rr':
        text += ' (Estimated from the heart signals, not measured directly.)'
    return dict(kind=kind, name=NAMES[kind], value=v, level=lv, score=s, normal=NORMAL[kind], text=text)


def now(rv, sensor=None):
    """rv: recent good-quality medians {hr, spo2, temp_core, rr, rr_quality} or None. sensor: hints from quality."""
    sensor = sensor or {}
    if not rv:
        return dict(level='unknown', headline='Waiting for readings',
                    text='Put the sensors on and keep still for a minute. Readings appear here automatically.',
                    vitals=[vital(k, None) for k in ('hr', 'spo2', 'temp', 'rr')], hints=sensor.get('hints', []))
    rr = rv.get('rr') if (rv.get('rr_quality') or 0) >= 0.3 else None
    vs = [vital('hr', rv.get('hr'), 'Check the chest/arm stickers and the finger sensor.'),
          vital('spo2', rv.get('spo2'), 'Place a finger gently on the clip sensor.'),
          vital('temp', rv.get('temp_core'), 'Check that the temperature sensor touches the skin.'),
          vital('rr', rr, 'Breathing rate needs a clean heart signal for about a minute.')]
    levels = [v['level'] for v in vs]
    if 'urgent' in levels:
        lv, head = 'urgent', 'Some readings need medical attention'
        text = ('One or more readings are far outside the normal range. If you feel unwell, contact a doctor or '
                'nurse now (emergency: 112). This prototype cannot diagnose.')
    elif 'attention' in levels:
        lv, head = 'attention', 'Some readings are outside the normal range'
        text = 'Rest and keep the sensors on. If you start to feel worse, contact a doctor.'
    elif all(l == 'unknown' for l in levels):
        lv, head = 'unknown', 'Waiting for reliable readings'
        text = 'The sensors are connected but the signals are not clean yet. Keep still for a minute.'
    else:
        lv, head = 'normal', 'Your readings are in the normal range'
        text = 'Nothing unusual right now. Keep the sensors on so trends can be followed.'
    return dict(level=lv, headline=head, text=text, vitals=vs, hints=sensor.get('hints', []))


def outlook(horizons, warn_fraction):
    """horizons: {'6': {risk, lo, hi, threshold, prevalence?}, ...} for the latest model hour."""
    out = []
    for h in sorted(horizons or {}, key=int):
        v = horizons[h]
        r, thr = v.get('risk'), v.get('threshold')
        if r is None or thr is None:
            continue
        if r >= thr:
            lv, txt = 'high', (f'Higher than usual chance of developing sepsis within {h} hours. '
                               'The model\'s alarm level is reached: please have a clinician review you.')
        elif r >= warn_fraction * thr:
            lv, txt = 'raised', f'Somewhat raised chance of developing sepsis within {h} hours. Keep monitoring.'
        else:
            lv, txt = 'low', f'Low estimated chance of developing sepsis within {h} hours.'
        prev = v.get('prevalence')
        out.append(dict(hours=int(h), level=lv, text=txt, risk=r, lo=v.get('lo'), hi=v.get('hi'), threshold=thr,
                        times_typical=(r / prev) if prev else None))
    return out

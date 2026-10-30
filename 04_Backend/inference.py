"""Loads the trained wearable models and turns hourly vitals into calibrated risks (6 / 12 / 24 h) + reasons.
Uses the SAME feature builder as training (02_ML/features.py)."""
import json, os, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '02_ML'))
import features as F  # noqa: E402

MODEL_DIR = os.path.join(ROOT, '02_ML', 'models')

PLAIN = {'HR': 'heart rate', 'O2Sat': 'SpO₂', 'Temp': 'temperature', 'Resp': 'breathing rate',
         'Age': 'age', 'Gender': 'sex', 'news2_partial': 'NEWS2 vitals score', 'qsofa_rr': 'breathing rate ≥ 22',
         'sirs_partial': 'SIRS-type criteria count'}
SUFFIX = {'': '', '_mean6': ' (6 h average)', '_std6': ' (6 h variability)', '_min6': ' (6 h minimum)',
          '_max6': ' (6 h maximum)', '_d1': ' (change in 1 h)', '_d6': ' (change in 6 h)',
          '_dev_base': ' (vs. patient\'s own baseline)'}


def plain_name(f):
    for base, txt in PLAIN.items():
        if f == base:
            return txt
        if f.startswith(base + '_'):
            return txt + SUFFIX.get(f[len(base):], ' ' + f[len(base):])
    return f


class _Horizon:
    def __init__(self, model_dir, files, calibration, threshold, prevalence):
        self.models = [lgb.Booster(model_file=os.path.join(model_dir, f)) for f in files]
        cx, cy = np.array(calibration['x']), np.array(calibration['y'])
        self.cal = lambda p: np.interp(p, cx, cy)
        self.threshold = float(threshold)
        self.prevalence = prevalence


class RiskModel:
    def __init__(self, model_dir=MODEL_DIR):
        self.meta = json.load(open(os.path.join(model_dir, 'meta.json')))
        self.cols = self.meta['features']
        hz = self.meta.get('horizons') or {   # older meta.json: 6 h only
            str(self.meta.get('horizon_h', 6)): dict(files=[f'wearable_seed{i}.txt' for i in range(self.meta['n_models'])],
                                                     calibration=self.meta['calibration'], threshold=self.meta['threshold'])}
        self.horizons = {int(h): _Horizon(model_dir, v['files'], v['calibration'], v['threshold'], v.get('test_hour_prevalence'))
                         for h, v in hz.items()}
        self.main_h = int(self.meta.get('horizon_h', 6))
        main = self.horizons[self.main_h]
        self.models, self.threshold = main.models, main.threshold

    def thresholds(self):
        return {h: m.threshold for h, m in self.horizons.items()}

    def predict_hours(self, hourly: pd.DataFrame):
        """hourly: rows = hours (oldest first), columns HR, O2Sat, Temp, Resp, Age, Gender.
        Returns per-hour dicts for every row (risk is causal: row t uses rows <= t).
        risk / risk_lo / risk_hi / reasons are for the main (6 h) horizon; 'horizons' holds every horizon."""
        X = F.patient_features(hourly)[self.cols]
        per_h = {}
        for h, m in self.horizons.items():
            P = np.stack([b.predict(X) for b in m.models])
            mean, sd = P.mean(0), P.std(0)
            per_h[h] = (m.cal(mean), m.cal(np.clip(mean - sd, 0, 1)), m.cal(np.clip(mean + sd, 0, 1)))
        contrib = np.mean([b.predict(X, pred_contrib=True) for b in self.models], axis=0)[:, :-1]
        out = []
        for i in range(len(X)):
            c = contrib[i]
            order = np.argsort(-c)
            up = [dict(feature=self.cols[j], label=plain_name(self.cols[j]), value=_f(X.iloc[i, j]), effect=float(c[j]))
                  for j in order[:3] if c[j] > 0]
            down = [dict(feature=self.cols[j], label=plain_name(self.cols[j]), value=_f(X.iloc[i, j]), effect=float(c[j]))
                    for j in order[::-1][:2] if c[j] < 0]
            hz = {str(h): dict(risk=float(r[i]), lo=float(lo[i]), hi=float(hi[i]), threshold=self.horizons[h].threshold)
                  for h, (r, lo, hi) in sorted(per_h.items())}
            r6, lo6, hi6 = per_h[self.main_h]
            out.append(dict(risk=float(r6[i]), risk_lo=float(lo6[i]), risk_hi=float(hi6[i]),
                            raising=up, lowering=down, horizons=hz))
        return out


def _f(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

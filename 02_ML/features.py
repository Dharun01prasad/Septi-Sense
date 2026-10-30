"""
SeptiSense feature builder - SHARED by training (train.py) and the live backend.
The same function turns hourly vitals into model features in both places, so the model
sees exactly the same kind of input in deployment as in evaluation.

Everything is CAUSAL: the features for hour t use only hours <= t.
"""
import numpy as np
import pandas as pd

# PhysioNet/CinC 2019 column names
VITALS_WEARABLE = ['HR', 'O2Sat', 'Temp', 'Resp']          # what our node measures (Resp derived from ECG/PPG)
VITALS_BP = ['SBP', 'MAP', 'DBP']                           # needs a cuff - ablation only
LABS = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium',
        'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
        'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
DEMOGRAPHICS = ['Age', 'Gender']

# One superset of features is computed for every patient; each model uses a subset.
# 'missingness' features (how often something was measured) describe clinician behaviour in the ICU.
# A wearable measures continuously, so the deployable model must NOT use them (train/deploy mismatch).
WIN = 6  # hours of history used for trend features


def _vital_feats(v, extra=False):
    f = [v, f'{v}_mean6', f'{v}_std6', f'{v}_min6', f'{v}_max6', f'{v}_d1', f'{v}_d6', f'{v}_dev_base']
    return f + ([f'{v}_n6', f'{v}_hrs_since'] if extra else [])


FEATURE_SETS = {
    # deployable: only what the wearable measures + age/sex typed in at admission
    'wearable': sum([_vital_feats(v) for v in VITALS_WEARABLE], []) + DEMOGRAPHICS +
                ['news2_partial', 'qsofa_rr', 'sirs_partial'],
    # + a blood-pressure cuff
    'wearable_bp': sum([_vital_feats(v) for v in VITALS_WEARABLE + VITALS_BP], []) + DEMOGRAPHICS +
                   ['news2_partial', 'news2_bp', 'qsofa_rr', 'qsofa_sbp', 'shock_index', 'sirs_partial'],
    # everything an ICU records (comparison only - our device cannot supply this)
    'full_icu': sum([_vital_feats(v, extra=True) for v in VITALS_WEARABLE + VITALS_BP + ['EtCO2']], []) +
                LABS + [f'{l}_hrs_since' for l in LABS] + DEMOGRAPHICS +
                ['news2_partial', 'news2_bp', 'qsofa_rr', 'qsofa_sbp', 'shock_index', 'sirs_partial'],
}



def news2_partial(hr, rr, spo2, temp, sbp=None):
    """NEWS2 sub-scores for the parameters we have (no consciousness / O2-therapy terms). Vectorised."""
    s = np.zeros(len(hr))
    rr = np.asarray(rr, float); hr = np.asarray(hr, float); spo2 = np.asarray(spo2, float); temp = np.asarray(temp, float)
    s += np.select([rr <= 8, rr <= 11, rr <= 20, rr <= 24, rr > 24], [3, 1, 0, 2, 3], 0) * ~np.isnan(rr)
    s += np.select([spo2 <= 91, spo2 <= 93, spo2 <= 95, spo2 > 95], [3, 2, 1, 0], 0) * ~np.isnan(spo2)
    s += np.select([temp <= 35.0, temp <= 36.0, temp <= 38.0, temp <= 39.0, temp > 39.0], [3, 1, 0, 1, 2], 0) * ~np.isnan(temp)
    s += np.select([hr <= 40, hr <= 50, hr <= 90, hr <= 110, hr <= 130, hr > 130], [3, 1, 0, 1, 2, 3], 0) * ~np.isnan(hr)
    if sbp is not None:
        sbp = np.asarray(sbp, float)
        s += np.select([sbp <= 90, sbp <= 100, sbp <= 110, sbp <= 219, sbp > 219], [3, 2, 1, 0, 3], 0) * ~np.isnan(sbp)
    return s


def patient_features(df):
    """df: one patient's hourly rows (PhysioNet column names; missing columns are treated as unmeasured).
    Returns the feature SUPERSET (DataFrame, same index). Select columns with FEATURE_SETS[name]."""
    n = len(df)
    get = lambda c: (df[c].astype(float) if c in df else pd.Series(np.nan, index=df.index))
    out = {}
    for v in VITALS_WEARABLE + VITALS_BP + ['EtCO2']:
        x = get(v)
        f = x.ffill()                                       # last known value (causal)
        out[v] = f
        r = f.rolling(WIN, min_periods=1)
        out[f'{v}_mean6'] = r.mean()
        out[f'{v}_std6'] = r.std()
        out[f'{v}_min6'] = r.min()
        out[f'{v}_max6'] = r.max()
        out[f'{v}_d1'] = f - f.shift(1)                      # change over the last hour
        out[f'{v}_d6'] = f - f.shift(WIN)                    # change over 6 hours
        base = f.expanding(min_periods=1).median().shift(1)  # personal baseline, from the past only
        out[f'{v}_dev_base'] = f - base                      # deviation from this patient's own normal
        out[f'{v}_n6'] = x.notna().astype(float).rolling(WIN, min_periods=1).sum()
        last = pd.Series(np.where(x.notna(), np.arange(n), np.nan), index=df.index).ffill()
        out[f'{v}_hrs_since'] = np.arange(n) - last
    for lab in LABS:
        x = get(lab)
        out[lab] = x.ffill()
        last = pd.Series(np.where(x.notna(), np.arange(n), np.nan), index=df.index).ffill()
        out[f'{lab}_hrs_since'] = np.arange(n) - last
    for d in DEMOGRAPHICS:
        out[d] = get(d).ffill().bfill()
    hr, rr, spo2, temp, sbp = out['HR'], out['Resp'], out['O2Sat'], out['Temp'], out['SBP']
    out['news2_partial'] = news2_partial(hr, rr, spo2, temp)
    out['news2_bp'] = news2_partial(hr, rr, spo2, temp, sbp)
    out['qsofa_rr'] = (rr >= 22).astype(float)
    out['qsofa_sbp'] = (sbp <= 100).astype(float)
    out['shock_index'] = hr / sbp.replace(0, np.nan)
    out['sirs_partial'] = (hr > 90).astype(float) + ((temp > 38) | (temp < 36)).astype(float) + (rr > 20).astype(float)
    return pd.DataFrame(out, index=df.index).astype('float32')

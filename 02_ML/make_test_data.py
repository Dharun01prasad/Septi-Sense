"""PIPELINE TEST ONLY: writes fake patients in the exact PhysioNet 2019 .psv format.
Never used for any reported result."""
import os, sys
import numpy as np

COLS = ('HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|'
        'Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|'
        'TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel').split('|')


def make(out, n, seed, start):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    for k in range(n):
        T = int(rng.integers(12, 60))
        septic = rng.random() < 0.08
        onset = int(rng.integers(8, T)) if septic else None
        rows = []
        for t in range(T):
            drift = 0.0 if not septic else max(0.0, (t - (onset - 10)) / 10)
            v = {c: np.nan for c in COLS}
            v['HR'] = 80 + 15 * drift + rng.normal(0, 6)
            v['O2Sat'] = 97 - 2 * drift + rng.normal(0, 1)
            v['Temp'] = 37 + 0.8 * drift + rng.normal(0, 0.3) if rng.random() < 0.4 else np.nan
            v['Resp'] = 17 + 5 * drift + rng.normal(0, 2)
            v['SBP'] = 120 - 15 * drift + rng.normal(0, 8)
            v['MAP'] = v['SBP'] * 0.7; v['DBP'] = v['SBP'] * 0.55
            if rng.random() < 0.1:
                v['Lactate'] = 1.2 + drift + rng.normal(0, 0.3)
                v['WBC'] = 9 + 4 * drift + rng.normal(0, 2)
            v['Age'] = 60.0; v['Gender'] = float(k % 2); v['ICULOS'] = t + 1; v['HospAdmTime'] = -1.0
            v['SepsisLabel'] = int(septic and t >= onset - 6)
            rows.append('|'.join('NaN' if (isinstance(x, float) and np.isnan(x)) else (f'{x:.2f}' if isinstance(x, float) else str(x))
                                 for x in (v[c] for c in COLS)))
        with open(os.path.join(out, f'p{start + k:06d}.psv'), 'w') as f:
            f.write('|'.join(COLS) + '\n' + '\n'.join(rows) + '\n')


if __name__ == '__main__':
    root = sys.argv[1]
    make(os.path.join(root, 'training_setA'), 1500, 1, 1)
    make(os.path.join(root, 'training_setB'), 1000, 2, 100001)
    print('fake data written to', root)

"""
SeptiSense - train and evaluate the sepsis early-warning model.

Dataset : PhysioNet/CinC Challenge 2019 (open access, CC BY 4.0). Hourly ICU records from two hospital systems.
Target  : SepsisLabel at hour t. In this dataset the label is 1 from 6 h BEFORE clinical sepsis onset (Sepsis-3),
          so the model answers "will this patient meet Sepsis-3 within the next 6 hours (or already is septic)?"
          12 h and 24 h outlook models use the same onset (first label hour + 6) with a longer window.
Split   : train + validate on hospital A (patient-level split), test ONCE on hospital B (never seen).
Leakage : features are causal (hour t uses hours <= t only); patients never cross splits; ICULOS and
          HospAdmTime are excluded; the deployable model has no 'how often was it measured' features.

Usage   : python3 train.py <data_root>   (data_root contains training_setA/ and training_setB/ with .psv files)
Output  : models/, results/metrics.json, results/*.png, replay/*.json
"""
import json, os, sys, time, glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_M, OUT_R, OUT_P = (os.path.join(HERE, d) for d in ('models', 'results', 'replay'))
SEEDS = [11, 22, 33, 44, 55]
ALARM_PERSIST = 2          # alarm only after risk stays above threshold for 2 consecutive hours
HORIZON_H = 6              # the dataset's label horizon
HORIZONS = (6, 12, 24)     # deployed outlook: risk of Sepsis-3 onset within 6 / 12 / 24 h
HORIZON_SEEDS = SEEDS[:3]  # ensemble size for the 12 h and 24 h models


# ------------------------------------------------------------------ loading
def _read(path):
    df = pd.read_csv(path, sep='|')
    df['pid'] = os.path.basename(path)[:-4]
    df['hour'] = np.arange(len(df))
    return df


def load_set(root, name):
    files = sorted(glob.glob(os.path.join(root, '**', name, '*.psv'), recursive=True))
    if not files:
        raise SystemExit(f'no .psv files found for {name} under {root}')
    with Pool(max(1, os.cpu_count())) as p:
        dfs = p.map(_read, files, chunksize=200)
    return pd.concat(dfs, ignore_index=True)


def _feat_one(df):
    X = F.patient_features(df)
    X['pid'] = df['pid'].values
    X['hour'] = df['hour'].values
    y = df['SepsisLabel'].values.astype(np.int8)
    X['y'] = y                                               # = 6 h horizon (the dataset's own label)
    # longer horizons from the same onset: onset = first label hour + 6, so y_H(t) = 1 from onset - H onwards
    t_l = int(np.argmax(y)) if y.max() else None
    for H in HORIZONS[1:]:
        X[f'y{H}'] = ((np.arange(len(df)) >= t_l + HORIZON_H - H) if t_l is not None else np.zeros(len(df), bool)).astype(np.int8)
    return X


def build_features(raw):
    groups = [g for _, g in raw.groupby('pid', sort=False)]
    with Pool(max(1, os.cpu_count())) as p:
        parts = p.map(_feat_one, groups, chunksize=100)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ model
PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_data_in_leaf=300,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              metric='average_precision', verbose=-1, num_threads=max(1, os.cpu_count()))


def fit(Xtr, ytr, Xva, yva, cols, seed):
    pos = ytr.mean()
    p = dict(PARAMS, seed=seed, scale_pos_weight=float(np.sqrt((1 - pos) / pos)))   # moderate re-weighting
    dtr = lgb.Dataset(Xtr[cols], ytr, free_raw_data=False)
    dva = lgb.Dataset(Xva[cols], yva, reference=dtr)
    return lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def predict(models, X, cols):
    P = np.stack([m.predict(X[cols], num_iteration=m.best_iteration) for m in models])
    return P.mean(0), P.std(0)


# ------------------------------------------------------------------ evaluation helpers
def boot_ci(y, s, pid, fn, n=200, seed=0):
    """Patient-level bootstrap 95 % CI."""
    rng = np.random.default_rng(seed)
    codes, ids = pd.factorize(pid)
    idx_by = pd.Series(np.arange(len(pid))).groupby(codes).apply(lambda r: r.values).tolist()
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(ids), len(ids))
        ix = np.concatenate([idx_by[i] for i in pick])
        if y[ix].min() == y[ix].max():
            continue
        vals.append(fn(y[ix], s[ix]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def alarm_runs(scores, thr):
    """True at hour t when the score has been >= thr for ALARM_PERSIST consecutive hours ending at t."""
    above = (scores >= thr).astype(int)
    return np.convolve(above, np.ones(ALARM_PERSIST, int), 'full')[:len(above)] >= ALARM_PERSIST


def alarm_eval(df, score_col, thr, horizon=HORIZON_H, n_chance=20, seed=0):
    """Patient-level alarm behaviour with the persistence rule.
    Septic patients (those already labelled at their first hour are excluded and counted separately):
      'warned in time' = an alarm hour inside the horizon window [onset - horizon, onset), onset = first label hour + 6.
      Lead time is measured inside that window, so it can never exceed the horizon. An alarm days earlier is not
      evidence of predicting this episode; those are reported separately (any_alarm_before_onset_rate).
    Chance baseline: the same window rule applied to the alarm sequences of randomly drawn NON-septic patients over
      the same hours since admission (keeps their real alarm rate and timing). The model is only useful if
      detection_rate is clearly above chance_detection_rate.
    Non-septic patients: alarm episodes per patient-day."""
    sep, non = [], []
    for _, g in df.groupby('pid', sort=False):
        run = alarm_runs(g[score_col].values, thr)
        y = g['y'].values
        (sep.append((int(np.argmax(y)), run)) if y.max() == 1 else non.append(run))
    on_arrival = sum(1 for t, _ in sep if t == 0)
    elig = [(t + HORIZON_H, run) for t, run in sep if t > 0]
    hit, any_before, lead = 0, 0, []
    for onset, run in elig:
        start, end = max(0, onset - horizon), min(onset, len(run))
        w = np.where(run[start:end])[0]
        any_before += bool(run[:end].any())
        if len(w):
            hit += 1
            lead.append(onset - (start + w[0]))
    lens = np.array([len(r) for r in non])
    rng = np.random.default_rng(seed)
    chance = []
    for onset, run in elig:
        start, end = max(0, onset - horizon), min(onset, len(run))
        cand = np.flatnonzero(lens >= end)
        if end > start and len(cand):
            pick = rng.choice(cand, min(n_chance, len(cand)), replace=False)
            chance.append(np.mean([non[i][start:end].any() for i in pick]))
    fa = sum(int(np.sum(r & ~np.r_[False, r[:-1]])) for r in non)
    lead = np.array(lead, float) if lead else np.array([np.nan])
    n = max(1, len(elig))
    return dict(horizon_h=horizon, septic_patients=len(elig), septic_on_arrival=on_arrival,
                detected_in_window=hit, detection_rate=hit / n,
                chance_detection_rate=float(np.mean(chance)) if chance else None,
                any_alarm_before_onset_rate=any_before / n,
                lead_time_median_h=float(np.nanmedian(lead)),
                lead_time_iqr_h=[float(np.nanpercentile(lead, 25)), float(np.nanpercentile(lead, 75))],
                false_alarms_per_patient_day=fa / max(1e-9, lens.sum() / 24))


def threshold_at_specificity(y, s, spec=0.90):
    neg = np.sort(s[y == 0])
    return float(neg[int(spec * (len(neg) - 1))])


# ------------------------------------------------------------------ main
def main(root):
    t0 = time.time()
    for d in (OUT_M, OUT_R, OUT_P):
        os.makedirs(d, exist_ok=True)
    print('loading hospital A ...'); A = load_set(root, 'training_setA')
    print('loading hospital B ...'); B = load_set(root, 'training_setB')
    print(f'  A: {A.pid.nunique()} patients, {len(A)} hours | B: {B.pid.nunique()} patients, {len(B)} hours  ({time.time()-t0:.0f}s)')
    print('building features ...')
    FA, FB = build_features(A), build_features(B)
    raw_B = B.set_index(['pid', 'hour'])
    print(f'  done ({time.time()-t0:.0f}s)')

    # patient-level split of hospital A, stratified by "ever septic"
    rng = np.random.default_rng(7)
    sep = FA.groupby('pid').y.max()
    val_ids = set()
    for lab in (0, 1):
        ids = sep[sep == lab].index.to_numpy(dtype=object).copy()   # plain numpy array: safe to shuffle (pandas 3)
        rng.shuffle(ids)
        val_ids |= set(ids[: int(0.2 * len(ids))])
    is_va = FA.pid.isin(val_ids).values
    TR, VA = FA[~is_va], FA[is_va]
    print(f'  train {TR.pid.nunique()} / val {VA.pid.nunique()} patients (hospital A) | test {FB.pid.nunique()} (hospital B)')

    results = {'dataset': 'PhysioNet/CinC Challenge 2019 (training_setA = hospital A, training_setB = hospital B)',
               'target': 'SepsisLabel(t): Sepsis-3 onset within the next 6 h or already septic',
               'horizon_targets': {str(H): f'Sepsis-3 onset within the next {H} h or already septic '
                                           f'(onset = first SepsisLabel hour + 6)' for H in HORIZONS},
               'alerting_definition': (f'alarm = risk >= threshold for {ALARM_PERSIST} consecutive hours; '
                                       'warned in time = an alarm hour within the horizon window before onset; '
                                       'chance = same rule on alarm sequences of random non-septic patients'),
               'split': {'train_patients': int(TR.pid.nunique()), 'val_patients': int(VA.pid.nunique()),
                         'test_patients': int(FB.pid.nunique()),
                         'test_hour_prevalence': float(FB.y.mean()),
                         'test_patient_prevalence': float(FB.groupby('pid').y.max().mean())},
               'models': {}, 'horizons': {}}
    curves = {}
    deployed = {}

    def evaluate(name, cols, ycol, horizon, seeds):
        """Train an ensemble on hospital A for one target, calibrate + pick the threshold on A-validation,
        then evaluate once on hospital B."""
        models = [fit(TR, TR[ycol].values, VA, VA[ycol].values, cols, s) for s in seeds]
        pva, _ = predict(models, VA, cols)
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(pva, VA[ycol].values)
        thr = float(iso.predict([threshold_at_specificity(VA[ycol].values, pva, 0.90)])[0])
        pte, sd = predict(models, FB, cols)
        pte_cal = iso.predict(pte)
        y = FB[ycol].values
        auroc, auprc = roc_auc_score(y, pte), average_precision_score(y, pte)
        al = alarm_eval(FB.assign(s=pte_cal), 's', thr, horizon)
        pred = pte_cal >= thr
        tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
        fn_, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
        r = dict(horizon_h=horizon, n_features=len(cols), n_models=len(models),
                 test_hour_prevalence=float(y.mean()),
                 auroc=float(auroc), auroc_ci95=boot_ci(y, pte, FB.pid.values, roc_auc_score),
                 auprc=float(auprc), auprc_ci95=boot_ci(y, pte, FB.pid.values, average_precision_score),
                 brier_calibrated=float(brier_score_loss(y, pte_cal)),
                 brier_prevalence_only=float(y.mean() * (1 - y.mean())),
                 threshold_calibrated=thr, threshold_rule='90 % hour-level specificity on hospital-A validation',
                 sensitivity=tp / max(1, tp + fn_), specificity=tn / max(1, tn + fp), ppv=tp / max(1, tp + fp),
                 npv=tn / max(1, tn + fn_), accuracy=(tp + tn) / max(1, len(y)),
                 balanced_accuracy=(tp / max(1, tp + fn_) + tn / max(1, tn + fp)) / 2, alarms=al)
        print(f'  {name:12s} {horizon:2d} h  AUROC {auroc:.3f}  AUPRC {auprc:.3f}  warned-in-window {al["detection_rate"]:.2f} '
              f'(chance {al["chance_detection_rate"] or 0:.2f})  FA/day {al["false_alarms_per_patient_day"]:.2f}  ({time.time()-t0:.0f}s)')
        return r, dict(models=models, iso=iso, thr=thr, cols=cols, sd=sd, pte=pte_cal, raw=pte, y=y)

    # ---- 6 h: deployed wearable model + ablation (what a BP cuff / full ICU data would add)
    for name in ('wearable', 'wearable_bp', 'full_icu'):
        cols = F.FEATURE_SETS[name]
        r, d = evaluate(name, cols, 'y', HORIZON_H, SEEDS if name == 'wearable' else SEEDS[:1])
        results['models'][name] = r
        curves[name] = (roc_curve(d['y'], d['raw']), precision_recall_curve(d['y'], d['raw']))
        if name == 'wearable':
            deployed[HORIZON_H] = d
            results['horizons'][str(HORIZON_H)] = r
            y, pte_cal = d['y'], d['pte']
            bins = np.linspace(0, 1, 11)
            q = np.clip(np.digitize(pte_cal, bins) - 1, 0, 9)
            r['calibration_bins'] = [(float(pte_cal[q == b].mean()), float(y[q == b].mean()), int((q == b).sum()))
                                     for b in range(10) if (q == b).sum() > 50]
            # operating curve: warned-in-window vs false alarms per day across thresholds (+ chance)
            op = []
            for t in np.unique(np.quantile(pte_cal, [0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99])):
                a = alarm_eval(FB.assign(s=pte_cal), 's', t, HORIZON_H, n_chance=5)
                op.append((float(t), a['detection_rate'], a['false_alarms_per_patient_day'], a['lead_time_median_h'],
                           a['chance_detection_rate']))
            r['operating_curve'] = op
            imp = pd.DataFrame({m_i: pd.Series(m.feature_importance('gain'), index=cols)
                                for m_i, m in enumerate(d['models'])}).mean(axis=1)
            r['top_features'] = [(k, float(v)) for k, v in imp.sort_values(ascending=False).head(15).items()]

    # ---- 12 h and 24 h outlook: same wearable features, longer targets
    for H in HORIZONS[1:]:
        r, d = evaluate('wearable', F.FEATURE_SETS['wearable'], f'y{H}', H, HORIZON_SEEDS)
        results['horizons'][str(H)] = r
        deployed[H] = d

    # clinical-score baseline on the same test set (6 h target)
    y = FB.y.values
    for score in ('news2_partial', 'news2_bp'):
        s = np.nan_to_num(FB[score].values, nan=0.0)
        results.setdefault('baselines', {})[score] = dict(auroc=float(roc_auc_score(y, s)),
                                                         auprc=float(average_precision_score(y, s)))
        curves[score] = (roc_curve(y, s), precision_recall_curve(y, s))
    print('  baselines', {k: round(v['auroc'], 3) for k, v in results['baselines'].items()})

    # ------------------------------------------------ save the deployable models
    # 6 h keeps the original file names (wearable_seed{i}.txt); 12/24 h are wearable_h{H}_seed{i}.txt
    hz_meta = {}
    for H, d in deployed.items():
        files = []
        for i, m in enumerate(d['models']):
            f = f'wearable_seed{i}.txt' if H == HORIZON_H else f'wearable_h{H}_seed{i}.txt'
            m.save_model(os.path.join(OUT_M, f), num_iteration=m.best_iteration)
            files.append(f)
        hz_meta[str(H)] = dict(files=files, n_models=len(files), threshold=d['thr'],
                               calibration={'x': d['iso'].X_thresholds_.tolist(), 'y': d['iso'].y_thresholds_.tolist()},
                               test_hour_prevalence=results['horizons'][str(H)]['test_hour_prevalence'])
    d6 = deployed[HORIZON_H]
    meta = dict(feature_set='wearable', features=d6['cols'], n_models=len(d6['models']),
                calibration=hz_meta[str(HORIZON_H)]['calibration'], threshold=d6['thr'],
                horizon_h=HORIZON_H, horizons=hz_meta, alarm_persist_h=ALARM_PERSIST,
                metrics=results['models']['wearable'], trained_on='hospital A (PhysioNet 2019 set A)',
                tested_on='hospital B (PhysioNet 2019 set B)')
    json.dump(meta, open(os.path.join(OUT_M, 'meta.json'), 'w'), indent=1)
    json.dump(results, open(os.path.join(OUT_R, 'metrics.json'), 'w'), indent=1)

    # ------------------------------------------------ replay patients (held-out hospital B)
    # 3 septic patients warned inside the 6 h window (random among those with the longest lead) + 1 non-septic
    FB['risk'] = d6['pte']
    picks = []
    for pid, g in FB.groupby('pid', sort=False):
        y_ = g.y.values
        if y_.max() == 1 and 10 <= np.argmax(y_) and len(g) >= 24:
            onset = int(np.argmax(y_)) + HORIZON_H
            run = alarm_runs(g.risk.values, d6['thr'])
            start = max(0, onset - HORIZON_H)
            hit = np.where(run[start:min(onset, len(run))])[0]
            if len(hit):
                picks.append((onset - (start + hit[0]), pid))
    picks = [p for _, p in sorted(picks, reverse=True)[:40]]
    rng2 = np.random.default_rng(3)
    chosen = list(rng2.choice(picks, size=min(3, len(picks)), replace=False)) if picks else []
    nons = [pid for pid, g in FB.groupby('pid', sort=False) if g.y.max() == 0 and len(g) >= 36]
    chosen += list(rng2.choice(nons, size=1, replace=False))
    for pid in chosen:
        g = FB[FB.pid == pid]
        r = raw_B.loc[pid]
        hours = []
        for h in g.hour.values:
            row = r.loc[h]
            hours.append({k: (None if pd.isna(row[k]) else float(row[k])) for k in ('HR', 'O2Sat', 'Temp', 'Resp', 'SBP')} |
                         {'hour': int(h), 'label': int(row['SepsisLabel'])})
        y_ = g.y.values
        json.dump(dict(pid=str(pid), source='PhysioNet 2019 training_setB (held-out hospital)',
                       age=float(g.Age.iloc[0]), gender=int(g.Gender.iloc[0]),
                       label_start_hour=(int(np.argmax(y_)) if y_.max() else None),
                       onset_hour=(int(np.argmax(y_)) + HORIZON_H if y_.max() else None),
                       hours=hours, offline_risk=[float(v) for v in g.risk.values]),
                  open(os.path.join(OUT_P, f'{pid}.json'), 'w'))
    print('  replay patients:', [str(p) for p in chosen])
    plots(results, curves)
    print(f'done in {time.time()-t0:.0f}s')


def plots(results, curves):
    names = {'wearable': 'Wearable only (deployed)', 'wearable_bp': 'Wearable + BP cuff', 'full_icu': 'Full ICU (vitals + labs)',
             'news2_partial': 'NEWS2 (HR, RR, SpO2, T)', 'news2_bp': 'NEWS2 + SBP'}
    colors = {'wearable': '#c2410c', 'wearable_bp': '#2563eb', 'full_icu': '#16a34a', 'news2_partial': '#6b7280', 'news2_bp': '#9ca3af'}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    for k, ((fpr, tpr, _), (pr, rc, _)) in curves.items():
        a = results['models'].get(k) or results['baselines'].get(k)
        ls = '--' if k.startswith('news2') else '-'
        ax[0].plot(fpr, tpr, ls, color=colors[k], lw=2, label=f"{names[k]}  AUROC {a['auroc']:.3f}")
        ax[1].plot(rc, pr, ls, color=colors[k], lw=2, label=f"{names[k]}  AUPRC {a['auprc']:.3f}")
    prev = results['split']['test_hour_prevalence']
    ax[0].plot([0, 1], [0, 1], ':', color='#999'); ax[1].axhline(prev, ls=':', color='#999')
    ax[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC - held-out hospital B')
    ax[1].set(xlabel='Recall', ylabel='Precision', title=f'Precision-recall - hospital B (prevalence {prev:.3f})')
    for a_ in ax:
        a_.legend(fontsize=8.5, loc='lower right' if a_ is ax[0] else 'upper right'); a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_R, 'roc_pr.png'), dpi=160); plt.close(fig)

    w = results['models']['wearable']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    c = np.array(w['calibration_bins'])
    ax[0].plot([0, 1], [0, 1], ':', color='#999'); ax[0].plot(c[:, 0], c[:, 1], 'o-', color='#c2410c', lw=2)
    ax[0].set(xlabel='Predicted risk', ylabel='Observed frequency', title='Calibration (wearable model, hospital B)')
    op = np.array([[np.nan if v is None else v for v in o] for o in w['operating_curve']], float)
    ax[1].plot(op[:, 2], op[:, 1] * 100, 'o-', color='#c2410c', lw=2, label='model')
    ax[1].plot(op[:, 2], op[:, 4] * 100, 's--', color='#6b7280', lw=1.5, label='chance (random non-septic alarm sequences)')
    a = w['alarms']
    ax[1].plot([a['false_alarms_per_patient_day']], [a['detection_rate'] * 100], '*', ms=16, color='#111', label='chosen threshold')
    ax[1].set(xlabel='False alarms per patient-day', ylabel=f'Septic patients warned within {HORIZON_H} h before onset (%)',
              title=f'Alert trade-off ({ALARM_PERSIST}-hour persistence rule)'); ax[1].legend(fontsize=8.5)
    tf = w['top_features'][:12][::-1]
    ax[2].barh([t[0] for t in tf], [t[1] for t in tf], color='#c2410c')
    ax[2].set(title='Most used features (gain)', xlabel='total gain')
    for a_ in ax[:2]:
        a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_R, 'calibration_alerts_features.png'), dpi=160); plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1])

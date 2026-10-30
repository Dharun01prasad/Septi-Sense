"""
Breathing-rate estimation from signals the node already streams (no extra sensor).

Three surrogate signals carry the breathing rhythm:
  EDR  - ECG-derived respiration: R-peak amplitude changes as the chest moves
  RSA  - respiratory sinus arrhythmia: heart rate speeds up on inhalation
  RIIV - respiratory-induced intensity variation of the finger PPG baseline
Each is band-passed to 0.1-0.6 Hz (6-36 breaths/min); its dominant frequency is an estimate,
with a quality = share of power at that peak. Estimates that agree are fused by a quality-weighted
median. This follows the 'respiratory surrogate fusion' idea from the PPG/ECG literature;
accuracy on our hardware is NOT validated (see the report).
"""
import numpy as np
from scipy.signal import butter, filtfilt, welch

BAND = (0.10, 0.60)


def _rate(x, fs):
    x = np.asarray(x, float)
    if len(x) < fs * 30 or not np.isfinite(x).all() or np.std(x) == 0:
        return None, 0.0
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
    b, a = butter(2, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype='band')
    y = filtfilt(b, a, x)
    f, p = welch(y, fs=fs, nperseg=min(len(y), int(fs * 32)))
    m = (f >= BAND[0]) & (f <= BAND[1])
    if not m.any() or p[m].sum() == 0:
        return None, 0.0
    k = np.argmax(p[m])
    fk = f[m][k]
    near = np.abs(f[m] - fk) <= 0.03
    quality = float(p[m][near].sum() / p[m].sum())
    return float(fk * 60.0), quality


def _beats_to_series(t, v, fs=4.0):
    t = np.asarray(t, float); v = np.asarray(v, float)
    if len(t) < 8:
        return None
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    return np.interp(grid, t, v)


def estimate(beat_t, beat_amp, ppg_t, ppg_ir, window_s=60.0):
    """beat_t: R-peak times (s), beat_amp: R amplitudes; ppg_t/ppg_ir: PPG samples (s, raw IR).
    Returns dict(rr, quality, parts) or rr=None."""
    parts = {}
    if len(beat_t) >= 10:
        t = np.asarray(beat_t, float)
        keep = t >= t[-1] - window_s
        t, a = t[keep], np.asarray(beat_amp, float)[keep]
        if len(t) >= 10:
            rr_iv = np.diff(t)
            ok = (rr_iv > 0.3) & (rr_iv < 2.0)
            edr = _beats_to_series(t, a)
            if edr is not None:
                parts['edr'] = _rate(edr, 4.0)
            if ok.sum() >= 8:
                rsa = _beats_to_series(t[1:][ok], rr_iv[ok])
                if rsa is not None:
                    parts['rsa'] = _rate(rsa, 4.0)
    if len(ppg_t) >= 25 * 30:
        t = np.asarray(ppg_t, float)
        keep = t >= t[-1] - window_s
        ir = np.asarray(ppg_ir, float)[keep]
        if len(ir) >= 25 * 30 and np.median(ir) > 50000:
            dec = ir[: len(ir) // 5 * 5].reshape(-1, 5).mean(1)   # 25 Hz -> 5 Hz
            parts['riiv'] = _rate(dec, 5.0)
    good = [(r, q) for r, q in parts.values() if r is not None and q >= 0.25]
    if not good:
        return dict(rr=None, quality=0.0, parts={k: v[0] for k, v in parts.items()})
    rates = np.array([g[0] for g in good]); qs = np.array([g[1] for g in good])
    order = np.argsort(rates); cw = np.cumsum(qs[order])
    fused = float(rates[order][np.searchsorted(cw, cw[-1] / 2)])
    agree = float(np.mean(np.abs(rates - fused) <= 3.0))      # do the surrogates agree within 3 br/min?
    return dict(rr=fused, quality=float(np.clip(qs.mean() * agree, 0, 1)), parts={k: v[0] for k, v in parts.items()})

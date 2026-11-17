"""How much does the label interpolation change window labels? (2026-11-15, analysis only)

Behind docs/guide/13_from_ratings_to_numbers.md O25 (8 s windows, all 483 student-sessions of
experiments 01-09). Run from the repo root: python -m scripts.analysis.label_interp_compare
"""
import glob
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from src.labels.attention import load_subject_labels, SELF_LABELER

DT, LEN, STEP, TAU = 0.1, 8.0, 4.0, 3.0

def curve(t, v, grid, kind):
    if "+stale" in kind:                        # <kind>+stale<s>: silent for more than s seconds -> 3
        base, st = kind.split("+stale")
        y = curve(t, v, grid, base)
        last = t[np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)]
        return np.where((grid - last > float(st)) & (grid >= t[0]), 3.0, y)
    return _curve(t, v, grid, kind)


def _curve(t, v, grid, kind):
    keep = np.r_[True, np.diff(t) > 1e-6]; t, v = t[keep], v[keep]
    if kind == "step" or len(t) < 2:
        i = np.searchsorted(t, grid, side="right") - 1
        return np.where(i >= 0, v[np.clip(i, 0, None)], np.nan)
    if kind == "backward":                      # value applies to the time BEFORE its event
        i = np.searchsorted(t, grid, side="left")
        out = v[np.clip(i, 0, len(v) - 1)].astype(float)
        out[grid < t[0]] = np.nan
        return out
    if kind == "ramp3":                         # step, with a 3 s linear ramp ending at each event
        x = np.sort(np.r_[t, t[1:] - 3.0]); x = x[x >= t[0]]
        y = curve(t, v, x + 1e-9, "step")
        y[np.isin(x, t[1:] - 3.0)] = curve(t, v, x[np.isin(x, t[1:] - 3.0)], "step")
        out = np.interp(grid, x, y); out[grid < t[0]] = np.nan; return out
    if kind.startswith("additive") or kind.startswith("shrunk"):
        sig = float(kind.lstrip("additveshrunk"))
        K = np.exp(-0.5 * ((grid[:, None] - t[None]) / sig) ** 2)
        num = (K * (v - 3)).sum(1)
        out = 3 + (num if kind.startswith("additive") else num / (1 + K.sum(1)))   # shrunk: kernel mean, prior weight 1 at 3
        out[grid < t[0]] = np.nan; return out
    if kind.startswith("decay"):                # step relaxing toward 3, half-life in seconds since the last event
        half = float(kind[5:]); y = curve(t, v, grid, "step")
        last = t[np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)]
        return 3 + (y - 3) * np.exp2(-np.maximum(grid - last, 0) / half)
    if kind.startswith("gauss"):                # gauss<sigma>[_back<lag>]: step (*) Gaussian, optionally shifted earlier
        from scipy.special import ndtr
        sig, _, lag = kind[5:].partition("_back")
        tt = t - float(lag or 0)
        out = v[0] + (ndtr((grid[:, None] - tt[1:]) / float(sig)) * np.diff(v)).sum(-1)
        out[grid < t[0]] = np.nan; return out
    if kind == "linear":
        out = np.interp(grid, t, v)
    elif kind == "pchip":
        out = PchipInterpolator(t, v, extrapolate=False)(grid); out[grid > t[-1]] = v[-1]
    elif kind == "cubic":
        out = CubicSpline(t, v, extrapolate=False)(grid); out[grid > t[-1]] = v[-1]
    out[grid < t[0]] = np.nan
    return out

kinds = ["step", "step+stale4", "gauss2", "gauss2+stale4", "step+stale8"]
cons = {k: [] for k in kinds}; over = []
for z in sorted(glob.glob("data/DIPSER/DIPSEER/*/*/subjects/*.zip")):
    L = {k: v for k, v in load_subject_labels(z).items() if len(v.t)}
    if not L: continue
    t0, t1 = min(v.t[0] for v in L.values()), max(v.t[-1] for v in L.values())
    grid = np.arange(t0, t1, DT)
    starts = np.arange(t0, t1 - LEN, STEP)
    idx = [(np.searchsorted(grid, s), np.searchsorted(grid, s + LEN)) for s in starts]
    for k in kinds:
        C = np.array([curve(v.t, v.value, grid, k) for v in L.values()])
        if k == "cubic": over.append(np.mean((C[np.isfinite(C)] < 1) | (C[np.isfinite(C)] > 5)))
        W = np.array([[np.nanmean(c[a:b]) if np.isfinite(c[a:b]).any() else np.nan for (a, b) in idx] for c in C])
        cons[k].append(np.nanmedian(W, 0))
base = np.concatenate(cons["step"])
print(f"student-windows: {len(base)}")
for k in kinds[1:]:
    c = np.concatenate(cons[k]); ok = np.isfinite(base) & np.isfinite(c)
    flip = np.mean((base[ok] >= TAU) != (c[ok] >= TAU))
    cr = np.round(c[ok], 2); flip_r = np.mean((base[ok] >= TAU) != (cr >= TAU))
    print(f"   exactly-3 under step: {np.mean(base[ok] == 3):.1%}; flips after rounding to 0.01: {flip_r:.1%}; engaged rate rounded {np.mean(cr >= TAU):.3f}")
    print(f"   {k}: student-windows at neutral (|consensus - 3| < 0.01): {np.mean(np.abs(c[ok] - 3) < 0.01):.1%} (step {np.mean(np.abs(base[ok] - 3) < 0.01):.1%}); outside 1-5: {np.mean((c[ok] < 1) | (c[ok] > 5)):.2%}")
    for thr in (3.5, 4.0):
        print(f"   at tau_att {thr}: flips {np.mean((base[ok] >= thr) != (c[ok] >= thr)):.1%}, engaged {np.mean(c[ok] >= thr):.3f} (step {np.mean(base[ok] >= thr):.3f})")
    print(f"{k:9s} vs step: mean |consensus diff| {np.mean(np.abs(c[ok]-base[ok])):.3f}  r {np.corrcoef(c[ok], base[ok])[0,1]:.3f}  "
          f"student label (>= {TAU}) flips {flip:.1%}  engaged rate {np.mean(c[ok]>=TAU):.3f} (step {np.mean(base[ok]>=TAU):.3f})")
print(f"cubic spline: share of time outside 1-5: {np.mean(over):.1%}")

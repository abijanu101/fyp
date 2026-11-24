"""How a labeler's change events become a value at any time (config `labels.interp`).

Labelers enter a rating only when it changes (offline, frame by frame at 1 fps), so a value between
entries exists only through an interpolation. `step` (hold until the next entry) is the literal
record; the others are models of reaction delay or uncertainty laid over it
(docs/guide/13_from_ratings_to_numbers.md, O25; docs/misc/targets.md §2.1). Every mode is NaN before
the labeler's first entry and holds the last value after the last one; values are clipped to 1-5.

    step    the value holds from its entry until the next one
    ramp    step, but each change is a linear ramp over `ramp_s` seconds ending at the entry
    linear  a straight line between consecutive entries
    pchip   a monotone cubic between entries (smooth, never past a rated value)
    cubic   a natural cubic spline through the entries (can overshoot; clipped)
    gauss   the step blurred by a Gaussian of width `sigma_s`: each change an S-curve centred on it
"""

from __future__ import annotations

import numpy as np

INTERP_MODES = ("step", "ramp", "linear", "pchip", "cubic", "gauss")
RATING_MIN, RATING_MAX = 1.0, 5.0


def interpolate(times: np.ndarray, values: np.ndarray, t: np.ndarray | float, mode: str = "step",
                ramp_s: float = 3.0, sigma_s: float = 2.0) -> np.ndarray:
    """Value of one labeler's rating curve at time(s) `t`."""
    if mode not in INTERP_MODES:
        raise ValueError(f"labels.interp.mode must be one of {INTERP_MODES}, got {mode!r}")
    times, values = np.asarray(times, float), np.asarray(values, float)
    t = np.asarray(t, float)
    if not len(values):
        return np.full(t.shape, np.nan)
    if len(times) > 1:                                  # several entries at one instant: the last wins
        keep = np.r_[np.diff(times) > 1e-9, True]
        times, values = times[keep], values[keep]
    before = t < times[0]
    if mode == "step" or len(times) == 1:
        i = np.searchsorted(times, t, side="right") - 1
        out = values[np.clip(i, 0, len(values) - 1)].astype(float)
    elif mode == "ramp":                                # knots: (entry - ramp, old value), (entry, new value)
        starts = np.maximum(times[1:] - ramp_s, times[:-1])
        x = np.concatenate([times[:1], np.ravel(np.column_stack([starts, times[1:]]))])
        y = np.concatenate([values[:1], np.ravel(np.column_stack([values[:-1], values[1:]]))])
        out = np.interp(t, x, y)
    elif mode == "linear":
        out = np.interp(t, times, values)
    elif mode == "gauss":                               # step (*) Gaussian: v0 + sum dv_i * Phi((t - t_i) / sigma)
        from scipy.special import ndtr
        out = values[0] + (ndtr((t[..., None] - times[1:]) / max(sigma_s, 1e-6)) * np.diff(values)).sum(-1)
    else:
        from scipy.interpolate import CubicSpline, PchipInterpolator
        f = PchipInterpolator(times, values) if mode == "pchip" else CubicSpline(times, values, bc_type="natural")
        out = np.where(t > times[-1], values[-1], f(np.clip(t, times[0], times[-1])))
    return np.where(before, np.nan, np.clip(out, RATING_MIN, RATING_MAX))


def window_mean(times: np.ndarray, values: np.ndarray, t0: float, t1: float, mode: str = "step",
                ramp_s: float = 3.0, sigma_s: float = 2.0, dt: float = 0.2) -> float:
    """Mean of the interpolated curve over [t0, t1), sampled every `dt` seconds (the camera's 5 fps).
    NaN where the window has no defined value (it lies before the labeler's first entry)."""
    grid = np.arange(t0, t1, dt) + dt / 2
    v = interpolate(times, values, grid, mode, ramp_s, sigma_s)
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def normalise(rating: np.ndarray | float) -> np.ndarray:
    """Rating 1-5 -> soft label 0-1, clipped (a rating outside 1-5 is clipped even where the parser
    should already have dropped it)."""
    r = np.clip(np.asarray(rating, float), RATING_MIN, RATING_MAX)
    return np.clip((r - RATING_MIN) / (RATING_MAX - RATING_MIN), 0.0, 1.0)


def to_rating(p: np.ndarray | float) -> np.ndarray:
    """Soft label / prediction 0-1 -> rating 1-5, both ends clipped."""
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    return np.clip(RATING_MIN + (RATING_MAX - RATING_MIN) * p, RATING_MIN, RATING_MAX)


def round_rating(r: np.ndarray | float) -> np.ndarray:
    """Nearest whole rating 1-5; halves go toward 3 (2.5 -> 3, 3.5 -> 3), the rule every
    classification view of the ratings uses (docs/misc/targets.md §3)."""
    return quantise(r, 1.0)


def quantise(r: np.ndarray | float, step: float = 1.0) -> np.ndarray:
    """Nearest multiple of `step` within 1-5 (step 1: whole ratings; 0.5: 1.0, 1.5, ..., 5.0).
    Values exactly halfway between two steps go toward 3."""
    r = np.clip(np.asarray(r, float), RATING_MIN, RATING_MAX)
    x = r / step
    half = np.isclose(x % 1, 0.5)
    centre = 3.0 / step
    q = np.where(half, np.where(x < centre, np.ceil(x), np.floor(x)), np.rint(x))
    return np.clip(q * step, RATING_MIN, RATING_MAX)


def majority_rating(ratings: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Most common value per row of `ratings` (rows x labelers; NaN = no prediction), after each
    value is quantised to multiples of `step` (config output.majority_resolution; 0.5 keeps a split
    panel as 2.5). Ties: the value closest to 3, then the lower one. NaN for rows with no value."""
    r = np.atleast_2d(np.asarray(ratings, float))
    levels = np.arange(RATING_MIN, RATING_MAX + step / 2, step)
    out = np.full(len(r), np.nan)
    for i, row in enumerate(r):
        row = row[np.isfinite(row)]
        if not len(row):
            continue
        idx = np.rint((quantise(row, step) - RATING_MIN) / step).astype(int)
        counts = np.bincount(idx, minlength=len(levels))
        best = levels[np.flatnonzero(counts == counts.max())]
        out[i] = best[np.argmin(np.abs(best - 3) * 10 + best)]
    return out


def collapse_map(groups) -> dict[int, float] | None:
    """Config `labels.collapse` -> {rating: value}, or None when absent (the 1-5 scale as entered).

    `groups` partitions 1-5 into K >= 2 consecutive groups, lowest first: [[1, 2], [3], [4, 5]] is
    LOW / MID / HIGH. Group k is placed at the k-th of K evenly spaced points on 1-5 (K = 3: 1, 3, 5),
    so the soft targets (0, 0.5, 1), to_rating and tau_att keep their meaning: with tau_att 3,
    "engaged" is MID or HIGH, the same split as rating >= 3."""
    if not groups:
        return None
    flat = [int(r) for g in groups for r in g]
    if flat != list(range(1, 6)) or any(not g for g in groups) or len(groups) < 2:
        raise ValueError(f"labels.collapse must split 1-5 into consecutive groups in order, got {groups!r}")
    step = (RATING_MAX - RATING_MIN) / (len(groups) - 1)
    return {int(r): RATING_MIN + k * step for k, g in enumerate(groups) for r in g}


def rating_levels(groups=None) -> np.ndarray:
    """The values a label can take: 1, 2, 3, 4, 5, or the collapsed levels (e.g. 1, 3, 5)."""
    m = collapse_map(groups)
    return np.arange(RATING_MIN, RATING_MAX + 1) if m is None else np.array(sorted(set(m.values())))


def round_to_levels(r: np.ndarray | float, levels: np.ndarray | None = None) -> np.ndarray:
    """Nearest of `levels` (default: whole ratings 1-5); exact halfway ties go toward 3, as in
    round_rating. Any shape; NaN stays NaN."""
    if levels is None:
        return round_rating(r)
    levels = np.asarray(levels, float)
    a = np.asarray(r, float)
    flat = np.clip(a.ravel(), RATING_MIN, RATING_MAX)
    d = np.abs(flat[:, None] - levels[None, :])
    best = np.isclose(d, np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1, keepdims=True))
    tie = np.where(best, np.abs(levels - 3)[None, :], np.inf)      # among the nearest: closest to 3
    out = np.where(np.isnan(flat), np.nan, levels[tie.argmin(axis=1)]).reshape(a.shape)
    return out[()] if out.ndim == 0 else out

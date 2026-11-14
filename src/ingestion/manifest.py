"""Per-camera timing audit for manifest.json (design §9): measured fps, gaps, duration."""

from __future__ import annotations

import numpy as np


def timing_audit(t: np.ndarray, gap_factor: float = 3.0) -> dict:
    """Measured fps from the median frame interval; a gap is any interval > gap_factor x median."""
    if len(t) < 2:
        return {"n_frames": int(len(t)), "fps": None, "duration_s": 0.0, "gaps": []}
    dt = np.diff(t)
    median = float(np.median(dt))
    gap_idx = np.flatnonzero(dt > gap_factor * median)
    return {
        "n_frames": int(len(t)),
        "t_first": float(t[0]),
        "t_last": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "fps": 1.0 / median,
        "median_dt_s": median,
        "max_dt_s": float(dt.max()),
        "gaps": [{"t": float(t[i]), "length_s": float(dt[i])} for i in gap_idx],
    }

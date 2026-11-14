from __future__ import annotations

import numpy as np

from ..core.types import Window


def make_windows(t_first: float, t_last: float, length_s: float, overlap: float) -> list[Window]:
    """Fixed-length windows with fractional overlap; a trailing partial window is dropped."""
    if not 0 <= overlap < 1:
        raise ValueError("window.overlap must be in [0, 1)")
    step = length_s * (1 - overlap)
    n = int(np.floor((t_last - t_first - length_s) / step)) + 1
    return [Window(t_first + i * step, t_first + i * step + length_s) for i in range(max(n, 0))]


def frames_in(t: np.ndarray, w: Window) -> np.ndarray:
    """Indices of timestamps inside [t_start, t_end)."""
    return np.flatnonzero((t >= w.t_start) & (t < w.t_end))

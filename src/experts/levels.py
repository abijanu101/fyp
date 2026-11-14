"""Early vs late aggregation (config `engagement.aggregate`).

The experts always see **rows**:
  early  one row per window: the window's Vec (design §6, §8 as written);
  late   one row per (window, visible track): that track's own BFMU features, plus the window's Vec
         as classroom context when `engagement.late_context` is true. Trained on per-student labels.

Each row carries the index of its window. Prediction gives each row its window's gate
probabilities, and the window score is the mean of its rows' probabilities: for late aggregation
that estimates the share of engaged students (which is how the window label itself is defined,
§8.1); for early aggregation a window has exactly one row, so it is the window's own prediction.
The gate, mixture, metrics and outputs stay window-level in both modes, so results compare directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

TRACK_PREFIX = "trk."
VEC_PREFIX = "vec."


def decision_threshold(aggregate: str, mixture_tau: float, tau_share: float) -> float:
    """Score at which a window counts as engaged. Early: the score is P(window engaged), so
    mixture.tau. Late: the score estimates the share of engaged students, and the window label is
    share >= labels.tau_share, so the same tau_share."""
    return tau_share if aggregate == "late" else mixture_tau


def student_key(vec_key: str) -> str:
    """The per-student column a window-level mean summarises: vec.kg.stillness.mean -> trk.kg.stillness.
    Other keys are returned unchanged. Lets window-level settings (hand_weighted weights) apply per student."""
    if vec_key.startswith(VEC_PREFIX) and vec_key.endswith(".mean"):
        return TRACK_PREFIX + vec_key[len(VEC_PREFIX):-len(".mean")]
    return vec_key


@dataclass
class EngagementRows:
    X: np.ndarray          # (R, F) expert inputs
    cols: list[str]        # feature names of X's columns
    window: np.ndarray     # (R,) index of each row's window in the window table
    exp: np.ndarray        # (R,) experiment id of each row (routes per-experiment training)
    y: np.ndarray          # (R,) training label: 1/0, NaN = unlabelled


def early_rows(windows: pd.DataFrame, vec_cols: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(X, cols, window index): one row per window."""
    return windows[vec_cols].to_numpy(float), list(vec_cols), np.arange(len(windows))


def late_rows(windows: pd.DataFrame, tracks: pd.DataFrame, context: bool,
              cols: list[str] | None = None) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(X, cols, window index): one row per (window, track).

    tracks: columns "window" (row index into `windows`), "tid" and trk.* per-track features.
    context: append the row's window Vec (vec.* columns). cols: fix the column list (at inference,
    the trained model's columns; missing ones become NaN)."""
    if cols is None:
        cols = sorted(c for c in tracks if c.startswith(TRACK_PREFIX))
        if context:
            cols += sorted(c for c in windows if c.startswith(VEC_PREFIX))
    w = tracks["window"].to_numpy(int)
    table = tracks.reset_index(drop=True)
    vec_needed = [c for c in cols if c.startswith(VEC_PREFIX)]
    if vec_needed:
        table = pd.concat([table, windows.reindex(columns=vec_needed).iloc[w].reset_index(drop=True)], axis=1)
    return table.reindex(columns=cols).to_numpy(float), list(cols), w


def window_scores(p_rows: np.ndarray, row_window: np.ndarray, n_windows: int, fill: float) -> np.ndarray:
    """Mean row probability per window; windows without rows (no visible track) get `fill`."""
    total = np.bincount(row_window, weights=p_rows, minlength=n_windows)
    count = np.bincount(row_window, minlength=n_windows)
    return np.where(count > 0, total / np.maximum(count, 1), fill)


def fit_rows(make_mixture: Callable[[], object], rows: EngagementRows, train_windows: np.ndarray):
    """Fit a fresh mixture on the labelled rows of the training windows. Returns (mixture, fill):
    fill is the training label rate, used for windows with no rows (no visible track)."""
    train = np.isin(rows.window, train_windows) & np.isfinite(rows.y)
    mix = make_mixture()
    mix.fit(rows.X[train], rows.y[train].astype(int), rows.exp[train])
    return mix, (float(rows.y[train].mean()) if train.any() else 0.5)


def fit_predict(make_mixture: Callable[[], object], rows: EngagementRows, train_windows: np.ndarray,
                test_windows: np.ndarray, g_test: np.ndarray) -> np.ndarray:
    """fit_rows + predict_windows: one engagement probability per test window (in `test_windows`
    order). g_test: (len(test_windows), K) context probabilities."""
    mix, fill = fit_rows(make_mixture, rows, train_windows)
    return predict_windows(mix, rows, test_windows, g_test, fill)


def predict_windows(mix, rows: EngagementRows, windows: np.ndarray, g: np.ndarray, fill: float) -> np.ndarray:
    """One probability per window in `windows`: each row gets its window's context g[...], and a
    window's score is the mean over its rows."""
    pos = np.full(int(max(rows.window.max(initial=0), windows.max(initial=0))) + 1, -1)
    pos[windows] = np.arange(len(windows))
    sel = np.isin(rows.window, windows)
    local = pos[rows.window[sel]]
    p_rows = mix.predict_proba(rows.X[sel], g[local]) if sel.any() else np.zeros(0)
    return window_scores(p_rows, local, len(windows), fill)

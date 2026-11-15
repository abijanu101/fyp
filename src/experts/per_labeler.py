"""Per-labeler training (config engagement.labelers = "per_labeler"; docs/misc/targets.md §7).

Training rows are windows of length window.length_s centred on each rating entry (session-start
entry excluded); a row is the rated student's own track features in that window, plus the window's
Vec when engagement.late_context is on. Every (labeler, entry) is its own row, used only by that
labeler's model; nothing is averaged in training. The target is a soft label in [0, 1] (the
normalised rating; labels.train_target picks the entry's rating or that labeler's interpolated
curve over the window). A model's output p reads back as the rating 1 + 4p.

Prediction (config output.prediction):
  per_labeler  every labeler model's rating, and their median
  majority     per student and window, the most common rating across the labeler models after
               quantising each to output.majority_resolution (0.5: 1.0, 1.5, ..., 5.0)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..core.types import SCENE_PREFIX, TRACK_PREFIX, VEC_PREFIX, Window
from ..labels.interp import majority_rating, to_rating

SELF_LABELER = "self_labeling"


def entry_windows(entries: pd.DataFrame, length_s: float) -> tuple[np.ndarray, list[Window]]:
    """Unique entry times and the windows centred on them (entries at one instant share a window)."""
    centres = np.unique(entries["t"].to_numpy(float))
    return centres, [Window(c - length_s / 2, c + length_s / 2) for c in centres]


def label_window_features(vec: list[dict], track_rows: list[dict], scene: list[dict],
                          centres: np.ndarray) -> pd.DataFrame:
    """Feature table of the centred windows: one row per (window centre, visible track), with
    t_centre, tid, trk.*, vec.* and scene.* columns (cached per session by the caller)."""
    rows = []
    for r in track_rows:
        i = r["window"]
        rows.append({"t_centre": float(centres[i]), "tid": int(r["tid"]),
                     **{TRACK_PREFIX + k: v for k, v in r.items() if k not in ("window", "tid")},
                     **{VEC_PREFIX + k: v for k, v in vec[i].items()},
                     **{SCENE_PREFIX + k: v for k, v in scene[i].items()}})
    return pd.DataFrame(rows)


def join_entries(entries: pd.DataFrame, feats: pd.DataFrame, tid_to_sid: dict[int, str]) -> pd.DataFrame:
    """One training row per (entry, linked track of the rated student visible in the entry's window).
    Entries whose student has no linked visible track in that window are dropped."""
    if feats.empty or entries.empty:
        return pd.DataFrame()
    f = feats.copy()
    f["sid"] = f["tid"].map(lambda t: tid_to_sid.get(int(t)))
    f = f.dropna(subset=["sid"])
    f["_key"] = f["t_centre"].astype(float).round(3)         # centres survive a CSV round trip
    e = entries.assign(_key=entries["t"].astype(float).round(3))
    return e.merge(f.drop(columns="t_centre"), on=["_key", "sid"], how="inner").drop(columns="_key")


class PerLabelerModel:
    """One mixture per labeler, each trained only on its own rows (soft targets)."""

    def __init__(self, mixtures: dict[str, object], fills: dict[str, float]):
        self.mixtures = mixtures            # {labeler: fitted ExpertMixture}
        self.fills = fills                  # {labeler: mean training target}: windows with no rows

    @classmethod
    def fit(cls, make_mixture: Callable[[], object], X: np.ndarray, target: np.ndarray, exp: np.ndarray,
            labeler: np.ndarray) -> "PerLabelerModel":
        mixtures, fills = {}, {}
        for name in sorted(set(labeler.tolist())):
            m = labeler == name
            if not m.any():
                continue
            mixtures[name] = make_mixture().fit(X[m], np.clip(target[m], 0, 1), exp[m])
            fills[name] = float(np.clip(target[m], 0, 1).mean())
        if not mixtures:
            raise ValueError("no labeler has training rows")
        return cls(mixtures, fills)

    @property
    def labelers(self) -> list[str]:
        return list(self.mixtures)

    def predict(self, X: np.ndarray, g: np.ndarray) -> dict[str, np.ndarray]:
        """{labeler: p in [0, 1] per row}."""
        return {n: np.clip(m.predict_proba(X, g), 0, 1) for n, m in self.mixtures.items()}


def window_curves(model: PerLabelerModel, X: np.ndarray, g_rows: np.ndarray, row_window: np.ndarray,
                  n_windows: int, mode: str, resolution: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Class-level rating curves over windows from per-student rows.

    per_labeler: each labeler's curve = mean over the window's students of that model's rating;
                 the final curve is the median over labeler curves.
    majority:    per student, the majority rating across labeler models (quantised to `resolution`);
                 the final curve is its mean over the window's students. Labeler curves are also
                 returned (for display), computed as in per_labeler.
    Windows without rows get each labeler's training mean (fill)."""
    if mode not in ("per_labeler", "majority"):
        raise ValueError(f"output.prediction must be 'per_labeler' or 'majority', got {mode!r}")
    preds = model.predict(X, g_rows) if len(X) else {n: np.zeros(0) for n in model.labelers}
    count = np.bincount(row_window, minlength=n_windows)
    curves = {}
    for n, p in preds.items():
        total = np.bincount(row_window, weights=to_rating(p), minlength=n_windows)
        curves[n] = np.where(count > 0, total / np.maximum(count, 1), to_rating(model.fills[n]))
    if mode == "per_labeler":
        final = np.median(np.stack(list(curves.values())), axis=0)
    else:
        maj = majority_rating(np.column_stack([to_rating(p) for p in preds.values()]), resolution) \
            if len(X) else np.zeros(0)
        total = np.bincount(row_window, weights=maj, minlength=n_windows)
        fallback = majority_rating(np.array([[to_rating(f) for f in model.fills.values()]]), resolution)[0]
        final = np.where(count > 0, total / np.maximum(count, 1), fallback)
    return final, curves

"""Per-labeler training (config engagement.labelers = "per_labeler"; docs/misc/targets.md §7).

Training rows are windows of length window.length_s centred on each rating entry (session-start
entry excluded); a row is the rated student's own track features in that window, plus the window's
Vec when engagement.use_class_context is on. Every (labeler, entry) is its own row, used only by that
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
    """One mixture per labeler, each trained only on its own rows (soft targets). With `pooled`
    (config engagement.labelers = "pooled"), one mixture is trained on every labeler's rows together
    and answers for all of them: the aggregated counterpart of the per-labeler design, evaluated
    the same way (docs/experiments/06_full_pipeline_and_labels.md, perspectivism)."""

    def __init__(self, mixtures: dict[str, object], fills: dict[str, float], pooled: bool = False):
        self.mixtures = mixtures            # {labeler: fitted ExpertMixture} (one shared object if pooled)
        self.fills = fills                  # {labeler: mean training target}: windows with no rows
        self.pooled = pooled

    @classmethod
    def fit(cls, make_mixture: Callable[[], object], X: np.ndarray, target: np.ndarray, exp: np.ndarray,
            labeler: np.ndarray, pooled: bool = False) -> "PerLabelerModel":
        names = sorted(set(labeler.tolist()))
        if not names:
            raise ValueError("no labeler has training rows")
        if pooled:
            one = make_mixture().fit(X, np.clip(target, 0, 1), exp)
            fill = float(np.clip(target, 0, 1).mean())
            return cls({n: one for n in names}, {n: fill for n in names}, pooled=True)
        mixtures, fills = {}, {}
        for name in names:
            m = labeler == name
            mixtures[name] = make_mixture().fit(X[m], np.clip(target[m], 0, 1), exp[m])
            fills[name] = float(np.clip(target[m], 0, 1).mean())
        return cls(mixtures, fills)

    @property
    def labelers(self) -> list[str]:
        return list(self.mixtures)

    def predict(self, X: np.ndarray, g: np.ndarray) -> dict[str, np.ndarray]:
        """{labeler: p in [0, 1] per row} (the same curve for every labeler when pooled)."""
        if self.pooled:
            p = np.clip(next(iter(self.mixtures.values())).predict_proba(X, g), 0, 1)
            return {n: p for n in self.mixtures}
        return {n: np.clip(m.predict_proba(X, g), 0, 1) for n, m in self.mixtures.items()}


def window_curves(model: PerLabelerModel, X: np.ndarray, g_rows: np.ndarray, row_window: np.ndarray,
                  n_windows: int, mode: str, resolution: float,
                  preds: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Class-level rating curves over windows from per-student rows.

    per_labeler: each labeler's curve = mean over the window's students of that model's rating;
                 the final curve is the median over labeler curves.
    majority:    per student, the majority rating across labeler models (quantised to `resolution`);
                 the final curve is its mean over the window's students. Labeler curves are also
                 returned (for display), computed as in per_labeler.
    Windows without rows get each labeler's training mean (fill). preds: model.predict(X, g_rows), if
    the caller already has it."""
    if mode not in ("per_labeler", "majority"):
        raise ValueError(f"output.prediction must be 'per_labeler' or 'majority', got {mode!r}")
    if preds is None:
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


def train_keep_mask(rows: pd.DataFrame, keep: dict | None, seed: int = 0) -> np.ndarray:
    """Config `labels.train_keep`: which label rows the models may train on, to rebalance the ratings.

    `keep` maps a label value (as a string: "1".."5", or the levels of labels.collapse, e.g. "1", "3",
    "5") to the probability an entry with that rating is kept; values not named keep 1.0. Lowering
    the common middle rating discourages median hugging. The draw is per entry_id, so every row of
    one entry (its tracks, cameras, augmented copies) is kept or dropped together, and is fixed by
    `seed`. Only training uses the mask: held-out entries are always scored in full."""
    if not keep:
        return np.ones(len(rows), bool)
    probs = {float(k): float(v) for k, v in keep.items()}
    if any(not 0.0 <= p <= 1.0 for p in probs.values()):
        raise ValueError(f"labels.train_keep probabilities must be in [0, 1], got {keep!r}")
    ratings = rows["rating"].to_numpy(float)
    unknown = set(probs) - set(np.unique(ratings).tolist())
    if unknown:
        raise ValueError(f"labels.train_keep names {sorted(unknown)}, which no entry has "
                         f"(ratings present: {sorted(np.unique(ratings).tolist())})")
    ids = rows["entry_id"].to_numpy()
    uniq = np.unique(ids)
    u = dict(zip(uniq, np.random.default_rng(seed).random(len(uniq))))
    return np.array([u[e] < probs.get(r, 1.0) for e, r in zip(ids, ratings)], bool)

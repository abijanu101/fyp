"""Window and per-student labels for one DIPSER session-camera (design §8.1)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..core.types import SessionKey, Window
from ..datasets import DipserIndex
from .attention import AttentionTrack, load_subject_labels
from .interp import collapse_map
from .window_labels import WindowLabel, WindowLabeler


@lru_cache(maxsize=32)
def _load(path: Path, offset_s: float, collapse: tuple | None = None) -> dict[str, AttentionTrack]:
    """A subject's labels; with `collapse` (labels.collapse as a tuple of tuples) every rating is
    mapped to its level first (interp.collapse_map), so every consumer sees the collapsed scale."""
    tracks = load_subject_labels(path, offset_s)
    m = collapse_map(collapse)
    if m is None:
        return tracks
    return {n: AttentionTrack(tr.t, np.array([m[int(v)] for v in tr.value], float)) for n, tr in tracks.items()}


def collapse_key(cfg: dict) -> tuple | None:
    """labels.collapse (cfg = the `labels` block) as a hashable key for _load; None when absent."""
    groups = cfg.get("collapse")
    return tuple(tuple(int(r) for r in g) for g in groups) if groups else None


def _students_and_labeler(index: DipserIndex, key: SessionKey, cfg: dict):
    """cfg is the config's `labels` block. students='camera' uses only subjects annotated as visible
    in this camera (abi_labels); 'all' uses every labeled subject in the session."""
    zips = index.subject_zips(key)
    if cfg["students"] == "camera":
        visible = set(index.annotations(key))
        zips = {sid: p for sid, p in zips.items() if sid in visible}
    students = {sid: _load(p, float(cfg["offset_s"]), collapse_key(cfg)) for sid, p in zips.items()}
    labeler = WindowLabeler(cfg["tau_att"], cfg["tau_share"], cfg["min_labeled_students"], cfg["exclude_self"])
    return students, labeler


def label_session(index: DipserIndex, key: SessionKey, windows: list[Window], cfg: dict) -> list[WindowLabel]:
    students, labeler = _students_and_labeler(index, key, cfg)
    return [labeler.label(students, w) for w in windows]


def student_labels_session(index: DipserIndex, key: SessionKey, windows: list[Window],
                           cfg: dict) -> dict[str, np.ndarray]:
    """{subject_id: (n_windows,) 1/0/NaN}: each student's engaged label per window."""
    students, labeler = _students_and_labeler(index, key, cfg)
    per_window = [labeler.student_labels(students, w) for w in windows]
    return {sid: np.array([pw[sid] for pw in per_window], float) for sid in students}


def labeler_labels_session(index: DipserIndex, key: SessionKey, windows: list[Window],
                           cfg: dict) -> dict[str, dict[str, np.ndarray]]:
    """{labeler: {subject_id: (n_windows,) 1/0/NaN}}: each labeler's own call per student and window
    (their time-weighted mean attention >= tau_att), without any consensus. A labeler who did not
    rate a student is absent from that student's entry, so callers can tell 'absent' from 'NaN
    before their first rating'."""
    students, labeler = _students_and_labeler(index, key, cfg)
    out: dict[str, dict[str, np.ndarray]] = {}
    for sid, tracks in students.items():
        for name, tr in tracks.items():
            m = np.array([tr.mean_over(w.t_start, w.t_end) for w in windows], float)
            out.setdefault(name, {})[sid] = np.where(np.isfinite(m), (m >= labeler.tau_att).astype(float), np.nan)
    return out


TRAIN_TARGETS = ("train_with_interp", "train_vanilla")


def label_entries(index: DipserIndex, key: SessionKey, cfg: dict) -> "pd.DataFrame":
    """Every rating entry of every labeler for the students of this session-camera's label pool, one
    row each: sid, labeler, t (seconds, camera clock), rating (as entered, clipped to 1-5; with
labels.collapse, the entry's level: 1, 3, 5 for LOW / MID / HIGH), target.

    The shared session-start entry (every labeler's first rating, at the same instant) is excluded:
    it is a judgement, but not a moment the labeler chose (docs/misc/targets.md §7.1 item 3).
    `target` is the training value of the window of length window.length_s centred on the entry,
    as a soft label in [0, 1] (config labels.train_target):
      train_with_interp  that labeler's interpolated curve (labels.interp) averaged over the window
      train_vanilla      the entry's own rating
    """
    import pandas as pd

    from .interp import normalise, window_mean

    lcfg = cfg["labels"]
    mode = lcfg["train_target"]
    if mode not in TRAIN_TARGETS:
        raise ValueError(f"labels.train_target must be one of {TRAIN_TARGETS}, got {mode!r}")
    interp = lcfg["interp"]
    half = float(cfg["window"]["length_s"]) / 2
    students, _ = _students_and_labeler(index, key, lcfg)
    rows = []
    for sid, labelers in students.items():
        tracks = {n: tr for n, tr in labelers.items() if len(tr.t)}
        if not tracks:
            continue
        start = min(tr.t[0] for tr in tracks.values())
        for name, tr in tracks.items():
            for t, v in zip(tr.t, tr.value):
                if abs(t - start) < 0.5:                          # the shared session-start entry
                    continue
                rating = float(np.clip(v, 1, 5))
                value = rating if mode == "train_vanilla" else window_mean(
                    tr.t, tr.value, t - half, t + half, interp["mode"],
                    interp.get("ramp_s", 3.0), interp.get("sigma_s", 2.0))
                rows.append({"sid": sid, "labeler": name, "t": float(t), "rating": rating,
                             "target": float(normalise(value))})
    return pd.DataFrame(rows, columns=["sid", "labeler", "t", "rating", "target"])

"""Window and per-student labels for one DIPSER session-camera (design §8.1)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..core.types import SessionKey, Window
from ..datasets import DipserIndex
from .attention import AttentionTrack, load_subject_labels
from .window_labels import WindowLabel, WindowLabeler


@lru_cache(maxsize=32)
def _load(path: Path, offset_s: float) -> dict[str, AttentionTrack]:
    return load_subject_labels(path, offset_s)


def _students_and_labeler(index: DipserIndex, key: SessionKey, cfg: dict):
    """cfg is the config's `labels` block. students='camera' uses only subjects annotated as visible
    in this camera (abi_labels); 'all' uses every labeled subject in the session."""
    zips = index.subject_zips(key)
    if cfg["students"] == "camera":
        visible = set(index.annotations(key))
        zips = {sid: p for sid, p in zips.items() if sid in visible}
    students = {sid: _load(p, float(cfg["offset_s"])) for sid, p in zips.items()}
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

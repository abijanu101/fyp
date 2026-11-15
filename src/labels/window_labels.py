"""Window engagement labels (design §8.1 steps 3-5). Training and evaluation only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import Window
from .attention import SELF_LABELER, AttentionTrack


@dataclass
class WindowLabel:
    engaged: int | None        # 1/0, or None when too few students are labeled (window dropped)
    share_engaged: float       # share of labelled students engaged; NaN if no labeled students
    n_labeled: int


class WindowLabeler:
    def __init__(self, tau_att: float = 4, tau_share: float = 0.5, min_labeled_students: int = 3,
                 exclude_self: bool = False):
        self.tau_att, self.tau_share = tau_att, tau_share
        self.min_labeled = min_labeled_students
        self.exclude_self = exclude_self

    def consensus(self, labelers: dict[str, AttentionTrack], w: Window) -> float:
        """Median across labelers of each labeler's time-weighted mean attention over the window."""
        means = [tr.mean_over(w.t_start, w.t_end) for name, tr in labelers.items()
                 if not (self.exclude_self and name == SELF_LABELER)]
        means = [m for m in means if np.isfinite(m)]
        return float(np.median(means)) if means else np.nan

    def student_labels(self, students: dict[str, dict[str, AttentionTrack]], w: Window) -> dict[str, float]:
        """Per student: 1.0 if their consensus attention >= tau_att, 0.0 if below, NaN if unlabelled.
        The same rule the window label counts with; late aggregation trains on these directly."""
        out = {}
        for sid, labelers in students.items():
            c = self.consensus(labelers, w)
            out[sid] = float(c >= self.tau_att) if np.isfinite(c) else np.nan
        return out

    def label(self, students: dict[str, dict[str, AttentionTrack]], w: Window) -> WindowLabel:
        """students: {subject_id: {labeler_name: AttentionTrack}}."""
        scores = np.array([self.consensus(lab, w) for lab in students.values()], float)
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            return WindowLabel(None, np.nan, 0)
        share = float(np.mean(scores >= self.tau_att))
        engaged = int(share >= self.tau_share) if len(scores) >= self.min_labeled else None
        return WindowLabel(engaged, share, len(scores))

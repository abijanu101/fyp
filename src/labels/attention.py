"""Attention label parsing (design §8.1 step 1).

Each labeler file is a JSON list of change events; only entries carrying `attention` change the
attention value, which then holds until the next such entry (a step function).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from ..core.clock import parse_clock

SELF_LABELER = "self_labeling"


@dataclass
class AttentionTrack:
    """Step function: value[i] holds on [t[i], t[i+1]); the last value holds indefinitely."""

    t: np.ndarray
    value: np.ndarray

    def at(self, t: float) -> float:
        """Value in force at time t; NaN before the first entry."""
        i = np.searchsorted(self.t, t, side="right") - 1
        return float(self.value[i]) if i >= 0 else np.nan

    def mean_over(self, t0: float, t1: float) -> float:
        """Time-weighted mean on [t0, t1). NaN if the window starts before the first entry
        (unlabeled for this student) or is empty."""
        if len(self.t) == 0 or t0 < self.t[0] or t1 <= t0:
            return np.nan
        edges = np.concatenate([self.t, [np.inf]])
        lo = np.clip(edges[:-1], t0, t1)
        hi = np.clip(edges[1:], t0, t1)
        return float(np.sum(self.value * (hi - lo)) / (t1 - t0))


VALID_ATTENTION = range(1, 6)


def parse_events(events: list[dict], offset_s: float = 0.0) -> AttentionTrack:
    """Turn a labeler's event list into an AttentionTrack; offset_s shifts labels onto the camera clock.

    Values outside 1-5 are dropped (the previous value keeps holding); DIPSER has at least one
    (attention "7" in group_01/experiment_03/subject_18 self_labeling, likely an emotion code)."""
    pts = sorted((parse_clock(e["datetime"]) + offset_s, int(e["attention"]))
                 for e in events if str(e.get("attention", "")).strip() != ""
                 and int(e["attention"]) in VALID_ATTENTION)
    t = np.array([p[0] for p in pts], float)
    v = np.array([p[1] for p in pts], float)
    return AttentionTrack(t, v)


def load_subject_labels(subject_zip: str | Path, offset_s: float = 0.0) -> dict[str, AttentionTrack]:
    """{labeler_name: AttentionTrack} from a DIPSER subject ZIP (labels/*.json inside)."""
    out = {}
    with zipfile.ZipFile(subject_zip) as zf:
        for member in sorted(zf.namelist()):
            p = PurePosixPath(member)
            if p.parent.name == "labels" and p.suffix == ".json":
                out[p.stem] = parse_events(json.loads(zf.read(member)), offset_s)
    return out

"""Tracklet extraction and visibility filtering (design §4 stage 4)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..core.types import TrackState, Tracklet


class TrackletBuilder:
    """Accumulates per-frame TrackStates (and per-frame BFMU signals) into Tracklets."""

    def __init__(self):
        self._rows: dict[int, list] = defaultdict(list)
        self._signals: dict[int, list[dict[str, float]]] = defaultdict(list)

    def add(self, t: float, tracks: list[TrackState], signals: dict[int, dict[str, float]]) -> None:
        for s in tracks:
            vel = s.velocity if s.velocity is not None else np.full(2, np.nan)
            self._rows[s.tid].append((t, *s.xyxy, s.conf, *vel))
            self._signals[s.tid].append(signals.get(s.tid, {}))

    def build(self) -> list[Tracklet]:
        out = []
        for tid, rows in sorted(self._rows.items()):
            a = np.asarray(rows, float)
            keys = sorted({k for d in self._signals[tid] for k in d})
            sig = {k: np.array([d.get(k, np.nan) for d in self._signals[tid]]) for k in keys}
            out.append(Tracklet(tid, a[:, 0], a[:, 1:5], a[:, 5], a[:, 6:8], sig))
        return out


def filter_visible(tracklets: list[Tracklet], frame_t: np.ndarray, fps: float,
                   min_visible_s: float, max_gap_s: float) -> list[Tracklet]:
    """Split each tracklet where the tracker lost it for longer than max_gap_s, then drop pieces
    visible for less than min_visible_s.

    A gap is measured in frames that exist but lack the track (converted to seconds with the
    measured fps), not in wall-clock time: DIPSER cameras drop frames for up to several seconds,
    and a track that survives such a drop under the same TID was never lost by the tracker.
    """
    frame_t = np.asarray(frame_t)
    out = []
    for tr in tracklets:
        if len(tr.t) == 0:
            continue
        pos = np.searchsorted(frame_t, tr.t)                  # frame index of each observation
        missing_s = (np.diff(pos) - 1) / fps
        cuts = np.flatnonzero(missing_s > max_gap_s) + 1
        for idx in np.split(np.arange(len(tr.t)), cuts):
            piece = Tracklet(tr.tid, tr.t[idx], tr.xyxy[idx], tr.conf[idx], tr.velocity[idx],
                             {k: v[idx] for k, v in tr.signals.items()})
            if len(idx) / fps >= min_visible_s:              # cumulative visible duration
                out.append(piece)
    return out

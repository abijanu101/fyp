"""Trackers (design §4): StrongSORT and BoT-SORT via BoxMOT, behind one interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.registry import Registry
from ..core.types import Detections, Frame, TrackState

TRACKERS = Registry("tracker")


class Tracker(ABC):
    @abstractmethod
    def update(self, frame: Frame, dets: Detections) -> list[TrackState]: ...


class BoxmotTracker(Tracker):
    """Wraps a BoxMOT tracker using its numpy interface.

    Kalman access (verified against BoxMOT 26.0): both trackers keep `active_tracks`, each with
    `.id` and a Kalman `.mean` whose indices 4:6 are centre velocity (vx, vy) in px/frame for
    the XYWH (BoT-SORT) and XYAH (StrongSORT) filters. If a future version changes this, the
    velocity becomes None and BFMU falls back to finite differences.
    """

    name: str

    def __init__(self, **options):
        from boxmot import create_tracker
        self.tracker = create_tracker(self.name, **options)

    def update(self, frame: Frame, dets: Detections) -> list[TrackState]:
        rows = np.column_stack([dets.xyxy, dets.conf, np.zeros(len(dets))]) if len(dets) else np.empty((0, 6))
        out = self.tracker.update(rows, frame.image)
        velocities = self._kalman_velocities()
        # BoxMOT numpy output columns: x1, y1, x2, y2, id, conf, cls, det_index
        return [TrackState(int(r[4]), r[:4].astype(float), float(r[5]), velocities.get(int(r[4])))
                for r in np.asarray(out).reshape(-1, 8)]

    # -- precomputed ReID embeddings (used by scripts/tune_tracker.py to embed each session once)
    def embed(self, image: np.ndarray, dets: Detections) -> np.ndarray:
        """(N, D) ReID embeddings for `dets`, computed exactly as the tracker does internally.
        Uses BoxMOT 26's `_resolve_input_embeddings`, a private API: tune_tracker --check verifies
        that tracking on these cached embeddings reproduces the tracker's own results."""
        if len(dets) == 0:
            return np.zeros((0, 0), np.float32)
        e = self.tracker._resolve_input_embeddings(
            geometry=dets.xyxy.astype(np.float32), embeddings=None, frame=image,
            scores=dets.conf.astype(np.float32), class_ids=np.zeros(len(dets), np.int64))
        return np.asarray(e, np.float32)

    def update_with_embeddings(self, frame: Frame, dets: Detections, embs: np.ndarray) -> list[TrackState]:
        """Like update(), but with precomputed embeddings, so the tracker runs no ReID model."""
        import torch
        from boxmot.structures import Boxes
        from boxmot.structures import Detections as BoxmotDetections
        d = BoxmotDetections(geometry=Boxes(torch.from_numpy(dets.xyxy.astype(np.float32))),
                             scores=torch.from_numpy(dets.conf.astype(np.float32)),
                             class_ids=torch.zeros(len(dets), dtype=torch.long),
                             embeddings=torch.from_numpy(np.ascontiguousarray(embs, np.float32)))
        out = self.tracker.update(d, frame.image)
        velocities = self._kalman_velocities()
        return [TrackState(int(i), b.astype(float), float(s), velocities.get(int(i)))
                for b, i, s in zip(out.geometry.values.numpy(), out.track_ids.numpy(), out.scores.numpy())]

    def _kalman_velocities(self) -> dict[int, np.ndarray]:
        out = {}
        for track in getattr(self.tracker, "active_tracks", []):
            mean = getattr(track, "mean", None)
            if mean is not None and len(mean) >= 6:
                out[int(track.id)] = np.asarray(mean[4:6], float).ravel()
        return out


@TRACKERS.register("botsort")
class BotSort(BoxmotTracker):
    name = "botsort"


@TRACKERS.register("strongsort")
class StrongSort(BoxmotTracker):
    name = "strongsort"

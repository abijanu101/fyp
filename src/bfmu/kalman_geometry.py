"""BFMU (a): Kalman/geometry motion features (design §5a). Classical, from tracker output only.

Velocities are normalised by the track's mean bbox diagonal, so units are diagonals/second
and features are comparable across rows of the classroom.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Tracklet
from .base import FEATURE_EXTRACTORS, FeatureExtractor


@FEATURE_EXTRACTORS.register("kalman_geometry")
class KalmanGeometry(FeatureExtractor):
    keys = ("kg.vx", "kg.vy", "kg.speed", "kg.accel", "kg.stillness", "kg.energy",
            "kg.events", "kg.aspect_trend", "kg.area_trend", "kg.cy_trend")

    def __init__(self, still_threshold: float = 0.05, use_kalman: bool = True):
        self.still_threshold = still_threshold
        self.use_kalman = use_kalman

    def window_features(self, tr: Tracklet, fps: float) -> dict[str, float]:
        if len(tr.t) < 3:
            return {k: np.nan for k in self.keys}
        w = tr.xyxy[:, 2] - tr.xyxy[:, 0]
        h = tr.xyxy[:, 3] - tr.xyxy[:, 1]
        diag = float(np.mean(np.hypot(w, h)))
        v = self.velocity(tr, fps) / diag                    # (T, 2) diagonals/s
        speed = np.linalg.norm(v, axis=1)
        accel = np.abs(np.diff(speed)) / np.maximum(np.diff(tr.t), 1e-6)
        duration = max(tr.t[-1] - tr.t[0], 1e-6)
        reversals = sum(np.count_nonzero(np.diff(np.sign(v[:, i])) != 0) for i in (0, 1))
        area = w * h
        return {
            "kg.vx": float(v[:, 0].mean()),
            "kg.vy": float(v[:, 1].mean()),
            "kg.speed": float(speed.mean()),
            "kg.accel": float(accel.mean()),
            "kg.stillness": float(np.mean(speed < self.still_threshold)),
            "kg.energy": float(np.sqrt(np.mean(speed ** 2))),
            "kg.events": reversals / duration,
            "kg.aspect_trend": _slope(tr.t, w / np.maximum(h, 1e-6)),
            "kg.area_trend": _slope(tr.t, area / max(area.mean(), 1e-6)),
            "kg.cy_trend": _slope(tr.t, tr.centers[:, 1] / max(h.mean(), 1e-6)),
        }

    def velocity(self, tr: Tracklet, fps: float) -> np.ndarray:
        """Centre velocity in px/s: Kalman state where exposed, finite differences otherwise."""
        fd = np.gradient(tr.centers, tr.t, axis=0)
        if not self.use_kalman:
            return fd
        kalman = tr.velocity * fps                          # px/frame -> px/s
        return np.where(np.isfinite(kalman), kalman, fd)


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(t) <= 0:
        return 0.0
    return float(np.polyfit(t - t[0], y, 1)[0])

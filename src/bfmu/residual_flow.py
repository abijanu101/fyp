"""BFMU (b): RoI-pooled residual optical flow (design §5b). Classical Farneback, no weights.

Inside each track box the box's own mean flow (whole-body translation, already captured by
the Kalman velocity) is subtracted; the residual magnitude is pooled into a fixed grid x grid
layout, normalised by box diagonal and frame interval, so output size and units do not depend
on box scale or frame rate.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core.types import Frame, Tracklet, TrackState
from .base import FEATURE_EXTRACTORS, FeatureExtractor, signal_mean


@FEATURE_EXTRACTORS.register("residual_flow")
class ResidualFlow(FeatureExtractor):
    uses_frames = True

    def __init__(self, grid: int = 2, scale: float = 0.5):
        self.grid, self.scale = grid, scale
        self.keys = tuple(f"rf.mag_{i}{j}" for i in range(grid) for j in range(grid))
        self.reset()

    def reset(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._prev_t: float | None = None

    def frame_signals(self, frame: Frame, tracks: list[TrackState]) -> dict[int, dict[str, float]]:
        gray = cv2.cvtColor(cv2.resize(frame.image, None, fx=self.scale, fy=self.scale), cv2.COLOR_BGR2GRAY)
        prev, prev_t = self._prev_gray, self._prev_t
        self._prev_gray, self._prev_t = gray, frame.t
        if prev is None or prev.shape != gray.shape:
            return {}
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dt = max(frame.t - prev_t, 1e-6)
        return {s.tid: self._pool(flow, s.xyxy * self.scale, dt) for s in tracks}

    def _pool(self, flow: np.ndarray, box: np.ndarray, dt: float) -> dict[str, float]:
        H, W = flow.shape[:2]
        x1, y1 = np.clip(np.floor(box[:2]).astype(int), 0, [W - 1, H - 1])
        x2, y2 = np.clip(np.ceil(box[2:]).astype(int), [x1 + 1, y1 + 1], [W, H])
        roi = flow[y1:y2, x1:x2]
        residual = np.linalg.norm(roi - roi.reshape(-1, 2).mean(0), axis=2)
        residual /= np.hypot(x2 - x1, y2 - y1) * dt         # diagonals/s
        rows = np.array_split(np.arange(residual.shape[0]), self.grid)
        cols = np.array_split(np.arange(residual.shape[1]), self.grid)
        return {f"rf.mag_{i}{j}": float(residual[np.ix_(r, c)].mean()) if len(r) and len(c) else np.nan
                for i, r in enumerate(rows) for j, c in enumerate(cols)}

    def window_features(self, tr: Tracklet, fps: float) -> dict[str, float]:
        return {k: signal_mean(tr.signals.get(k)) for k in self.keys}


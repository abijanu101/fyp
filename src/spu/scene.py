"""Scene-level frame features for the SPU gate (design §7). Read from frames, not tracks, so the
gate does not depend on tracking quality. The final feature set is still open; `global_stats`
is the first classical candidate (motion energy, coarse intensity layout, colour, texture)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from ..core.registry import Registry
from ..core.types import Frame

SCENE_EXTRACTORS = Registry("scene feature extractor")


class SceneFeatureExtractor(ABC):
    def reset(self) -> None:
        """Clear per-session state."""

    @abstractmethod
    def frame_features(self, frame: Frame) -> dict[str, float]: ...

    @staticmethod
    def window_features(rows: list[dict[str, float]]) -> dict[str, float]:
        """Mean and std over the window's frames."""
        if not rows:
            return {}
        keys = sorted(rows[0])
        a = np.array([[r.get(k, np.nan) for k in keys] for r in rows], float)
        out = {f"{k}.mean": float(v) for k, v in zip(keys, np.nanmean(a, 0))}
        out.update({f"{k}.std": float(v) for k, v in zip(keys, np.nanstd(a, 0))})
        return out


@SCENE_EXTRACTORS.register("global_stats")
class GlobalStats(SceneFeatureExtractor):
    def __init__(self, size: tuple[int, int] = (64, 36), grid: int = 3):
        self.size, self.grid = tuple(size), grid
        self.reset()

    def reset(self) -> None:
        self._prev: np.ndarray | None = None

    def frame_features(self, frame: Frame) -> dict[str, float]:
        small = cv2.resize(frame.image, self.size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float) / 255
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(float) / 255
        diff = float(np.abs(gray - self._prev).mean()) if self._prev is not None else np.nan
        self._prev = gray
        out = {"motion": diff,
               "sat": float(hsv[..., 1].mean()), "val": float(hsv[..., 2].mean()),
               "edges": float((cv2.Canny((gray * 255).astype(np.uint8), 50, 150) > 0).mean())}
        for i, rows in enumerate(np.array_split(gray, self.grid, axis=0)):
            for j, cell in enumerate(np.array_split(rows, self.grid, axis=1)):
                out[f"int_{i}{j}"] = float(cell.mean())
        return out

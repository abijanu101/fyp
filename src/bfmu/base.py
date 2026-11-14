"""BFMU feature-extractor contract (design §5).

An extractor has two hooks:
  frame_signals   called once per frame with pixels and active tracks; returns per-track
                  per-frame values that are stored on the Tracklet (for pixel-based families
                  such as residual flow and SCB3). Most extractors leave it empty.
  window_features called per visible track per window with the tracklet sliced to that
                  window; returns {key: value} for the keys in `keys`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.registry import Registry
from ..core.types import Frame, Tracklet, TrackState

FEATURE_EXTRACTORS = Registry("BFMU feature extractor")


class FeatureExtractor(ABC):
    keys: tuple[str, ...] = ()
    # True if frame_signals reads pixels. Such an extractor's config is part of the tracking-cache
    # key (changing it needs a new frame pass); window-only extractors re-run from cached tracklets.
    uses_frames: bool = False

    def reset(self) -> None:
        """Clear per-session state before a new session starts."""

    def frame_signals(self, frame: Frame, tracks: list[TrackState]) -> dict[int, dict[str, float]]:
        return {}

    @abstractmethod
    def window_features(self, tr: Tracklet, fps: float) -> dict[str, float]: ...


def signal_mean(x: np.ndarray | None) -> float:
    """Mean of a per-frame signal over the window, ignoring NaN; NaN if there is no value."""
    if x is None or not np.isfinite(x).any():
        return np.nan
    return float(np.nanmean(x))

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.registry import Registry
from ..core.types import Tracklet, Window

CLUSTER_EXTRACTORS = Registry("CFMU extractor")


class ClusterFeatureExtractor(ABC):
    @abstractmethod
    def window_features(self, window: Window, tracks: list[Tracklet],
                        per_track: list[dict[str, float]]) -> dict[str, float]:
        """Return `cluster.*` features to concatenate onto Vec. Inputs are the window's visible
        tracklets (sliced) and their BFMU per-track features, in the same order."""


@CLUSTER_EXTRACTORS.register("null")
class NullClusterExtractor(ClusterFeatureExtractor):
    def window_features(self, window, tracks, per_track):
        return {}

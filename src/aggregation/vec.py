"""Window aggregation into the classroom-level vector Vec (design §6).

Vec = track count
    + per BFMU key: mean, std, and (for keys with a configured threshold) fraction of tracks above it
    + coarse spatial occupancy heatmap of visible-track centroids
    + CFMU `cluster.*` features (empty while CFMU is the null extractor)

aggregate_with_tracks also returns the per-track rows Vec was built from; late aggregation
(config engagement.aggregate = "late") gives those to the experts instead of, or beside, Vec.

Every window gets the same keys; NaN marks "no visible track", imputed downstream.
Region-of-interest activity (board / explaining position), the design's weak stand-in for
visual focus of attention, is out of scope for now along with VFoA itself.
"""

from __future__ import annotations

import numpy as np

from ..bfmu.base import FeatureExtractor
from ..cfmu.extractor import ClusterFeatureExtractor
from ..core.types import Tracklet, Window


class WindowAggregator:
    def __init__(self, extractors: list[FeatureExtractor], cfmu: ClusterFeatureExtractor,
                 thresholds: dict[str, float] | None = None, heatmap_grid: int = 3,
                 min_coverage: float = 0.5):
        self.extractors = extractors
        self.cfmu = cfmu
        self.thresholds = thresholds or {}
        self.grid = heatmap_grid
        self.min_coverage = min_coverage

    @property
    def track_keys(self) -> list[str]:
        return [k for e in self.extractors for k in e.keys]

    def visible(self, window: Window, tracklets: list[Tracklet]) -> list[Tracklet]:
        """Tracks covering at least `min_coverage` of the window."""
        length = window.t_end - window.t_start
        out = []
        for tr in tracklets:
            s = tr.slice(window.t_start, window.t_end)
            if len(s.t) >= 2 and (s.t[-1] - s.t[0]) >= self.min_coverage * length:
                out.append(s)
        return out

    def aggregate(self, window: Window, tracklets: list[Tracklet], fps: float,
                  frame_wh: tuple[int, int]) -> dict[str, float]:
        return self.aggregate_with_tracks(window, tracklets, fps, frame_wh)[0]

    def aggregate_with_tracks(self, window: Window, tracklets: list[Tracklet], fps: float,
                              frame_wh: tuple[int, int]) -> tuple[dict[str, float], list[dict[str, float]]]:
        """Vec plus the per-track features it was built from ({"tid", <track_keys>} per visible track),
        which late aggregation feeds to the experts directly."""
        tracks = self.visible(window, tracklets)
        per_track = [{k: v for e in self.extractors for k, v in e.window_features(tr, fps).items()}
                     for tr in tracks]
        vec: dict[str, float] = {"n_tracks": float(len(tracks))}
        for key in self.track_keys:
            vals = np.array([pt.get(key, np.nan) for pt in per_track], float)
            ok = vals[np.isfinite(vals)]
            vec[f"{key}.mean"] = float(ok.mean()) if len(ok) else np.nan
            vec[f"{key}.std"] = float(ok.std()) if len(ok) else np.nan
            if key in self.thresholds:
                vec[f"{key}.frac"] = float(np.mean(ok > self.thresholds[key])) if len(ok) else np.nan
        vec.update(self._heatmap(tracks, frame_wh))
        vec.update(self.cfmu.window_features(window, tracks, per_track))
        rows = [{"tid": float(tr.tid), **{k: pt.get(k, np.nan) for k in self.track_keys}}
                for tr, pt in zip(tracks, per_track)]
        return vec, rows

    def _heatmap(self, tracks: list[Tracklet], frame_wh: tuple[int, int]) -> dict[str, float]:
        g = self.grid
        counts = np.zeros((g, g))
        for tr in tracks:
            cx, cy = tr.centers.mean(axis=0)
            i = min(int(cy / frame_wh[1] * g), g - 1)
            j = min(int(cx / frame_wh[0] * g), g - 1)
            counts[max(i, 0), max(j, 0)] += 1
        counts /= max(len(tracks), 1)
        return {f"occ_{i}{j}": float(counts[i, j]) for i in range(g) for j in range(g)}

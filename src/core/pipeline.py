"""Pipeline wiring (design §3). Every stage is built by registry name from the config.

Two passes, so the expensive one runs once per tracking setup:
    track()     frame pass: ingestion, detection, tracking, pixel-level BFMU signals, scene
                features -> SessionTrack (persisted by core.store; keyed by tracking_hash)
    features()  feature pass: tracklet filtering, window BFMU, aggregation (Vec), scene windows
                -> SessionFeatures (keyed by feature_hash; seconds per session from a cached track)
    predict()   stages 7-9: gate, mixture, graph records, from SessionFeatures + trained models

Training (scripts/train_models.py) and inference (scripts/run_pipeline.py) share track() and
features(), so a model is trained on features produced by exactly the code that serves it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..aggregation import WindowAggregator, frames_in, make_windows
from ..bfmu import FEATURE_EXTRACTORS
from ..cfmu import CLUSTER_EXTRACTORS
from ..detection import DETECTORS
from ..evaluation import StageTimer
from ..ingestion import FRAME_SOURCES, timing_audit
from ..output import build_graph
from ..spu import SCENE_EXTRACTORS, context_probs
from ..tracking import TRACKERS, TrackletBuilder, filter_visible
from .config import config_hash
from .types import SessionFeatures, SessionKey, SessionTrack

VEC_PREFIX, SCENE_PREFIX = "vec.", "scene."


def _build(registry, entry: dict):
    """Build from a {"name": ..., **kwargs} config entry."""
    kwargs = dict(entry)
    return registry.build(kwargs.pop("name"), **kwargs)


def _frame_level_bfmu(cfg: dict) -> list[dict]:
    return [e for e in cfg["bfmu"] if FEATURE_EXTRACTORS.get(e["name"]).uses_frames]


def tracking_hash(cfg: dict) -> str:
    """Everything the frame pass depends on. Dataset scope is excluded: it selects sessions,
    it does not change any one session's result."""
    return config_hash({k: cfg[k] for k in ("ingestion", "detector", "tracker", "scene")}
                       | {"bfmu_frame": _frame_level_bfmu(cfg)})


def feature_hash(cfg: dict) -> str:
    """Everything SessionFeatures depends on: the frame pass plus the feature-pass settings."""
    return config_hash({"tracking": tracking_hash(cfg)}
                       | {k: cfg[k] for k in ("tracklets", "bfmu", "cfmu", "window", "aggregation")})


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.hash = config_hash(cfg)
        self._detector = None
        self.extractors = [_build(FEATURE_EXTRACTORS, e) for e in cfg["bfmu"]]
        self.scene = _build(SCENE_EXTRACTORS, cfg["scene"])
        agg = cfg["aggregation"]
        self.aggregator = WindowAggregator(
            self.extractors, _build(CLUSTER_EXTRACTORS, cfg["cfmu"]),
            thresholds=agg.get("thresholds"), heatmap_grid=agg.get("heatmap_grid", 3),
            min_coverage=agg.get("min_coverage", 0.5))

    @property
    def detector(self):
        """Built on first use so the pipeline can be constructed (and tested) without a GPU."""
        if self._detector is None:
            self._detector = _build(DETECTORS, self.cfg["detector"])
        return self._detector

    def new_tracker(self):
        return TRACKERS.build(self.cfg["tracker"]["name"], **self.cfg["tracker"].get("options", {}))

    # ------------------------------------------------------------------ frame pass
    def track(self, path: str | Path, key: SessionKey | None = None) -> SessionTrack:
        timer = StageTimer()
        with timer.stage("ingestion_index"):
            source = _build(FRAME_SOURCES, {**self.cfg["ingestion"], "path": path})
        tracker = self.new_tracker()
        frame_extractors = [e for e in self.extractors if e.uses_frames]
        for e in frame_extractors:
            e.reset()
        self.scene.reset()
        builder = TrackletBuilder()
        frame_t, scene_rows, frame_wh = [], [], (1, 1)

        frames = iter(source)
        while True:
            with timer.stage("ingestion_decode"):
                frame = next(frames, None)
            if frame is None:
                break
            frame_wh = (frame.image.shape[1], frame.image.shape[0])
            with timer.stage("detection"):
                dets = self.detector.detect(frame)
            with timer.stage("tracking"):
                tracks = tracker.update(frame, dets)
            signals: dict[int, dict[str, float]] = {}
            for e in frame_extractors:                      # timed per extractor: they differ a lot in cost
                with timer.stage(f"bfmu_frame.{type(e).__name__}"):
                    for tid, vals in e.frame_signals(frame, tracks).items():
                        signals.setdefault(tid, {}).update(vals)
            builder.add(frame.t, tracks, signals)
            with timer.stage("scene"):
                frame_t.append(frame.t)
                scene_rows.append(self.scene.frame_features(frame))

        return SessionTrack(key, np.asarray(frame_t, float), frame_wh, builder.build(), scene_rows,
                            timing_audit(np.asarray(frame_t, float)), timer.report())

    # ------------------------------------------------------------------ feature pass
    def features(self, st: SessionTrack) -> SessionFeatures:
        timer = StageTimer()
        fps = st.timing.get("fps") or 1.0
        tl = self.cfg["tracklets"]
        with timer.stage("tracklets"):
            tracklets = filter_visible(st.tracklets, st.frame_t, fps, tl["min_visible_s"], tl["max_gap_s"])
        win = self.cfg["window"]
        windows = make_windows(st.timing.get("t_first", 0.0), st.timing.get("t_last", 0.0),
                               win["length_s"], win["overlap"])
        with timer.stage("aggregation"):
            vec = [self.aggregator.aggregate(w, tracklets, fps, st.frame_wh) for w in windows]
            scene = [self.scene.window_features([st.scene_rows[i] for i in frames_in(st.frame_t, w)])
                     for w in windows]
        manifest = {"session": str(st.key) if st.key else None, "timing": st.timing,
                    "n_raw_tracks": len(st.tracklets), "n_tracklets": len(tracklets),
                    "frame_wh": list(st.frame_wh), "stage_seconds": {**st.stage_seconds, **timer.report()},
                    "config_hash": self.hash, "tracking_hash": tracking_hash(self.cfg),
                    "feature_hash": feature_hash(self.cfg)}
        return SessionFeatures(st.key, windows, vec, scene, manifest)

    def extract(self, path: str | Path, key: SessionKey | None = None) -> SessionFeatures:
        return self.features(self.track(path, key))

    # ------------------------------------------------------------------ stages 7-9
    def predict(self, feats: SessionFeatures, models: dict) -> list[dict]:
        """models: bundle saved by scripts/train_models.py (gate, mixture, column lists)."""
        df = to_frame(feats)
        X_vec = df.reindex(columns=models["vec_cols"]).to_numpy(float)
        X_gate = df.reindex(columns=models["gate_cols"]).to_numpy(float)
        classes = self.cfg["classes"]
        g = context_probs("classified", classes, gate=models["gate"], X_gate=X_gate)
        p = models["mixture"].predict_proba(X_vec, g)
        return build_graph(feats.windows, g, classes, p, df[VEC_PREFIX + "n_tracks"].to_numpy(),
                           tau=self.cfg["mixture"]["tau"])


def to_frame(feats: SessionFeatures) -> pd.DataFrame:
    """One row per window: t_start, t_end, vec.* and scene.* columns."""
    rows = [{"t_start": w.t_start, "t_end": w.t_end,
             **{VEC_PREFIX + k: v for k, v in vec.items()},
             **{SCENE_PREFIX + k: v for k, v in sc.items()}}
            for w, vec, sc in zip(feats.windows, feats.vec, feats.scene)]
    return pd.DataFrame(rows)

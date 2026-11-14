"""BFMU (c): SCB student-behaviour detectors (design §5c).

Runs one or more pretrained SCB YOLOv7 checkpoints on each frame and turns their output into
per-track, per-frame signals. Two modes:

  soft (default)  No thresholds and no NMS. Every candidate box is assigned to the track it
                  overlaps best (IoU), provided it lies mostly inside that track's box
                  (containment >= min_overlap). Per track and frame, each class signal is the
                  highest objectness x class probability among its candidates, plus the highest
                  objectness alone; 0 when nothing landed on the track. The detector's classes do
                  not have to be right on DIPSER: the experts learn what its responses (and its
                  confusions, since YOLOv7 class probabilities are independent sigmoids and can
                  co-fire) mean here, so the decision is made in-domain.
  flags           Thresholded (conf / class_conf) and NMS'd detections, Hungarian-matched to
                  tracks per class on containment; a 0/1 flag per class. Kept for comparison.

Window features per track: mean of each signal over the window's frames, and in soft mode also
its peak (`.max`), which keeps brief events such as a raised hand.

Config (bfmu entry):
    {"name": "scb", "mode": "soft", "min_overlap": 0.5,
     "models": [{"weights": "...best.pt", "classes": ["hand_raising", "read", "write"],
                 "conf": 0.25, "class_conf": {"hand_raising": 0.6}}]}
`conf`/`class_conf` only apply in flags mode (pick values with scripts/dipser/scb_viewer.py).
`classes` must match the checkpoint's class order (names lower-cased, '-'/' ' -> '_'); this is
checked when the model loads, and keeps feature keys known without loading any weights.
With several models, give each a short "tag" (default m0, m1, ...) for its objectness key.

The DIPSER domain sanity check (design §11): scripts/dipser/check_scb.py, scripts/dipser/scb_viewer.py.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core.types import Frame, Tracklet, TrackState
from ..tracking.association import containment_matrix, iou_matrix
from .base import FEATURE_EXTRACTORS, FeatureExtractor, signal_mean


def normalize_class(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


@FEATURE_EXTRACTORS.register("scb")
class ScbBehaviour(FeatureExtractor):
    uses_frames = True

    def __init__(self, models: list[dict], mode: str = "soft", min_overlap: float = 0.5,
                 obj_floor: float = 0.01, device: str | None = None):
        if mode not in ("soft", "flags"):
            raise ValueError(f"scb mode must be 'soft' or 'flags', got {mode!r}")
        self.specs = [dict(m) for m in models]
        self.mode, self.min_overlap, self.obj_floor, self.device = mode, min_overlap, obj_floor, device
        self.thresholds = [[m.get("class_conf", {}).get(c, m.get("conf", 0.25)) for c in m["classes"]]
                           for m in self.specs]
        self.signal_keys = [f"scb.{c}" for m in self.specs for c in m["classes"]]
        if mode == "soft":
            self.signal_keys += [f"scb.{m.get('tag', f'm{i}')}.obj" for i, m in enumerate(self.specs)]
            self.keys = tuple(k + suffix for k in self.signal_keys for suffix in ("", ".max"))
        else:
            self.keys = tuple(self.signal_keys)
        self._detectors = None

    @property
    def detectors(self):
        """Loaded on first use, so building the pipeline (and tests) needs no weights or GPU."""
        if self._detectors is None:
            from .yolov7 import Yolov7Detector
            self._detectors = []
            for spec, thr in zip(self.specs, self.thresholds):
                det = Yolov7Detector(spec["weights"], conf=min(thr), iou=spec.get("iou", 0.45),
                                     imgsz=spec.get("imgsz", 640), device=self.device)
                found = [normalize_class(n) for n in det.names]
                if found != list(spec["classes"]):
                    raise ValueError(f"{spec['weights']}: config classes {spec['classes']} != checkpoint {found}")
                self._detectors.append(det)
        return self._detectors

    def frame_signals(self, frame: Frame, tracks: list[TrackState]) -> dict[int, dict[str, float]]:
        if not tracks:
            return {}
        boxes = np.stack([s.xyxy for s in tracks])
        tids = [s.tid for s in tracks]
        out = {tid: {k: 0.0 for k in self.signal_keys} for tid in tids}
        for i, (det, spec, thr) in enumerate(zip(self.detectors, self.specs, self.thresholds)):
            if self.mode == "soft":
                xyxy, obj, probs = det.candidates(frame.image, self.obj_floor)
                scores = np.column_stack([obj[:, None] * probs, obj])          # (N, nc + 1)
                names = [f"scb.{c}" for c in spec["classes"]] + [f"scb.{spec.get('tag', f'm{i}')}.obj"]
                for tid, best in zip(tids, self.soft_scores(xyxy, scores, boxes)):
                    out[tid].update(zip(names, map(float, best)))
            else:
                xyxy, conf, cls = det.detect(frame.image)
                for ci, name in enumerate(spec["classes"]):
                    keep = (cls == ci) & (conf >= thr[ci])
                    for tid in self.match(xyxy[keep], boxes, tids):
                        out[tid][f"scb.{name}"] = 1.0
        return out

    def soft_scores(self, cand_boxes: np.ndarray, cand_scores: np.ndarray, track_boxes: np.ndarray) -> np.ndarray:
        """(T, S) per-track maximum of the candidates' scores; each candidate goes to its best-IoU
        track if it lies at least min_overlap inside it. Tracks with no candidate get 0."""
        out = np.zeros((len(track_boxes), cand_scores.shape[1]))
        if len(cand_boxes) == 0:
            return out
        best = iou_matrix(cand_boxes, track_boxes).argmax(axis=1)
        inside = containment_matrix(cand_boxes, track_boxes)[np.arange(len(cand_boxes)), best] >= self.min_overlap
        for t in range(len(track_boxes)):
            m = inside & (best == t)
            if m.any():
                out[t] = cand_scores[m].max(axis=0)
        return out

    def match(self, behaviour_boxes: np.ndarray, track_boxes: np.ndarray, tids: list[int]) -> list[int]:
        """Flags mode: track ids that received a behaviour box of one class."""
        if len(behaviour_boxes) == 0:
            return []
        cover = containment_matrix(behaviour_boxes, track_boxes)
        rows, cols = linear_sum_assignment(-cover)
        return [tids[c] for r, c in zip(rows, cols) if cover[r, c] >= self.min_overlap]

    def window_features(self, tr: Tracklet, fps: float) -> dict[str, float]:
        out = {}
        for k in self.signal_keys:
            x = tr.signals.get(k)
            out[k] = signal_mean(x)
            if self.mode == "soft":
                out[k + ".max"] = float(np.nanmax(x)) if x is not None and np.isfinite(x).any() else np.nan
        return out

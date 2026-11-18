"""BFMU (d): head direction from a keypoint pose model (experiment 09, visual focus of attention).

Runs an ultralytics YOLO pose model (COCO keypoints) and reads, per tracked student and frame, where
the head points relative to the body, from five face keypoints and the shoulders:

  hp.yaw          nose x minus the midpoint of the reference pair, over the pair's width. The pair is
                  the ears when both are visible, else the shoulders. 0 = facing the camera's axis;
                  the sign says which side (image left negative). Mirrored copies flip it.
  hp.head_height  shoulder midpoint y minus nose y, over shoulder width. Smaller = head lowered (or
                  slumped), larger = head up.
  hp.face_conf    mean keypoint confidence of the nose and both eyes: the face is visible to this
                  camera (facing it) when high, turned away when low.
  hp.ear_asym     left-ear confidence minus right-ear confidence: one ear visible = head turned.
  hp.matched      1 if a pose instance was matched to the track in this frame, else 0.
Values are NaN where the needed keypoints are below `kp_conf` (and all NaN, matched 0, when no pose
matched the track).

Input modes (the pilot, scripts/analysis/head_pose_pilot.py, compares frame and crops; heads are
about 35-50 px wide in the 1280x720 frames):
  tracks  no model of its own: the person detector is the pose model (detector.weights a *-pose.pt),
          and each track carries the keypoints of the detection its update used (the tracker's
          det_index). One network per frame, and the pairing is exact. Experiment 09's setting.
  frame   the whole frame at `imgsz` (640 halves the heads, 1280 keeps them);
  crops   the upper part of each track box (top `crop_top` of its height, plus `crop_margin`),
          resized to `crop_size`, batched; keypoints mapped back to the frame.
Pose instances are matched to tracks one-to-one by box IoU (Hungarian), at least `min_iou`.

Window features per track: mean and SD of yaw and head height, mean |yaw|, the share of frames with
the head low (head height below `low_head`), mean face confidence, mean ear asymmetry, and the share
of frames with a matched pose.

Config (bfmu entry):
    {"name": "head_pose", "weights": "models/yolo11n-pose.pt", "mode": "frame", "imgsz": 1280,
     "conf": 0.1, "kp_conf": 0.5, "min_iou": 0.3}
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core.types import Frame, Tracklet, TrackState
from ..tracking.association import iou_matrix
from .base import FEATURE_EXTRACTORS, FeatureExtractor, signal_mean

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO = range(7)
SIGNALS = ("hp.yaw", "hp.head_height", "hp.face_conf", "hp.ear_asym", "hp.matched")


def head_geometry(xy: np.ndarray, conf: np.ndarray, kp_conf: float = 0.5) -> dict[str, float]:
    """Head direction from one person's COCO keypoints: xy (17, 2) in pixels, conf (17,)."""
    ok = conf >= kp_conf
    out = {k: np.nan for k in SIGNALS}
    out["hp.matched"] = 1.0
    out["hp.face_conf"] = float(np.mean(conf[[NOSE, L_EYE, R_EYE]]))
    out["hp.ear_asym"] = float(conf[L_EAR] - conf[R_EAR])
    shoulders = ok[L_SHO] and ok[R_SHO]
    sho_w = abs(xy[L_SHO, 0] - xy[R_SHO, 0]) if shoulders else np.nan
    if ok[NOSE]:
        if ok[L_EAR] and ok[R_EAR] and abs(xy[L_EAR, 0] - xy[R_EAR, 0]) > 1:
            mid, w = (xy[L_EAR, 0] + xy[R_EAR, 0]) / 2, abs(xy[L_EAR, 0] - xy[R_EAR, 0])
            out["hp.yaw"] = float((xy[NOSE, 0] - mid) / w)
        elif shoulders and sho_w > 1:
            mid = (xy[L_SHO, 0] + xy[R_SHO, 0]) / 2
            out["hp.yaw"] = float((xy[NOSE, 0] - mid) / sho_w)
        if shoulders and sho_w > 1:
            sho_y = (xy[L_SHO, 1] + xy[R_SHO, 1]) / 2
            out["hp.head_height"] = float((sho_y - xy[NOSE, 1]) / sho_w)
    return out


def match_poses(pose_boxes: np.ndarray, track_boxes: np.ndarray, min_iou: float) -> dict[int, int]:
    """One-to-one {track index: pose index} by maximum total IoU, keeping pairs with IoU >= min_iou."""
    if len(pose_boxes) == 0 or len(track_boxes) == 0:
        return {}
    iou = iou_matrix(pose_boxes, track_boxes)
    rows, cols = linear_sum_assignment(-iou)
    return {int(c): int(r) for r, c in zip(rows, cols) if iou[r, c] >= min_iou}


@FEATURE_EXTRACTORS.register("head_pose")
class HeadPose(FeatureExtractor):
    uses_frames = True

    def __init__(self, weights: str = "models/yolo11n-pose.pt", mode: str = "frame", imgsz: int = 1280,
                 conf: float = 0.1, kp_conf: float = 0.5, min_iou: float = 0.3, crop_top: float = 0.6,
                 crop_margin: float = 0.15, crop_size: int = 256, low_head: float = 0.1, half: bool = True,
                 device: str | None = None):
        if mode not in ("tracks", "frame", "crops"):
            raise ValueError(f"head_pose mode must be 'tracks', 'frame' or 'crops', got {mode!r}")
        self.weights, self.mode, self.imgsz, self.conf, self.kp_conf = weights, mode, imgsz, conf, kp_conf
        self.min_iou, self.crop_top, self.crop_margin, self.crop_size = min_iou, crop_top, crop_margin, crop_size
        self.low_head, self.half, self.device = low_head, half, device
        self.signal_keys = SIGNALS
        self.keys = ("hp.yaw.mean", "hp.yaw.std", "hp.yaw.abs", "hp.head_height.mean", "hp.head_height.std",
                     "hp.head_low", "hp.face_conf", "hp.ear_asym", "hp.matched")
        self._model = None

    @property
    def model(self):
        """Loaded on first use, so building the pipeline (and tests) needs no weights or GPU."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def _predict(self, images, imgsz: int):
        kw = {"conf": self.conf, "imgsz": imgsz, "verbose": False, "half": self.half}
        if self.device is not None:
            kw["device"] = self.device
        return self.model.predict(images, **kw)

    def frame_signals(self, frame: Frame, tracks: list[TrackState]) -> dict[int, dict[str, float]]:
        if not tracks:
            return {}
        out = {s.tid: {k: (0.0 if k == "hp.matched" else np.nan) for k in SIGNALS} for s in tracks}
        if self.mode == "tracks":                 # keypoints came with the detection (pose detector)
            for s in tracks:
                if s.keypoints is not None:
                    out[s.tid] = head_geometry(np.asarray(s.keypoints, float), np.asarray(s.kp_conf, float),
                                               self.kp_conf)
            return out
        boxes = np.stack([s.xyxy for s in tracks]).astype(float)
        if self.mode == "frame":
            res = self._predict(frame.image, self.imgsz)[0]
            if res.keypoints is None or len(res.boxes) == 0:
                return out
            pose_boxes = res.boxes.xyxy.cpu().numpy()
            kxy, kc = res.keypoints.xy.cpu().numpy(), res.keypoints.conf.cpu().numpy()
            for t, p in match_poses(pose_boxes, boxes, self.min_iou).items():
                out[tracks[t].tid] = head_geometry(kxy[p], kc[p], self.kp_conf)
            return out
        crops, origins = self._crops(frame.image, boxes)
        for t, (res, (x0, y0, sx, sy)) in enumerate(zip(self._predict(crops, self.crop_size), origins)):
            if res.keypoints is None or len(res.boxes) == 0:
                continue
            # the instance whose box best covers the crop's centre column: the tracked student
            pb = res.boxes.xyxy.cpu().numpy()
            cx = self.crop_size / 2
            p = int(np.argmin(np.abs((pb[:, 0] + pb[:, 2]) / 2 - cx)))
            kxy = res.keypoints.xy.cpu().numpy()[p] * [sx, sy] + [x0, y0]
            out[tracks[t].tid] = head_geometry(kxy, res.keypoints.conf.cpu().numpy()[p], self.kp_conf)
        return out

    def _crops(self, image: np.ndarray, boxes: np.ndarray):
        """Upper-body crops of each track box, resized to crop_size; (crops, (x0, y0, sx, sy) each)."""
        import cv2
        h, w = image.shape[:2]
        crops, origins = [], []
        for x1, y1, x2, y2 in boxes:
            bw, bh = x2 - x1, (y2 - y1) * self.crop_top
            m = self.crop_margin
            cx1, cy1 = max(0, int(x1 - m * bw)), max(0, int(y1 - m * bh))
            cx2, cy2 = min(w, int(x2 + m * bw)), min(h, int(y1 + bh + m * bh))
            if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                cx1, cy1, cx2, cy2 = 0, 0, min(w, 4), min(h, 4)
            crop = cv2.resize(image[cy1:cy2, cx1:cx2], (self.crop_size, self.crop_size))
            crops.append(crop)
            origins.append((cx1, cy1, (cx2 - cx1) / self.crop_size, (cy2 - cy1) / self.crop_size))
        return crops, origins

    def window_features(self, tr: Tracklet, fps: float) -> dict[str, float]:
        def sig(k):
            return tr.signals.get(k)
        yaw, hh = sig("hp.yaw"), sig("hp.head_height")
        def sd(x):
            return float(np.nanstd(x)) if x is not None and np.isfinite(x).sum() >= 2 else np.nan
        low = np.nan
        if hh is not None and np.isfinite(hh).any():
            low = float(np.mean(hh[np.isfinite(hh)] < self.low_head))
        return {"hp.yaw.mean": signal_mean(yaw), "hp.yaw.std": sd(yaw),
                "hp.yaw.abs": signal_mean(np.abs(yaw)) if yaw is not None else np.nan,
                "hp.head_height.mean": signal_mean(hh), "hp.head_height.std": sd(hh), "hp.head_low": low,
                "hp.face_conf": signal_mean(sig("hp.face_conf")), "hp.ear_asym": signal_mean(sig("hp.ear_asym")),
                "hp.matched": signal_mean(sig("hp.matched"))}

"""Dataclasses that stages exchange. Stages talk to each other only through these.

Time convention: every timestamp is seconds since local midnight (float). Frame
filenames and labeler JSON use the same wall clock, so no conversion is needed
beyond the configurable label offset (design §8.1 step 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, order=True)
class SessionKey:
    """One session-camera instance, e.g. group_01/experiment_03/context_cam_02."""

    group: str
    experiment: str
    camera: str

    @property
    def exp_id(self) -> str:
        """Two-digit experiment id ('01'..'09'), the context class label."""
        return self.experiment.rsplit("_", 1)[1]

    def __str__(self) -> str:
        return f"{self.group}/{self.experiment}/{self.camera}"


@dataclass
class Frame:
    index: int
    t: float                 # seconds since midnight
    image: np.ndarray        # HxWx3 uint8, BGR


@dataclass
class Detections:
    """Per-frame person detections."""

    xyxy: np.ndarray         # (N, 4) float
    conf: np.ndarray         # (N,)
    keypoints: np.ndarray | None = None   # (N, 17, 2) COCO keypoints, when the detector is a pose model
    kp_conf: np.ndarray | None = None     # (N, 17) their confidences

    def __len__(self) -> int:
        return len(self.conf)

    def take(self, keep: np.ndarray) -> "Detections":
        """The detections selected by a boolean mask or indices, keypoints included."""
        kp = None if self.keypoints is None else self.keypoints[keep]
        kc = None if self.kp_conf is None else self.kp_conf[keep]
        return Detections(self.xyxy[keep], self.conf[keep], kp, kc)

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4)), np.zeros(0))


@dataclass
class TrackState:
    """One active track in one frame, as reported by the tracker."""

    tid: int
    xyxy: np.ndarray         # (4,)
    conf: float
    velocity: np.ndarray | None = None   # Kalman centre velocity (vx, vy) in px/frame, if exposed
    keypoints: np.ndarray | None = None  # (17, 2) of the detection this frame's update used (pose detector)
    kp_conf: np.ndarray | None = None    # (17,)


# Column prefixes of the feature tables (one definition, used by the pipeline and the experts).
VEC_PREFIX, SCENE_PREFIX, TRACK_PREFIX = "vec.", "scene.", "trk."


@dataclass
class Tracklet:
    """Per-track time series (design §4). `signals` holds per-frame BFMU channels
    (e.g. residual flow, SCB scores) aligned with `t`."""

    tid: int
    t: np.ndarray            # (T,)
    xyxy: np.ndarray         # (T, 4)
    conf: np.ndarray         # (T,)
    velocity: np.ndarray     # (T, 2); NaN where the tracker did not expose Kalman state
    signals: dict[str, np.ndarray] = field(default_factory=dict)

    def take(self, idx) -> "Tracklet":
        """The observations selected by `idx` (boolean mask or indices), signals included."""
        return Tracklet(self.tid, self.t[idx], self.xyxy[idx], self.conf[idx], self.velocity[idx],
                        {k: v[idx] for k, v in self.signals.items()})

    def slice(self, t0: float, t1: float) -> "Tracklet":
        return self.take((self.t >= t0) & (self.t < t1))

    @property
    def centers(self) -> np.ndarray:
        return np.stack([(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2,
                         (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2], axis=1)


@dataclass(frozen=True)
class Window:
    t_start: float
    t_end: float


@dataclass
class SessionTrack:
    """Output of the frame pass (stages 1-5 frame-level hooks) for one session; persisted to disk
    so the cheap feature pass (filtering, window BFMU, aggregation) can re-run without video.

    `tracklets` are RAW tracker output (visibility filtering happens in the feature pass, so its
    thresholds can change without re-tracking). `frame_t` lists every decoded frame, which lets
    gap filtering tell tracker losses apart from camera frame drops."""

    key: SessionKey | None
    frame_t: np.ndarray
    frame_wh: tuple[int, int]
    tracklets: list[Tracklet]
    scene_rows: list[dict[str, float]]   # per frame, aligned with frame_t
    timing: dict
    stage_seconds: dict[str, float]


@dataclass
class SessionFeatures:
    """Output of the shared extraction stages (1-6 plus scene features) for one session.

    `vec[i]` is the window feature vector Vec (design §6); `scene[i]` is the SPU
    gate's frame-level input (design §7). Both are flat {key: float} dicts.
    """

    key: SessionKey | None
    windows: list[Window]
    vec: list[dict[str, float]]
    scene: list[dict[str, float]]
    manifest: dict
    # One row per (window, visible track): {"window": i, "tid": ..., <per-track BFMU features>}.
    # Used by late aggregation (config engagement.aggregate = "late"); never part of any output.
    track_rows: list[dict[str, float]] = field(default_factory=list)

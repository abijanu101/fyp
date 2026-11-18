"""Shared person detector (design §4): one detector feeds every tracker, so tracker comparisons
isolate tracker behaviour."""

from __future__ import annotations

from abc import ABC, abstractmethod


from ..core.registry import Registry
from ..core.types import Detections, Frame

DETECTORS = Registry("detector")


class PersonDetector(ABC):
    @abstractmethod
    def detect(self, frame: Frame) -> Detections: ...


@DETECTORS.register("yolo")
class YoloDetector(PersonDetector):
    """Ultralytics YOLO restricted to COCO class 0 (person)."""

    def __init__(self, weights: str, conf: float, imgsz: int = 640, device: str | None = None,
                 half: bool = True):
        import torch
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.kw = dict(classes=[0], conf=conf, imgsz=imgsz, device=self.device, verbose=False,
                       quantize="fp16" if half and self.device != "cpu" else None)

    def detect(self, frame: Frame) -> Detections:
        """Person boxes; with a pose model (e.g. yolo11s-pose.pt) also each person's 17 COCO
        keypoints, which the tracker hands to the track the detection updates (bfmu head_pose,
        mode "tracks", reads them: no second model)."""
        res = self.model.predict(frame.image, **self.kw)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty()
        kp = getattr(res, "keypoints", None)
        if kp is not None and kp.xy is not None and len(kp.xy) == len(boxes):
            return Detections(boxes.xyxy.cpu().numpy().astype(float), boxes.conf.cpu().numpy().astype(float),
                              kp.xy.cpu().numpy().astype(float), kp.conf.cpu().numpy().astype(float))
        return Detections(boxes.xyxy.cpu().numpy().astype(float), boxes.conf.cpu().numpy().astype(float))

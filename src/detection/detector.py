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
        boxes = self.model.predict(frame.image, **self.kw)[0].boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty()
        return Detections(boxes.xyxy.cpu().numpy().astype(float), boxes.conf.cpu().numpy().astype(float))

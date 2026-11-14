"""Minimal YOLOv7 inference for the SCB behaviour checkpoints (design §5c).

The SCB authors publish YOLOv7 checkpoints, which pickle YOLOv7's own module classes, so the
official repo (lib/yolov7, pinned commit) must be importable. Only its model definitions are
used; letterboxing and NMS are done here with OpenCV/torchvision.

Safe loading: checkpoints are third-party pickles. They are scanned for the globals they
reference (without unpickling), refused if any global falls outside ALLOWED, and then loaded
with torch's weights_only unpickler restricted to exactly those globals, so a tampered file
cannot execute code.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Must precede any torchvision import: in the fyp-local env (torchvision 0.24.0+cu126, Pillow 13.0)
# importing torchvision first makes Pillow's _imaging DLL fail to load ("The operating system
# cannot run %1"); importing Pillow first works. YOLOv7's models/common.py imports torchvision.
import PIL.Image  # noqa: F401

import cv2
import numpy as np

YOLOV7_REPO = Path(__file__).resolve().parents[2] / "lib" / "yolov7"
ALLOWED = ("models.common.", "models.yolo.", "models.experimental.", "torch.nn.modules.",
           "numpy.core.multiarray._reconstruct", "numpy.dtype", "numpy.ndarray",
           "numpy.dtypes.", "collections.OrderedDict")


def _resolve(name: str):
    module, _, attr = name.rpartition(".")
    if module.startswith("numpy.core"):                  # numpy 2 moved numpy.core -> numpy._core
        module = module.replace("numpy.core", "numpy._core", 1)
    return getattr(importlib.import_module(module), attr)


def load_yolov7(weights: str | Path, repo: str | Path = YOLOV7_REPO):
    import torch
    repo = str(Path(repo).resolve())
    if not Path(repo, "models", "yolo.py").is_file():
        raise FileNotFoundError(f"YOLOv7 repo not found at {repo} (git clone https://github.com/WongKinYiu/yolov7)")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    names = torch.serialization.get_unsafe_globals_in_checkpoint(str(weights))
    bad = [n for n in names if not n.startswith(ALLOWED)]
    if bad:
        raise RuntimeError(f"refusing to load {weights}: unexpected globals {bad}")
    # numpy 2 rebuilds a pickled numpy.dtype as a numpy.dtypes.<X>DType instance; those classes
    # never appear by name in the file, so allow the numeric ones explicitly (plain data types).
    dtype_classes = [type(np.dtype(t)) for t in (np.float64, np.float32, np.float16, np.int64,
                                                 np.int32, np.int16, np.int8, np.uint8, np.bool_)]
    with torch.serialization.safe_globals([(_resolve(n), n) for n in names] + dtype_classes):
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=True)
    model = (ckpt.get("ema") or ckpt["model"]).float().eval()
    return model


class Yolov7Detector:
    def __init__(self, weights: str, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640,
                 device: str | None = None, half: bool = True):
        import torch
        self.torch = torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = half and self.device != "cpu"
        model = load_yolov7(weights).to(self.device)
        self.model = model.half() if self.half else model
        self.names = [str(n) for n in model.names]
        self.stride = int(model.stride.max())
        self.conf, self.iou, self.imgsz = conf, iou, imgsz

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        r = self.imgsz / max(h, w)
        nh, nw = round(h * r), round(w * r)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, 0, (-nh) % self.stride, 0, (-nw) % self.stride,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r

    def _forward(self, image_bgr: np.ndarray):
        """Raw head output (anchors, 5 + nc): cx, cy, w, h, objectness, per-class sigmoid; and scale."""
        torch = self.torch
        img, r = self._letterbox(image_bgr)
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))).to(self.device)
        x = (x.half() if self.half else x.float()) / 255
        with torch.no_grad():
            return self.model(x[None])[0][0].float(), r

    @staticmethod
    def _xyxy(pred):
        import torch
        return torch.cat([pred[:, :2] - pred[:, 2:4] / 2, pred[:, :2] + pred[:, 2:4] / 2], 1)

    def candidates(self, image_bgr: np.ndarray, obj_floor: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Every candidate box before thresholding and NMS (objectness >= obj_floor, which only drops
        empty anchors): (xyxy (N,4) in original pixels, objectness (N,), class probabilities (N, nc))."""
        pred, r = self._forward(image_bgr)
        pred = pred[pred[:, 4] >= obj_floor]
        return (self._xyxy(pred) / r).cpu().numpy(), pred[:, 4].cpu().numpy(), pred[:, 5:].cpu().numpy()

    def detect(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Thresholded, NMS'd detections: (xyxy (N,4) in original pixels, conf (N,), class index (N,))."""
        import torchvision
        pred, r = self._forward(image_bgr)
        conf, cls = (pred[:, 4:5] * pred[:, 5:]).max(1)
        keep = conf >= self.conf
        pred, conf, cls = pred[keep], conf[keep], cls[keep]
        xyxy = self._xyxy(pred)
        k = torchvision.ops.batched_nms(xyxy, conf, cls, self.iou)
        return (xyxy[k] / r).cpu().numpy(), conf[k].cpu().numpy(), cls[k].cpu().numpy().astype(int)

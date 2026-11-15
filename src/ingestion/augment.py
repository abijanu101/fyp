"""Training-time data augmentation of whole sessions (config `augmentation`).

An augmented copy of a session is the same recording seen through one fixed transformation: the
colour shift, the horizontal flip and the masked rectangles are drawn **once per (session, copy)**
and applied to every frame, so tracking and the window features see a consistent, plausible video
(another camera white balance, a mirrored room, a fixed occluder), not per-frame flicker. Only the
sensor noise is redrawn per frame, at a level fixed for the session.

Copies are never written to disk: `AugmentedZipFrameSource` augments frames as they are read, and
the parameters are seeded per (session, copy) and the noise per frame, so a copy is the same video
every time it is read. `run_pipeline extract` with `augmentation.copies > 0` extracts features from
the copies, `train_models` adds the copies of the training groups to each fold, and
`tune_tracker --augment` tracks on a copy. Held-out groups are always scored on original footage,
and inference never augments.

Order: hflip -> colour -> noise -> mask. Labels: colour, noise and masks leave the manual student
boxes unchanged; hflip mirrors them (`hflip_boxes`), which `train_models` applies before linking
tracks to students.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from ..core.config import config_hash
from ..core.types import Frame
from .source import ZipFrameSource

NOISE_KINDS = ("gaussian", "salt & pepper", "speckle")


# ----------------------------------------------------------------------------- transforms

def hflip_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
    out = np.asarray(boxes, float).copy()
    out[:, 0], out[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
    return out


def _colour_reference(img, brightness=0.0, contrast=1.0, saturation=1.0, hue_deg=0.0, gamma=1.0):
    """Per-pixel float version: the definition `colour` reproduces exactly (kept for the test)."""
    x = np.clip(img.astype(np.float32) * contrast + brightness, 0, 255)
    if saturation != 1.0 or hue_deg != 0.0:
        hsv = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + hue_deg / 2) % 180          # OpenCV hue is 0..179 (2 deg per unit)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if gamma != 1.0:
        x = 255.0 * (x / 255.0) ** (1.0 / gamma)
    return np.clip(x, 0, 255).astype(np.uint8)


def _lut(f) -> np.ndarray:
    """A 256-entry uint8 table of f over the values 0..255, computed in float32 like the reference."""
    return np.clip(f(np.arange(256, dtype=np.float32)), 0, 255).astype(np.uint8)


def colour(img, brightness=0.0, contrast=1.0, saturation=1.0, hue_deg=0.0, gamma=1.0):
    """Brightness/contrast, then hue/saturation (HSV), then gamma. Every step maps 0-255 values to
    0-255 values, so each is a lookup table (cv2.LUT): about 20x faster than per-pixel float maths
    and bit-identical to `_colour_reference` (tests/test_augment.py)."""
    if saturation == 1.0 and hue_deg == 0.0:                  # gamma then acts on unrounded values
        return _colour_reference(img, brightness, contrast, saturation, hue_deg, gamma)
    x = cv2.LUT(img, _lut(lambda v: v * contrast + brightness))
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    h, sat, v = cv2.split(hsv)
    h = cv2.LUT(h, _lut(lambda u: (u + hue_deg / 2) % 180))
    sat = cv2.LUT(sat, _lut(lambda u: u * saturation))
    x = cv2.cvtColor(cv2.merge([h, sat, v]), cv2.COLOR_HSV2BGR)
    if gamma != 1.0:
        x = cv2.LUT(x, _lut(lambda u: 255.0 * (u / 255.0) ** (1.0 / gamma)))
    return x


def noise(img, kind: str, strength: float, rng: np.random.Generator):
    """strength 0..1 (0.1 is already clearly visible)."""
    if strength <= 0:
        return img
    x = img.astype(np.float32)
    z = lambda: rng.standard_normal(x.shape, dtype=np.float32)       # float32: half the cost of float64
    if kind == "gaussian":
        n = z()
        n *= np.float32(50 * strength)                                 # in place: same values, no temporaries
        x += n
    elif kind == "salt & pepper":
        r = rng.random(x.shape[:2], dtype=np.float32)
        x[r < 0.075 * strength] = 0
        x[r > 1 - 0.075 * strength] = 255
    elif kind == "speckle":
        n = z()
        n *= x
        n *= np.float32(0.6 * strength)
        x += n
    else:
        raise ValueError(f"noise kind {kind!r} not in {NOISE_KINDS}")
    np.clip(x, 0, 255, out=x)
    return x.astype(np.uint8)


def apply_masks(img, rects, kind: str, rng: np.random.Generator):
    """rects: [(x0, y0, w, h)] as fractions of the frame; each is filled grey or with noise."""
    h, w = img.shape[:2]
    if kind == "none" or not rects:
        return img
    out = img.copy()
    for fx, fy, fw, fh in rects:
        sl = (slice(int(fy * h), int((fy + fh) * h)), slice(int(fx * w), int((fx + fw) * w)))
        out[sl] = rng.integers(0, 256, out[sl].shape, dtype=np.uint8) if kind.startswith("random") else 127
    return out


# ----------------------------------------------------------------------------- per-session parameters

AUG_VERSION = 2   # bump when the transforms themselves change, so old copies are never mixed with new


def aug_tag(aug_cfg: dict, copy: int) -> str:
    """File-name tag of one augmented copy, tied to the augmentation settings and code version (not
    to `copies`), so changing either never reuses stale copies and raising `copies` keeps the
    existing ones."""
    settings = {k: v for k, v in aug_cfg.items() if k != "copies"}
    return f"aug{copy}-{config_hash({**settings, 'version': AUG_VERSION})[:8]}"


def sample_augmentation(aug_cfg: dict, session: str, copy: int) -> dict:
    """One fixed set of transformation parameters for (session, copy); deterministic."""
    rng = np.random.default_rng([aug_cfg.get("seed", 0), copy, zlib.crc32(session.encode())])
    u = lambda r: float(rng.uniform(-r, r))
    m = aug_cfg.get("mask", {})
    rects = []
    if m.get("kind", "none") != "none" and rng.random() < m.get("p", 0.0):
        lo, hi = m.get("count", [1, 1])
        for _ in range(int(rng.integers(lo, hi + 1))):
            s = float(rng.uniform(*m.get("size", [0.05, 0.1])))
            rects.append((float(rng.uniform(0, 1 - s)), float(rng.uniform(0, 1 - s)), s, s))
    n = aug_cfg.get("noise", {})
    return {
        "copy": copy,
        "hflip": bool(rng.random() < aug_cfg.get("hflip", 0.0)),
        "brightness": u(aug_cfg.get("brightness", 0.0)),
        "contrast": 1.0 + u(aug_cfg.get("contrast", 0.0)),
        "saturation": 1.0 + u(aug_cfg.get("saturation", 0.0)),
        "hue_deg": u(aug_cfg.get("hue_deg", 0.0)),
        "gamma": float(np.exp(u(np.log1p(aug_cfg.get("gamma", 0.0))))),
        "noise_kind": str(rng.choice(n.get("kinds", ["gaussian"]))),
        "noise": float(rng.uniform(0, n.get("max", 0.0))),
        "mask_kind": m.get("kind", "none") if rects else "none",
        "mask_rects": rects,
        "noise_seed": int(rng.integers(2 ** 31)),
    }


def augment_frame(img: np.ndarray, p: dict, frame_index: int) -> np.ndarray:
    """Apply one session's parameters `p` to one frame. Noise is redrawn per frame (seeded by the
    frame index, so a frame always gets the same noise); everything else is fixed per session."""
    rng = np.random.default_rng([p["noise_seed"], frame_index])
    if p["hflip"]:
        img = img[:, ::-1]
    img = colour(img, p["brightness"], p["contrast"], p["saturation"], p["hue_deg"], p["gamma"])
    img = noise(img, p["noise_kind"], p["noise"], rng)
    img = apply_masks(img, p["mask_rects"], p["mask_kind"], np.random.default_rng(p["noise_seed"]))
    return np.ascontiguousarray(img)


# ----------------------------------------------------------------------------- the frame source

class AugmentedZipFrameSource(ZipFrameSource, Dataset):
    """A camera ZIP seen through one fixed augmentation (`params` from `sample_augmentation`).

    Behaves exactly like ZipFrameSource (same timestamps, iteration, `read`), and is also a
    torch Dataset: `len(src)` frames, `src[i]` the augmented Frame (None if unreadable)."""

    def __init__(self, path: str | Path, params: dict, stride: int = 1, max_frames: int | None = None):
        super().__init__(path, stride=stride, max_frames=max_frames)
        self.params = params

    def __len__(self) -> int:
        return len(self._t)

    def __getitem__(self, i: int) -> Frame | None:
        return self.read(i)

    def _decode(self, zf, i):
        frame = super()._decode(zf, i)
        if frame is None:
            return None
        return Frame(frame.index, frame.t, augment_frame(frame.image, self.params, frame.index))

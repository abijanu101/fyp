"""Training-time data augmentation of whole sessions (config `augmentation`).

An augmented copy of a session is the same recording seen through one fixed transformation: the
colour shift, the horizontal flip and the masked rectangles are drawn **once per (session, copy)**
and applied to every frame, so tracking and the window features see a consistent, plausible video
(another camera white balance, a mirrored room, a fixed occluder), not per-frame flicker. Only the
sensor noise is redrawn per frame, at a level fixed for the session.

Copies are materialised on disk by `scripts/augment_dataset.py` as complete DIPSER-shaped datasets
(`augmented_root`: camera ZIPs of augmented frames, manual boxes mirrored where flipped, labels
linked), so any tool can run on one by pointing `dataset.root` at it (the tracker tuning too).
`run_pipeline extract` with `augmentation.copies > 0` extracts features from them, and
`train_models` adds the copies of the training groups to each fold. Held-out groups are always
scored on original footage, and inference never augments.

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
MASK_KINDS = ("none", "cutout (grey)", "random erasing (noise)")


# ----------------------------------------------------------------------------- transforms

def hflip_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
    out = np.asarray(boxes, float).copy()
    out[:, 0], out[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
    return out


def colour(img, brightness=0.0, contrast=1.0, saturation=1.0, hue_deg=0.0, gamma=1.0):
    x = np.clip(img.astype(np.float32) * contrast + brightness, 0, 255)
    if saturation != 1.0 or hue_deg != 0.0:
        hsv = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + hue_deg / 2) % 180          # OpenCV hue is 0..179 (2 deg per unit)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if gamma != 1.0:
        x = 255.0 * (x / 255.0) ** (1.0 / gamma)
    return np.clip(x, 0, 255).astype(np.uint8)


def noise(img, kind: str, strength: float, rng: np.random.Generator):
    """strength 0..1 (0.1 is already clearly visible)."""
    if strength <= 0:
        return img
    x = img.astype(np.float32)
    z = lambda: rng.standard_normal(x.shape, dtype=np.float32)       # float32: half the cost of float64
    if kind == "gaussian":
        x += z() * np.float32(50 * strength)
    elif kind == "salt & pepper":
        r = rng.random(x.shape[:2], dtype=np.float32)
        x[r < 0.075 * strength] = 0
        x[r > 1 - 0.075 * strength] = 255
    elif kind == "speckle":
        x += x * z() * np.float32(0.6 * strength)
    else:
        raise ValueError(f"noise kind {kind!r} not in {NOISE_KINDS}")
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_masks(img, rects, kind: str, rng: np.random.Generator):
    """rects: [(x0, y0, w, h)] as fractions of the frame. Returns (image, masked-pixel bool map)."""
    h, w = img.shape[:2]
    masked = np.zeros((h, w), bool)
    if kind == "none" or not rects:
        return img, masked
    out = img.copy()
    for fx, fy, fw, fh in rects:
        sl = (slice(int(fy * h), int((fy + fh) * h)), slice(int(fx * w), int((fx + fw) * w)))
        out[sl] = rng.integers(0, 256, out[sl].shape, dtype=np.uint8) if kind.startswith("random") else 127
        masked[sl] = True
    return out, masked


# ----------------------------------------------------------------------------- per-session parameters

AUG_VERSION = 2   # bump when the transforms themselves change, so old copies are never mixed with new


def aug_tag(aug_cfg: dict, copy: int) -> str:
    """File-name tag of one augmented copy, tied to the augmentation settings and code version (not
    to `copies`), so changing either never reuses stale copies and raising `copies` keeps the
    existing ones."""
    settings = {k: v for k, v in aug_cfg.items() if k not in ("copies", "root")}
    return f"aug{copy}-{config_hash({**settings, 'version': AUG_VERSION})[:8]}"


def sample_params(aug_cfg: dict, session: str, copy: int) -> dict:
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
    img, _ = apply_masks(img, p["mask_rects"], p["mask_kind"], np.random.default_rng(p["noise_seed"]))
    return np.ascontiguousarray(img)


# ----------------------------------------------------------------------------- the frame source

class AugmentedZipFrameSource(ZipFrameSource, Dataset):
    """A camera ZIP seen through one fixed augmentation (`params` from `sample_params`).

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


# ----------------------------------------------------------------------------- materialised copies

def augmented_root(aug_cfg: dict, copy: int) -> Path:
    """Where augmented copy `copy` is written as a complete DIPSER-shaped dataset."""
    return Path(aug_cfg.get("root", "data/DIPSER_aug")) / aug_tag(aug_cfg, copy)


def write_augmented_zip(src_zip: Path, dst_zip: Path, params: dict, quality: int = 95) -> int:
    """Write one camera ZIP seen through `params`: images/<same timestamp name>.jpg, JPEG at
    `quality` (noisy frames do not compress as PNG: ~2.2 MB vs ~0.5 MB per frame). Atomic; returns
    the frame width (for mirroring the manual boxes)."""
    import zipfile
    src = AugmentedZipFrameSource(src_zip, params)
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_zip.with_suffix(".tmp.zip")
    width = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:      # JPEGs are already compressed
        for frame in src:
            ok, buf = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            name = Path(src._members[frame.index]).stem + ".jpg"
            zf.writestr(f"images/{name}", buf.tobytes())
            width = frame.image.shape[1]
    tmp.replace(dst_zip)
    return width

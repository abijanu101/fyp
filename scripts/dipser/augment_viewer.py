"""Augmentation viewer: preview colour shifts, noise, horizontal flip and masking on a DIPSER frame,
with the manual student boxes (abi_labels) transformed alongside.

    python -m scripts.dipser.augment_viewer group_01 experiment_03 context_cam_02
    python -m scripts.dipser.augment_viewer group_02 experiment_05 context_cam_01 --start 120

Colour and noise are the pipeline's own transforms (src/ingestion/augment.py, used for augmented
training copies, config `augmentation`); the viewer adds student occlusion, which the pipeline does
not use. Order: flip -> colour -> noise -> mask. The hue slider spans +/-10 degrees; the
pipeline's default range is +/-3.

How each transform treats the labels (the student boxes)
  colour, noise   boxes unchanged
  hflip           boxes mirrored: x1' = W - x2, x2' = W - x1
  masking         boxes unchanged, but each box shows the share of it still visible

Window "controls": sliders (brightness, contrast, saturation, hue, gamma, noise strength, mask
count and size, seed, frame position).
Side panel: radio groups (noise type, mask type, view); click an option, or use the keys.

Keys      f  flip on/off      n  next noise type      m  next mask type      v  next view
          r  new random draw (seed + 1)               , .  -/+1 s     [ ]  -/+30 s
          b  boxes on/off     s  save PNG to tmp/augment/                q / Esc  quit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from scripts.run_pipeline import parse_sets
from src.core.config import load_config
from src.core.types import SessionKey
from src.datasets import DipserIndex
from src.ingestion import ZipFrameSource
from src.ingestion.augment import colour, noise      # the pipeline's own transforms (training copies)

FONT = cv2.FONT_HERSHEY_SIMPLEX
WIN, CTRL = "Augmentation viewer", "controls"
PANEL_W, PANEL_MIN_H, HEADER_H = 250, 560, 44
NOISE_TYPES = ["gaussian", "salt & pepper", "speckle"]
MASK_TYPES = ["none", "cutout (grey)", "random erasing (noise)", "student occlusion"]
VIEWS = ["augmented", "original", "side by side"]

# slider name -> (max, default); values are mapped to parameters in `params()`
SLIDERS = {
    "brightness (-100..100)": (200, 100),
    "contrast %": (300, 100),
    "saturation %": (300, 100),
    "hue (-10..10 deg)": (20, 10),
    "gamma x100": (300, 100),
    "noise strength": (100, 0),
    "mask count": (20, 0),
    "mask size %": (60, 15),
    "seed": (99, 0),
}


# ----------------------------------------------------------------------------- transforms
# Each takes and returns BGR uint8 images and (N, 4) xyxy boxes.

def hflip(img, boxes):
    w = img.shape[1]
    out = boxes.copy()
    out[:, 0], out[:, 2] = w - boxes[:, 2], w - boxes[:, 0]
    return img[:, ::-1].copy(), out


def mask(img, boxes, kind, count, size, rng):
    """size 0..1: side of a rectangle as a share of the frame (cutout, erasing) or of the chosen
    student's box (student occlusion). Returns (image, masked-pixel bool map)."""
    h, w = img.shape[:2]
    masked = np.zeros((h, w), bool)
    if kind == "none" or count <= 0 or size <= 0:
        return img, masked
    out = img.copy()
    for _ in range(count):
        if kind == "student occlusion":
            if not len(boxes):
                break
            x1, y1, x2, y2 = boxes[rng.integers(len(boxes))]
            bw, bh = max(1, int((x2 - x1) * size)), max(1, int((y2 - y1) * size))
            x0 = int(rng.uniform(x1, max(x1, x2 - bw)))
            y0 = int(rng.uniform(y1, max(y1, y2 - bh)))
        else:
            bw, bh = max(1, int(w * size)), max(1, int(h * size))
            x0, y0 = int(rng.integers(0, max(1, w - bw))), int(rng.integers(0, max(1, h - bh)))
        sl = (slice(max(0, y0), min(h, y0 + bh)), slice(max(0, x0), min(w, x0 + bw)))
        if kind == "random erasing (noise)":
            out[sl] = rng.integers(0, 256, out[sl].shape, dtype=np.uint8)
        else:
            out[sl] = 127
        masked[sl] = True
    return out, masked


def visible_share(boxes, masked):
    out = []
    h, w = masked.shape
    for x1, y1, x2, y2 in boxes.astype(int):
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        area = max(1, (x2 - x1) * (y2 - y1))
        out.append(1.0 - masked[y1:y2, x1:x2].sum() / area)
    return np.array(out)


# ----------------------------------------------------------------------------- viewer
class AugmentViewer:
    def __init__(self, args, cfg):
        index = DipserIndex.from_config(cfg)
        self.key = SessionKey(args.group, args.experiment, args.camera)
        self.source = ZipFrameSource(index.frames_zip(self.key))
        self.t = self.source.timestamps
        if not len(self.t):
            raise SystemExit("no frames in archive")
        ann = index.annotations(self.key)
        self.sids = sorted(ann)
        self.boxes = np.stack([ann[s] for s in self.sids]) if self.sids else np.zeros((0, 4))
        if not self.sids:
            print(f"note: no manual boxes for {self.key}")
        self.idx = min(int(np.searchsorted(self.t, self.t[0] + args.start)), len(self.t) - 1)
        self.scale = args.scale
        self.flip, self.show_boxes = False, True
        self.noise_i, self.mask_i, self.view_i = 0, 0, 0
        self.values = {k: d for k, (_, d) in SLIDERS.items()}
        self.radio_hits: list[tuple[tuple[int, int, int, int], str, int]] = []
        self._img = (None, None)

    # -- state
    def frame(self) -> np.ndarray:
        if self._img[0] != self.idx:
            f = self.source.read(self.idx)
            self._img = (self.idx, f.image if f is not None else np.zeros((720, 1280, 3), np.uint8))
        return self._img[1]

    def params(self) -> dict:
        v = self.values
        return {"brightness": v["brightness (-100..100)"] - 100, "contrast": v["contrast %"] / 100,
                "saturation": v["saturation %"] / 100, "hue_deg": v["hue (-10..10 deg)"] - 10,
                "gamma": max(v["gamma x100"], 10) / 100, "noise": v["noise strength"] / 100,
                "mask_count": v["mask count"], "mask_size": v["mask size %"] / 100, "seed": v["seed"]}

    def read_trackbars(self) -> None:
        for name in SLIDERS:
            try:
                pos = cv2.getTrackbarPos(name, CTRL)
            except cv2.error:             # no window (snapshot mode)
                return
            if pos >= 0:
                self.values[name] = pos

    # -- pipeline
    def augment(self):
        p = self.params()
        rng = np.random.default_rng(p["seed"])
        img, boxes, applied = self.frame(), self.boxes.astype(float), []
        if self.flip:
            img, boxes = hflip(img, boxes)
            applied.append("hflip")
        col = {k: p[k] for k in ("brightness", "contrast", "saturation", "hue_deg", "gamma")}
        if col != {"brightness": 0, "contrast": 1.0, "saturation": 1.0, "hue_deg": 0, "gamma": 1.0}:
            img = colour(img, **col)
            applied.append(f"colour (b {p['brightness']:+d}, c {p['contrast']:.2f}, s {p['saturation']:.2f}, "
                           f"h {p['hue_deg']:+d}, g {p['gamma']:.2f})")
        kind = NOISE_TYPES[self.noise_i]
        if p["noise"] > 0:
            img = noise(img, kind, p["noise"], rng)
            applied.append(f"{kind} noise {p['noise']:.2f}")
        mk = MASK_TYPES[self.mask_i]
        img, masked = mask(img, boxes, mk, p["mask_count"], p["mask_size"], rng)
        if mk != "none" and p["mask_count"] > 0:
            applied.append(f"{mk} x{p['mask_count']} at {p['mask_size']:.0%}")
        return img, boxes, masked, applied

    # -- rendering
    def draw_boxes(self, img, boxes, vis=None):
        s = self.scale
        for k, (b, sid) in enumerate(zip(boxes, self.sids)):
            v = 1.0 if vis is None else vis[k]
            col = (0, 220, 0) if v > 0.75 else (0, 200, 255) if v > 0.4 else (0, 0, 255)
            p1, p2 = (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s))
            cv2.rectangle(img, p1, p2, col, 2)
            text = sid.replace("subject_", "S") + ("" if vis is None else f" {v:.0%}")
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.42, 1)
            cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 6, p1[1]), col, -1)
            cv2.putText(img, text, (p1[0] + 3, p1[1] - 4), FONT, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def panel(self, height: int) -> np.ndarray:
        img = np.full((height, PANEL_W, 3), 32, np.uint8)
        self.radio_hits = []
        y = 22

        def group(title, options, current, attr):
            nonlocal y
            cv2.putText(img, title, (10, y), FONT, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
            y += 8
            for i, opt in enumerate(options):
                y += 22
                cv2.circle(img, (20, y - 5), 7, (200, 200, 200), 1, cv2.LINE_AA)
                if i == current:
                    cv2.circle(img, (20, y - 5), 4, (90, 220, 255), -1, cv2.LINE_AA)
                cv2.putText(img, opt, (36, y), FONT, 0.43, (220, 220, 220), 1, cv2.LINE_AA)
                self.radio_hits.append(((8, y - 16, PANEL_W - 8, y + 4), attr, i))
            y += 24

        group("noise type  [n]", NOISE_TYPES, self.noise_i, "noise_i")
        group("mask type  [m]", MASK_TYPES, self.mask_i, "mask_i")
        group("view  [v]", VIEWS, self.view_i, "view_i")
        group("hflip  [f]", ["off", "on"], int(self.flip), "flip")
        cv2.putText(img, "labels:", (10, y), FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        rows = ["colour, noise: unchanged", "hflip: boxes mirrored" if self.flip else "hflip: (off)",
                "mask: % = share of box visible", "green >75%  orange >40%", "red <=40%"]
        for r in rows:
            y += 20
            cv2.putText(img, r, (14, y), FONT, 0.4, (190, 190, 190), 1, cv2.LINE_AA)
        return img

    def render(self) -> np.ndarray:
        self.read_trackbars()
        aug, boxes, masked, applied = self.augment()
        size = (int(aug.shape[1] * self.scale), int(aug.shape[0] * self.scale))
        a = cv2.resize(aug, size, interpolation=cv2.INTER_AREA)
        o = cv2.resize(self.frame(), size, interpolation=cv2.INTER_AREA)
        if self.show_boxes:
            a = self.draw_boxes(a, boxes, visible_share(boxes, masked) if masked.any() else None)
            o = self.draw_boxes(o, self.boxes)
        view = VIEWS[self.view_i]
        body = a if view == "augmented" else o if view == "original" else np.hstack([o, a])
        header = np.full((HEADER_H, body.shape[1], 3), 30, np.uint8)
        t = float(self.t[self.idx] - self.t[0])
        cv2.putText(header, f"{self.key}  +{t:.1f}s  frame {self.idx + 1}/{len(self.t)}  "
                            f"view: {view}  seed {self.params()['seed']}",
                    (8, 17), FONT, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(header, "applied: " + (" -> ".join(applied) if applied else "nothing (identity)"),
                    (8, 36), FONT, 0.42, (120, 230, 120), 1, cv2.LINE_AA)
        main = np.vstack([header, body])
        h = max(main.shape[0], PANEL_MIN_H)
        if main.shape[0] < h:                                 # wide, short views: keep the panel whole
            main = np.vstack([main, np.full((h - main.shape[0], main.shape[1], 3), 22, np.uint8)])
        return np.hstack([main, self.panel(h)])

    # -- interaction
    def make_controls(self) -> None:
        cv2.namedWindow(CTRL, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CTRL, 520, 40 * (len(SLIDERS) + 1))
        for name, (mx, default) in SLIDERS.items():
            cv2.createTrackbar(name, CTRL, default, mx, lambda _: None)
        cv2.createTrackbar("frame", CTRL, self.idx, len(self.t) - 1, lambda _: None)

    def on_mouse(self, event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x0 = getattr(self, "_main_w", 1 << 30)
        for (a, b, c, d), attr, i in self.radio_hits:
            if a <= x - x0 <= c and b <= y <= d:
                setattr(self, attr, bool(i) if attr == "flip" else i)

    def seek(self, seconds: float) -> None:
        self.idx = int(np.clip(np.searchsorted(self.t, self.t[self.idx] + seconds), 0, len(self.t) - 1))
        try:
            cv2.setTrackbarPos("frame", CTRL, self.idx)
        except cv2.error:
            pass

    def run(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.make_controls()
        first = True
        while True:
            pos = cv2.getTrackbarPos("frame", CTRL)
            if pos >= 0 and pos != self.idx:
                self.idx = pos
            img = self.render()
            self._main_w = img.shape[1] - PANEL_W
            cv2.imshow(WIN, img)
            if first:
                cv2.resizeWindow(WIN, img.shape[1], img.shape[0])
                first = False
            key = cv2.waitKeyEx(30)
            if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("f"):
                self.flip = not self.flip
            elif key == ord("n"):
                self.noise_i = (self.noise_i + 1) % len(NOISE_TYPES)
            elif key == ord("m"):
                self.mask_i = (self.mask_i + 1) % len(MASK_TYPES)
            elif key == ord("v"):
                self.view_i = (self.view_i + 1) % len(VIEWS)
            elif key == ord("b"):
                self.show_boxes = not self.show_boxes
            elif key == ord("r"):
                cv2.setTrackbarPos("seed", CTRL, (self.values["seed"] + 1) % (SLIDERS["seed"][0] + 1))
            elif key in (ord(","), ord(".")):
                self.seek(-1 if key == ord(",") else 1)
            elif key in (ord("["), ord("]")):
                self.seek(-30 if key == ord("[") else 30)
            elif key == ord("s"):
                out = Path("tmp/augment")
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"{self.key.group}_{self.key.experiment}_{self.key.camera}_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(str(path), img)
                print(f"saved {path}  ({' | '.join(self.augment()[3]) or 'identity'})")
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", nargs="?", default="group_01")
    ap.add_argument("experiment", nargs="?", default="experiment_03")
    ap.add_argument("camera", nargs="?", default="context_cam_02")
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--start", type=float, default=30.0, help="seconds into the session")
    ap.add_argument("--scale", type=float, default=0.6, help="display scale")
    ap.add_argument("--snapshot", help="render once to this PNG and exit (no window); use --preset")
    ap.add_argument("--preset", nargs="*", default=[],
                    help="snapshot settings: slider=value (slider names as shown) and flip, noise=<i>, mask=<i>, view=<i>")
    args = ap.parse_args()
    viewer = AugmentViewer(args, load_config(args.config, parse_sets(args.set)))
    if args.snapshot:
        for p in args.preset:
            k, _, v = p.partition("=")
            if k == "flip":
                viewer.flip = True
            elif k in ("noise", "mask", "view"):
                setattr(viewer, f"{k}_i", int(v))
            else:
                match = next(s for s in SLIDERS if s.startswith(k))
                viewer.values[match] = int(v)
        cv2.imwrite(args.snapshot, viewer.render())
        print(f"wrote {args.snapshot}")
        return
    viewer.run()


if __name__ == "__main__":
    main()

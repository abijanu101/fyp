"""SCB behaviour viewer: play a DIPSER camera with the SCB behaviour detections drawn live.

    python -m scripts.dipser.scb_viewer group_01 experiment_04 context_cam_03
    python -m scripts.dipser.scb_viewer group_02 experiment_03 context_cam_01 --start 120 --precompute

Uses the SCB models configured in the `scb` BFMU entry (configs/deliverable3.json). Each frame is
run once at a low confidence floor (--floor) and cached, so the per-class threshold sliders and
scrubbing back are instant. Use the sliders to choose per-class thresholds by eye.

What you see
  video     behaviour boxes (colour per class, with confidence) above each class's threshold;
            person-detector boxes in thin grey when toggled with `p`
  timeline  one strip per class: detections per frame above the current threshold (darker = more),
            grey = frame not analysed yet; white line = playhead

Mouse     click a strip to seek
Keys      space play/pause      left/right  -/+1 frame     , .  -/+5 s     [ ]  -/+30 s
          + -   playback speed  1-9  show/hide class        p  person boxes
          r     analyse every remaining frame (progress in the console)        q / Esc quit
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from scripts.run_pipeline import parse_sets
from src.bfmu.scb import normalize_class
from src.bfmu.yolov7 import Yolov7Detector
from src.core.config import load_config
from src.core.pipeline import Pipeline
from src.datasets import DipserIndex
from src.core.types import SessionKey
from src.ingestion import ZipFrameSource

COLOURS = [(0, 200, 255), (0, 220, 0), (255, 140, 0), (220, 0, 220), (0, 0, 255), (255, 255, 0)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_H, ROW_H, AXIS_H, LEFT = 30, 18, 16, 120
KEY_LEFT, KEY_RIGHT = (2424832, 65361), (2555904, 65363)   # Windows, X11
WIN = "SCB behaviour viewer"


class ScbViewer:
    def __init__(self, args, cfg: dict):
        index = DipserIndex.from_config(cfg)
        self.key = SessionKey(args.group, args.experiment, args.camera)
        self.source = ZipFrameSource(index.frames_zip(self.key))
        self.t = self.source.timestamps
        if not len(self.t):
            raise SystemExit("no frames in archive")
        spec = next(e for e in cfg["bfmu"] if e["name"] == "scb")
        self.models, self.classes, self.thresholds = [], [], []
        for m in spec["models"]:
            det = Yolov7Detector(m["weights"], conf=args.floor)
            names = [normalize_class(n) for n in det.names]
            self.models.append((det, len(self.classes)))            # (detector, offset into self.classes)
            self.classes += names
            self.thresholds += [m.get("class_conf", {}).get(n, m.get("conf", 0.25)) for n in names]
        self.visible = [True] * len(self.classes)
        self.person_pipe = Pipeline(cfg)
        self.show_persons = False

        self.det_cache: dict[int, list[tuple[np.ndarray, float, int]]] = {}   # frame -> [(xyxy, conf, class)]
        self.person_cache: dict[int, np.ndarray] = {}
        self.img_cache: tuple[int | None, np.ndarray | None] = (None, None)
        self._strip_key, self._strips = None, None

        self.scale = args.scale
        h, w = self.frame_image(0).shape[:2]
        self.vw, self.vh = int(w * self.scale), int(h * self.scale)
        self.strip_w = self.vw - LEFT
        self.idx = min(int(np.searchsorted(self.t, self.t[0] + args.start)), len(self.t) - 1)
        self.playing, self.speed = False, 1.0
        self._restart_clock()

    # -- data
    def frame_image(self, i: int) -> np.ndarray:
        if self.img_cache[0] != i:
            f = self.source.read(i)
            self.img_cache = (i, f.image if f is not None else np.zeros((720, 1280, 3), np.uint8))
        return self.img_cache[1]

    def detections(self, i: int) -> list[tuple[np.ndarray, float, int]]:
        if i not in self.det_cache:
            img, out = self.frame_image(i), []
            for det, offset in self.models:
                xyxy, conf, cls = det.detect(img)
                out += [(b, float(c), offset + int(k)) for b, c, k in zip(xyxy, conf, cls)]
            self.det_cache[i] = out
        return self.det_cache[i]

    def persons(self, i: int) -> np.ndarray:
        if i not in self.person_cache:
            f = self.source.read(i)
            self.person_cache[i] = self.person_pipe.detector.detect(f).xyxy if f is not None else np.zeros((0, 4))
        return self.person_cache[i]

    def analyse_all(self) -> None:
        todo = [i for i in range(len(self.t)) if i not in self.det_cache]
        start = time.monotonic()
        for n, i in enumerate(todo, 1):
            self.detections(i)
            if n % 50 == 0 or n == len(todo):
                print(f"analysed {n}/{len(todo)} frames ({(time.monotonic() - start) / n * 1000:.0f} ms/frame)")

    # -- rendering
    def render(self) -> np.ndarray:
        self.read_trackbars()
        img = cv2.resize(self.frame_image(self.idx), (self.vw, self.vh), interpolation=cv2.INTER_AREA)
        s = self.scale
        if self.show_persons:
            for b in self.persons(self.idx):
                cv2.rectangle(img, (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s)), (170, 170, 170), 1)
        counts = [0] * len(self.classes)
        for b, c, k in self.detections(self.idx):
            if c < self.thresholds[k] or not self.visible[k]:
                continue
            counts[k] += 1
            col = COLOURS[k % len(COLOURS)]
            p1, p2 = (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s))
            cv2.rectangle(img, p1, p2, col, 2)
            label = f"{self.classes[k]} {c:.2f}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
            cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 6, p1[1]), col, -1)
            cv2.putText(img, label, (p1[0] + 3, p1[1] - 4), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        header = np.full((HEADER_H, self.vw, 3), 30, np.uint8)
        t = float(self.t[self.idx])
        txt = (f"{self.key}  +{t - self.t[0]:.1f}s  {self.idx + 1}/{len(self.t)}  "
               f"{'PLAY' if self.playing else 'PAUSE'} x{self.speed:g}  analysed {len(self.det_cache)}  | "
               + "  ".join(f"{n}:{c}" for n, c in zip(self.classes, counts)))
        cv2.putText(header, txt, (8, 20), FONT, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        return np.vstack([header, img, self.timeline()])

    def strips(self) -> np.ndarray:
        """(n_classes, strip_w) mean detections/frame above threshold per pixel column; NaN = not analysed."""
        key = (len(self.det_cache), tuple(self.thresholds))
        if key != self._strip_key:
            col_of = (np.arange(len(self.t)) * self.strip_w // len(self.t)).astype(int)
            total = np.zeros((len(self.classes), self.strip_w))
            seen = np.zeros(self.strip_w)
            for i, dets in self.det_cache.items():
                seen[col_of[i]] += 1
                for _, c, k in dets:
                    if c >= self.thresholds[k]:
                        total[k, col_of[i]] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                self._strips = np.where(seen > 0, total / seen, np.nan)
            self._strip_key = key
        return self._strips

    def timeline(self) -> np.ndarray:
        n = len(self.classes)
        img = np.full((AXIS_H + n * ROW_H + 6, self.vw, 3), 22, np.uint8)
        dur = self.t[-1] - self.t[0]
        for sec in range(0, int(dur) + 1, 30):
            x = LEFT + int(sec / max(dur, 1e-9) * self.strip_w)
            cv2.line(img, (x, AXIS_H - 4), (x, img.shape[0]), (55, 55, 55), 1)
            if sec % 60 == 0:
                cv2.putText(img, f"{sec // 60}:00", (x + 2, AXIS_H - 4), FONT, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
        strips = self.strips()
        peak = max(np.nanmax(strips) if np.isfinite(strips).any() else 1.0, 1.0)
        for k, name in enumerate(self.classes):
            y = AXIS_H + k * ROW_H
            col = np.array(COLOURS[k % len(COLOURS)], float)
            label = f"{k + 1} {name}" + ("" if self.visible[k] else " (hidden)")
            cv2.putText(img, label, (6, y + ROW_H - 5), FONT, 0.4,
                        (230, 230, 230) if self.visible[k] else (120, 120, 120), 1, cv2.LINE_AA)
            v = strips[k]
            row = np.where(np.isfinite(v)[:, None], (np.nan_to_num(v) / peak)[:, None] * col, 70)
            img[y + 1:y + ROW_H - 1, LEFT:LEFT + self.strip_w] = row.astype(np.uint8)[None]
        x = LEFT + int((self.t[self.idx] - self.t[0]) / max(dur, 1e-9) * self.strip_w)
        cv2.line(img, (x, AXIS_H - 4), (x, img.shape[0]), (255, 255, 255), 1)
        self.timeline_y0 = HEADER_H + self.vh
        return img

    # -- interaction
    def make_trackbars(self) -> None:
        for k, name in enumerate(self.classes):
            cv2.createTrackbar(f"{k + 1} {name} %", WIN, int(round(self.thresholds[k] * 100)), 100, lambda _: None)

    def read_trackbars(self) -> None:
        for k, name in enumerate(self.classes):
            try:
                pos = cv2.getTrackbarPos(f"{k + 1} {name} %", WIN)
            except cv2.error:              # no window yet (e.g. rendering headless)
                return
            if pos >= 0:
                self.thresholds[k] = pos / 100

    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and y >= getattr(self, "timeline_y0", 1 << 30) and x >= LEFT:
            frac = (x - LEFT) / self.strip_w
            self.seek(self.t[0] + frac * (self.t[-1] - self.t[0]))

    def seek(self, t: float) -> None:
        self.idx = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        self._restart_clock()

    def step(self, n: int) -> None:
        self.idx = int(np.clip(self.idx + n, 0, len(self.t) - 1))
        self._restart_clock()

    def _restart_clock(self) -> None:
        self.wall0, self.play_t0 = time.monotonic(), float(self.t[self.idx])

    def run(self) -> None:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.make_trackbars()
        first = True
        while True:
            if self.playing:
                target = self.play_t0 + (time.monotonic() - self.wall0) * self.speed
                self.idx = min(int(np.searchsorted(self.t, target, side="right")) - 1, len(self.t) - 1)
                if self.idx >= len(self.t) - 1:
                    self.playing = False
            img = self.render()
            cv2.imshow(WIN, img)
            if first:
                cv2.resizeWindow(WIN, img.shape[1], min(img.shape[0] + 40 * len(self.classes), 1000))
                first = False
            key = cv2.waitKeyEx(15)
            if key in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                self.playing = not self.playing
                self._restart_clock()
            elif key in KEY_LEFT:
                self.step(-1)
            elif key in KEY_RIGHT:
                self.step(1)
            elif key in (ord(","), ord(".")):
                self.seek(float(self.t[self.idx]) + (-5 if key == ord(",") else 5))
            elif key in (ord("["), ord("]")):
                self.seek(float(self.t[self.idx]) + (-30 if key == ord("[") else 30))
            elif key in (ord("+"), ord("=")):
                self.speed = min(self.speed * 2, 32)
                self._restart_clock()
            elif key in (ord("-"), ord("_")):
                self.speed = max(self.speed / 2, 0.25)
                self._restart_clock()
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(self.classes):
                self.visible[key - ord("1")] ^= True
            elif key == ord("p"):
                self.show_persons = not self.show_persons
            elif key == ord("r"):
                self.analyse_all()
                self._restart_clock()
        cv2.destroyAllWindows()
        print("final thresholds:", {n: round(v, 2) for n, v in zip(self.classes, self.thresholds)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", nargs="?", default="group_01")
    ap.add_argument("experiment", nargs="?", default="experiment_04")
    ap.add_argument("camera", nargs="?", default="context_cam_03")
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--start", type=float, default=0.0, help="start this many seconds into the session")
    ap.add_argument("--scale", type=float, default=0.75, help="video display scale")
    ap.add_argument("--floor", type=float, default=0.05, help="inference confidence floor (sliders go above it)")
    ap.add_argument("--precompute", action="store_true", help="analyse every frame before opening the window")
    args = ap.parse_args()
    viewer = ScbViewer(args, load_config(args.config, parse_sets(args.set)))
    if args.precompute:
        viewer.analyse_all()
    viewer.run()


if __name__ == "__main__":
    main()

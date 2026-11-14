"""DIPSER label viewer: play a context camera with every student's attention labels alongside.

    python scripts/dipser/label_viewer.py group_01 experiment_01 context_cam_01
    python scripts/dipser/label_viewer.py group_02 experiment_04 context_cam_03 --start 120 --exclude-self

Standalone (OpenCV + numpy only). Reads frames straight from the camera ZIP, labels from
subjects/subject_XX.zip (labels/*.json), and the static boxes from abi_labels.

What you see
  video     each annotated student's box, coloured by the labelers' median attention at the
            current time (or by a single labeler, see `l`); the selected student's box lists every labeler's current value
  timeline  one strip per student over the whole session (median attention), plus one strip per
            labeler for the selected student; white line = playhead
  colours   1 red, 2 orange, 3 yellow, 4 light green, 5 green, grey = no label yet

Mouse     click a box or a timeline row label to select a student; click a strip to seek
Keys      space play/pause      left/right  -/+1 frame     , .  -/+5 s     [ ]  -/+30 s
          + -   playback speed  n p  next/prev student      x  include/exclude self-report
          l     colour by: median -> each labeler in turn  v  labeler values on every box
          b     toggle boxes    q / Esc quit
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "DIPSER"
CLOCK = re.compile(r"(\d{1,2})[_:](\d{2})[_:](\d{2})[_:](\d+)")
SELF = "S"

# BGR colours for attention 1..5; fractional medians are interpolated.
PALETTE = np.array([(60, 60, 220), (40, 140, 245), (40, 215, 235), (100, 205, 120), (40, 150, 40)], float)
NO_LABEL = (90, 90, 90)
FONT = cv2.FONT_HERSHEY_SIMPLEX

HEADER_H, ROW_H, SUB_ROW_H, AXIS_H, LEGEND_H, LEFT = 30, 16, 12, 18, 22, 120

KEY_LEFT, KEY_RIGHT = (2424832, 65361), (2555904, 65363)   # Windows, X11


# ----------------------------------------------------------------------------- data

def parse_clock(text: str) -> float:
    h, m, s, frac = CLOCK.fullmatch(text).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(frac) / 10 ** len(frac)


def fmt_clock(t: float) -> str:
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


def short_name(labeler: str) -> str:
    return SELF if labeler == "self_labeling" else "L" + labeler.rsplit("_", 1)[-1].lstrip("0")


class Frames:
    def __init__(self, path: Path):
        self.zf = zipfile.ZipFile(path)
        members = [m for m in self.zf.namelist()
                   if PurePosixPath(m).parent.name == "images" and m.lower().endswith((".png", ".jpg", ".jpeg"))]
        members.sort(key=lambda m: parse_clock(PurePosixPath(m).stem))
        self.members = members
        self.t = np.array([parse_clock(PurePosixPath(m).stem) for m in members])

    def read(self, i: int) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(self.zf.read(self.members[i]), np.uint8), cv2.IMREAD_COLOR)
        return img if img is not None else np.zeros((720, 1280, 3), np.uint8)


def load_labels(exp_dir: Path, offset: float) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """{subject: {labeler: (change times, attention values)}}; attention is a step function."""
    out = {}
    for z in sorted((exp_dir / "subjects").glob("subject_*.zip")):
        labs = {}
        with zipfile.ZipFile(z) as zf:
            for m in sorted(zf.namelist()):
                p = PurePosixPath(m)
                if p.parent.name != "labels" or p.suffix != ".json":
                    continue
                pts = sorted((parse_clock(e["datetime"]) + offset, int(e["attention"]))
                             for e in json.loads(zf.read(m))
                             if str(e.get("attention", "")).strip() and 1 <= int(e["attention"]) <= 5)
                labs[short_name(p.stem)] = (np.array([a for a, _ in pts]), np.array([b for _, b in pts], float))
        out[z.stem] = dict(sorted(labs.items(), key=lambda kv: (kv[0] == SELF, kv[0])))
    return out


def value_at(track: tuple[np.ndarray, np.ndarray], t: np.ndarray | float) -> np.ndarray:
    times, vals = track
    i = np.searchsorted(times, t, side="right") - 1
    return np.where(i >= 0, vals[np.clip(i, 0, max(len(vals) - 1, 0))] if len(vals) else np.nan, np.nan)


def consensus(labs: dict, t: np.ndarray | float, exclude_self: bool) -> np.ndarray:
    vals = [value_at(tr, t) for name, tr in labs.items() if not (exclude_self and name == SELF)]
    if not vals:
        return np.full(np.shape(t), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns -> NaN
        return np.nanmedian(np.stack(vals), axis=0)


def colours(v: np.ndarray) -> np.ndarray:
    v = np.atleast_1d(np.asarray(v, float))
    x = np.clip(np.nan_to_num(v, nan=1.0) - 1, 0, 4)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, 4)
    c = PALETTE[lo] + (PALETTE[hi] - PALETTE[lo]) * (x - lo)[:, None]
    c[~np.isfinite(v)] = NO_LABEL
    return c.astype(np.uint8)


def colour(v: float) -> tuple[int, int, int]:
    return tuple(int(c) for c in colours(np.array([v]))[0])


# ----------------------------------------------------------------------------- viewer

class Viewer:
    def __init__(self, args):
        root = Path(args.root)
        exp_dir = root / "DIPSEER" / args.group / args.experiment
        self.title = f"{args.group} / {args.experiment} / {args.camera}"
        self.frames = Frames(exp_dir / f"{args.camera}.zip")
        if not len(self.frames.t):
            raise SystemExit("no frames in archive")
        self.labels = load_labels(exp_dir, args.offset)
        ann_path = root / "abi_labels" / args.group / f"{args.experiment}.json"
        ann = json.loads(ann_path.read_text()).get(args.camera, {}) if ann_path.is_file() else {}
        self.boxes = {s: (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]) for s, b in ann.items()}
        # Students visible in this camera first, then the rest of the session.
        self.subjects = sorted(self.labels, key=lambda s: (s not in self.boxes, s))
        # Colour source: None = median of labelers, otherwise one labeler's short name.
        names = {n for labs in self.labels.values() for n in labs}
        self.sources = [None] + sorted(names, key=lambda n: (n == SELF, n))
        self.source = None

        self.scale = args.scale
        h, w = self.frames.read(0).shape[:2]
        self.vw, self.vh = int(w * self.scale), int(h * self.scale)
        self.strip_w = self.vw - LEFT
        self.t0, self.t1 = float(self.frames.t[0]), float(self.frames.t[-1])

        self.idx = int(np.searchsorted(self.frames.t, self.t0 + args.start)) if args.start else 0
        self.idx = min(self.idx, len(self.frames.t) - 1)
        self.selected = self.subjects[0] if self.subjects else None
        self.exclude_self = args.exclude_self
        self.verbose = False
        self.show_boxes = True
        self.playing, self.speed = False, 1.0
        self._frame_cache = (None, None)
        self._strip_cache: dict = {}
        self.rows: list[tuple[int, int, str]] = []   # (y0, y1, subject) of timeline rows, for clicks
        self.timeline_y0 = 0

    def coloured_value(self, labs: dict, t):
        """Median of labelers, or the chosen single labeler (NaN if they did not rate this student)."""
        if self.source is None:
            return consensus(labs, t, self.exclude_self)
        return value_at(labs[self.source], t) if self.source in labs else np.full(np.shape(t), np.nan)

    # -- timeline strips (precomputed per subject/labeler/mode)
    def strip(self, subject: str, labeler: str | None, height: int) -> np.ndarray:
        key = (subject, labeler, height, self.exclude_self, self.source)
        if key not in self._strip_cache:
            t = self.t0 + (np.arange(self.strip_w) + 0.5) / self.strip_w * (self.t1 - self.t0)
            labs = self.labels[subject]
            v = self.coloured_value(labs, t) if labeler is None else value_at(labs[labeler], t)
            row = colours(v)[None].repeat(height - 2, axis=0)
            if labeler is not None:     # tick marks where this labeler actually changed the value
                for tc in labs[labeler][0]:
                    x = int((tc - self.t0) / (self.t1 - self.t0) * self.strip_w)
                    if 0 <= x < self.strip_w:
                        row[:3, x] = (255, 255, 255)
            self._strip_cache[key] = row
        return self._strip_cache[key]

    def t_to_x(self, t: float) -> int:
        return LEFT + int((t - self.t0) / max(self.t1 - self.t0, 1e-9) * self.strip_w)

    # -- rendering
    def frame(self) -> np.ndarray:
        if self._frame_cache[0] != self.idx:
            img = cv2.resize(self.frames.read(self.idx), (self.vw, self.vh), interpolation=cv2.INTER_AREA)
            self._frame_cache = (self.idx, img)
        return self._frame_cache[1].copy()

    def render(self) -> np.ndarray:
        t = float(self.frames.t[self.idx])
        video = self.frame()
        if self.show_boxes:
            self.draw_boxes(video, t)
        timeline = self.draw_timeline(t)
        header = np.full((HEADER_H, self.vw, 3), 30, np.uint8)
        mode = ("median, no self" if self.exclude_self else "median, with self") if self.source is None             else f"labeler {self.source} only"
        txt = (f"{self.title.replace(' / ', '/')}  {fmt_clock(t)} (+{t - self.t0:.1f}s)  "
               f"{self.idx + 1}/{len(self.frames.t)}  {'PLAY' if self.playing else 'PAUSE'} x{self.speed:g}  "
               f"colour: {mode}")
        cv2.putText(header, txt, (8, 20), FONT, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        self.timeline_y0 = HEADER_H + self.vh
        self.rows = [(y0 + self.timeline_y0, y1 + self.timeline_y0, s) for y0, y1, s in self.rows]
        return np.vstack([header, video, timeline])

    def draw_boxes(self, img: np.ndarray, t: float) -> None:
        s = self.scale
        for subj, (x1, y1, x2, y2) in self.boxes.items():
            labs = self.labels.get(subj, {})
            med = float(self.coloured_value(labs, t)) if labs else np.nan
            sel = subj == self.selected
            p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
            cv2.rectangle(img, p1, p2, colour(med), 4 if sel else 2)
            head = f"{subj.replace('subject_', 's')} {'-' if np.isnan(med) else f'{med:g}'}"
            lines = [head]
            if sel or self.verbose:
                lines.append(" ".join(f"{n}:{'-' if np.isnan(v) else int(v)}"
                                      for n, v in ((n, float(value_at(tr, t))) for n, tr in labs.items())))
            for k, line in enumerate(lines):
                (tw, th), _ = cv2.getTextSize(line, FONT, 0.45, 1)
                y = p1[1] + 4 + (k + 1) * (th + 6)
                cv2.rectangle(img, (p1[0] + 2, y - th - 4), (p1[0] + tw + 8, y + 3), (0, 0, 0), -1)
                cv2.putText(img, line, (p1[0] + 5, y), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_timeline(self, t: float) -> np.ndarray:
        n_sub = len(self.labels.get(self.selected, {})) if self.selected else 0
        height = AXIS_H + len(self.subjects) * ROW_H + n_sub * SUB_ROW_H + LEGEND_H + 6
        img = np.full((height, self.vw, 3), 22, np.uint8)
        # time axis (every 30 s, labelled every 60 s)
        for sec in range(0, int(self.t1 - self.t0) + 1, 30):
            x = self.t_to_x(self.t0 + sec)
            cv2.line(img, (x, AXIS_H - 5), (x, height - LEGEND_H), (55, 55, 55), 1)
            if sec % 60 == 0:
                cv2.putText(img, f"{sec // 60}:00", (x + 2, AXIS_H - 6), FONT, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        y, self.rows = AXIS_H, []
        for subj in self.subjects:
            in_view = subj in self.boxes
            sel = subj == self.selected
            label = subj.replace("subject_", "s") + ("" if in_view else " (off cam)")
            if sel:
                cv2.rectangle(img, (0, y), (LEFT - 4, y + ROW_H - 1), (90, 70, 40), -1)
            cv2.putText(img, label, (6, y + ROW_H - 4), FONT, 0.4,
                        (255, 255, 255) if in_view else (140, 140, 140), 1, cv2.LINE_AA)
            img[y + 1:y + ROW_H - 1, LEFT:LEFT + self.strip_w] = self.strip(subj, None, ROW_H)
            self.rows.append((y, y + ROW_H, subj))
            y += ROW_H
            if sel:
                for name in self.labels[subj]:
                    cv2.putText(img, f"   {name}", (6, y + SUB_ROW_H - 2), FONT, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
                    img[y + 1:y + SUB_ROW_H - 1, LEFT:LEFT + self.strip_w] = self.strip(subj, name, SUB_ROW_H)
                    self.rows.append((y, y + SUB_ROW_H, subj))
                    y += SUB_ROW_H
        x = self.t_to_x(t)
        cv2.line(img, (x, AXIS_H - 4), (x, y), (255, 255, 255), 1)
        # legend
        ly = height - LEGEND_H + 4
        lx = LEFT
        for v, name in [(1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5"), (np.nan, "no label")]:
            cv2.rectangle(img, (lx, ly), (lx + 12, ly + 12), colour(v), -1)
            cv2.putText(img, name, (lx + 16, ly + 11), FONT, 0.4, (210, 210, 210), 1, cv2.LINE_AA)
            lx += 40 if name != "no label" else 80
        cv2.putText(img, "white ticks = labeler changed value", (lx + 10, ly + 11), FONT, 0.4,
                    (160, 160, 160), 1, cv2.LINE_AA)
        return img

    # -- interaction
    def seek_time(self, t: float) -> None:
        self.idx = int(np.clip(np.searchsorted(self.frames.t, t), 0, len(self.frames.t) - 1))
        self._restart_clock()

    def step(self, frames: int) -> None:
        self.idx = int(np.clip(self.idx + frames, 0, len(self.frames.t) - 1))
        self._restart_clock()

    def _restart_clock(self) -> None:
        self.wall0, self.play_t0 = time.monotonic(), float(self.frames.t[self.idx])

    def on_mouse(self, event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if HEADER_H <= y < HEADER_H + self.vh:                     # click in video -> select box
            fx, fy = x / self.scale, (y - HEADER_H) / self.scale
            hits = [(abs((b[2] - b[0]) * (b[3] - b[1])), s) for s, b in self.boxes.items()
                    if b[0] <= fx <= b[2] and b[1] <= fy <= b[3]]
            if hits:
                self.selected = min(hits)[1]
            return
        for y0, y1, subj in self.rows:                            # click in timeline
            if y0 <= y < y1:
                if x < LEFT:
                    self.selected = subj
                else:
                    self.seek_time(self.t0 + (x - LEFT) / self.strip_w * (self.t1 - self.t0))
                return

    def cycle(self, d: int) -> None:
        if self.subjects:
            i = self.subjects.index(self.selected) if self.selected in self.subjects else 0
            self.selected = self.subjects[(i + d) % len(self.subjects)]

    def run(self) -> None:
        win = "DIPSER label viewer"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(win, self.on_mouse)
        self._restart_clock()
        first = True
        while True:
            if self.playing:
                target = self.play_t0 + (time.monotonic() - self.wall0) * self.speed
                self.idx = min(int(np.searchsorted(self.frames.t, target, side="right")) - 1, len(self.frames.t) - 1)
                if self.idx >= len(self.frames.t) - 1:
                    self.playing = False
            img = self.render()
            cv2.imshow(win, img)
            if first:
                cv2.resizeWindow(win, img.shape[1], min(img.shape[0], 1000))
                first = False
            key = cv2.waitKeyEx(15)
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                self.playing = not self.playing
                self._restart_clock()
            elif key in KEY_LEFT:
                self.step(-1)
            elif key in KEY_RIGHT:
                self.step(1)
            elif key in (ord(","), ord(".")):
                self.seek_time(float(self.frames.t[self.idx]) + (-5 if key == ord(",") else 5))
            elif key in (ord("["), ord("]")):
                self.seek_time(float(self.frames.t[self.idx]) + (-30 if key == ord("[") else 30))
            elif key in (ord("+"), ord("=")):
                self.speed = min(self.speed * 2, 32)
                self._restart_clock()
            elif key in (ord("-"), ord("_")):
                self.speed = max(self.speed / 2, 0.25)
                self._restart_clock()
            elif key == ord("n"):
                self.cycle(1)
            elif key == ord("p"):
                self.cycle(-1)
            elif key == ord("x"):
                self.exclude_self = not self.exclude_self
            elif key == ord("l"):
                self.source = self.sources[(self.sources.index(self.source) + 1) % len(self.sources)]
            elif key == ord("v"):
                self.verbose = not self.verbose
            elif key == ord("b"):
                self.show_boxes = not self.show_boxes
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", nargs="?", default="group_01")
    ap.add_argument("experiment", nargs="?", default="experiment_01")
    ap.add_argument("camera", nargs="?", default="context_cam_01")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="DIPSER root (contains DIPSEER/ and abi_labels/)")
    ap.add_argument("--offset", type=float, default=0.0, help="seconds added to label times (label->camera clock)")
    ap.add_argument("--start", type=float, default=0.0, help="start this many seconds into the session")
    ap.add_argument("--scale", type=float, default=0.75, help="video display scale")
    ap.add_argument("--exclude-self", action="store_true", help="median over external labelers only")
    Viewer(ap.parse_args()).run()


if __name__ == "__main__":
    main()

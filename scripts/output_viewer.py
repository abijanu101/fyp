"""Pipeline output viewer: play a session with what the pipeline made of it.

    python -m scripts.output_viewer group_01 experiment_03 context_cam_01 [--set output_dir=\"tmp/e2e6\" ...]
    python -m scripts.output_viewer group_02 experiment_05 context_cam_01 --models outputs/models/<hash>.joblib

Reads only cached results, so scrubbing is instant: the frame pass (`tracks/<hash>/<session>.npz`),
the feature pass (`features/<hash>/<session>.csv` and `.tracks.csv`, from `run_pipeline extract`)
and a model bundle from `train_models` (default: the one trained under this exact config, else the
newest). Use the same --set values the session was extracted and trained with.

What you see
  video     every track in this frame. Coloured red -> green by that student's predicted engagement
            in the current window (late aggregation); white = tracked but not scored in this window
            (visible < aggregation.min_coverage of it); thin grey = dropped by the tracklet filter.
            With labels on (`l`): the linked student (SID) and their own label (1 / 0 / -)
  header    current window, gate prediction vs true activity, window score vs cut-off,
            labelled share engaged and window label
  timeline  red = engagement score per window, grey = labelled share engaged, dashed = cut-off,
            shaded = current window; bottom strips: predicted activity (top) over true activity

This is a development tool: per-student scores and identities exist only here and in training,
never in the pipeline's outputs.

Mouse     click the timeline to seek
Keys      space play/pause      left/right  -/+1 frame     , .  -/+5 s     [ ]  -/+30 s
          + -   playback speed  l  labels and SIDs          d  dropped tracks       q / Esc quit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from scripts.run_pipeline import features_dir, parse_sets, session_stem, tracks_dir
from src.core.config import config_hash, load_config
from src.core.store import load_track
from src.core.types import SessionKey, Window
from src.datasets import DipserIndex
from src.experts import late_rows, window_scores
from src.ingestion import ZipFrameSource
from src.labels import label_session, student_labels_session
from src.tracking import associate_sids
from src.tracking.tracklets import filter_visible

FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_H, PLOT_H, STRIP_H, LEFT = 46, 110, 9, 44
KEY_LEFT, KEY_RIGHT = (2424832, 65361), (2555904, 65363)   # Windows, X11
ACTIVITY = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148), (75, 86, 140)]
WIN = "Pipeline output viewer"


def engagement_colour(p: float) -> tuple[int, int, int]:
    """BGR from red (0) through yellow (0.5) to green (1)."""
    p = float(np.clip(p, 0, 1))
    return (0, int(220 * min(1, 2 * p)), int(230 * min(1, 2 * (1 - p))))


def pick_models(cfg: dict, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    models = Path(cfg["output_dir"]) / "models"
    exact = models / f"{config_hash(cfg)}.joblib"
    if exact.exists():
        return exact
    found = sorted(models.glob("*.joblib"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"no model bundle in {models}; run scripts.train_models first (same --set values)")
    print(f"WARNING: no model trained under this exact config; using the newest, {found[-1].name}")
    return found[-1]


class OutputViewer:
    def __init__(self, args, cfg: dict):
        index = DipserIndex.from_config(cfg)
        self.key = SessionKey(args.group, args.experiment, args.camera)
        stem = session_stem(self.key)
        npz, csv = tracks_dir(cfg) / f"{stem}.npz", features_dir(cfg) / f"{stem}.csv"
        tcsv = features_dir(cfg) / f"{stem}.tracks.csv"
        if not (npz.exists() and csv.exists() and tcsv.exists()):
            raise SystemExit(f"{self.key} is not extracted under this config. Run:\n  python -m scripts.run_pipeline "
                             f"extract --reuse --session {self.key} [same --set values]")
        self.source = ZipFrameSource(index.frames_zip(self.key))
        self.t = self.source.timestamps

        # frame pass: every raw track observation, and which survive the tracklet filter
        st = load_track(npz)
        fps = st.timing.get("fps") or 1.0
        kept = filter_visible(st.tracklets, st.frame_t, fps, cfg["tracklets"]["min_visible_s"],
                              cfg["tracklets"]["max_gap_s"])
        kept_obs = {(tr.tid, round(float(x), 6)) for tr in kept for x in tr.t}
        self.boxes: dict[int, list[tuple[int, np.ndarray, bool]]] = {}     # frame -> [(tid, xyxy, kept)]
        for tr in st.tracklets:
            for x, b in zip(tr.t, tr.xyxy):
                i = int(np.searchsorted(self.t, x - 1e-6))
                if i < len(self.t) and abs(self.t[i] - x) < 1e-4:
                    self.boxes.setdefault(i, []).append((tr.tid, b, (tr.tid, round(float(x), 6)) in kept_obs))

        # model pass, as Pipeline.predict does it, keeping the per-student scores
        df, tracks = pd.read_csv(csv), pd.read_csv(tcsv)
        path = pick_models(cfg, args.models)
        m = joblib.load(path)
        if m["config_hash"] != config_hash(cfg):
            print(f"WARNING: {path.name} was trained under a different config than the one given")
        self.classes = cfg["classes"]
        self.g = m["gate"].predict_proba(df.reindex(columns=m["gate_cols"]).to_numpy(float))
        self.late = m.get("aggregate", "early") == "late"
        if self.late:
            X, _, w = late_rows(df, tracks, context=False, cols=m["row_cols"])
        else:
            X, w = df.reindex(columns=m["row_cols"]).to_numpy(float), np.arange(len(df))
        p_rows = m["mixture"].predict_proba(X, self.g[w]) if len(w) else np.zeros(0)
        self.score = window_scores(p_rows, w, len(df), m["fill"])
        self.tau = m.get("tau", cfg["mixture"]["tau"])
        self.p_student = ({(int(wi), int(tid)): float(p) for wi, tid, p in zip(w, tracks["tid"], p_rows)}
                          if self.late else {})
        self.t0w, self.t1w = df["t_start"].to_numpy(), df["t_end"].to_numpy()
        self.mid = (self.t0w + self.t1w) / 2

        # labels and identities (development only)
        windows = [Window(a, b) for a, b in zip(self.t0w, self.t1w)]
        labels = label_session(index, self.key, windows, cfg["labels"])
        self.share = np.array([l.share_engaged for l in labels], float)
        self.wlabel = [l.engaged for l in labels]
        self.sid = associate_sids(st.tracklets, index.annotations(self.key), cfg["association"]["min_iou"],
                                  cfg["association"]["min_match_share"])
        self.slabel = student_labels_session(index, self.key, windows, cfg["labels"])

        print(f"{self.key}: {len(df)} windows, {len(kept)} kept tracklets of {len(st.tracklets)}, "
              f"aggregate={'late' if self.late else 'early'}, cut-off {self.tau:g}, models {path.name[:12]}")
        self.show_labels, self.show_dropped = True, True
        self.img_cache: tuple[int | None, np.ndarray | None] = (None, None)
        self.scale = args.scale
        h, w_ = self.frame_image(0).shape[:2]
        self.vw, self.vh = int(w_ * self.scale), int(h * self.scale)
        self.plot_w = self.vw - LEFT - 10
        self.idx = min(int(np.searchsorted(self.t, self.t[0] + args.start)), len(self.t) - 1)
        self.playing, self.speed = False, 1.0
        self._restart_clock()

    # -- data
    def frame_image(self, i: int) -> np.ndarray:
        if self.img_cache[0] != i:
            f = self.source.read(i)
            self.img_cache = (i, f.image if f is not None else np.zeros((720, 1280, 3), np.uint8))
        return self.img_cache[1]

    def boxes_near(self, i: int) -> list:
        """Tracks of frame i, or of the most recent analysed frame up to 3 back (ingestion.stride > 1
        analyses only every n-th frame)."""
        for j in range(i, max(i - 4, -1), -1):
            if j in self.boxes:
                return self.boxes[j]
        return []

    def window_at(self, t: float) -> int | None:
        """The window containing t whose centre is nearest (windows overlap by 50%)."""
        inside = np.flatnonzero((self.t0w <= t) & (t < self.t1w))
        return int(inside[np.argmin(np.abs(self.mid[inside] - t))]) if len(inside) else None

    # -- rendering
    def render(self) -> np.ndarray:
        t = float(self.t[self.idx])
        wi = self.window_at(t)
        img = cv2.resize(self.frame_image(self.idx), (self.vw, self.vh), interpolation=cv2.INTER_AREA)
        s = self.scale
        for tid, b, kept in self.boxes_near(self.idx):
            p1, p2 = (int(b[0] * s), int(b[1] * s)), (int(b[2] * s), int(b[3] * s))
            if not kept:
                if self.show_dropped:
                    cv2.rectangle(img, p1, p2, (150, 150, 150), 1)
                continue
            p = self.p_student.get((wi, tid)) if wi is not None else None
            col = engagement_colour(p) if p is not None else (235, 235, 235)
            cv2.rectangle(img, p1, p2, col, 2)
            text = f"{tid}" + (f" p{p:.2f}" if p is not None else "")
            sid = self.sid.get(tid)
            if self.show_labels and sid:
                lab = self.slabel.get(sid, np.full(len(self.mid), np.nan))[wi] if wi is not None else np.nan
                text += f" {sid.replace('subject_', 'S')} L{'-' if not np.isfinite(lab) else int(lab)}"
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.42, 1)
            cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 6, p1[1]), col, -1)
            cv2.putText(img, text, (p1[0] + 3, p1[1] - 4), FONT, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        return np.vstack([self.header(t, wi), img, self.timeline(wi)])

    def header(self, t: float, wi: int | None) -> np.ndarray:
        img = np.full((HEADER_H, self.vw, 3), 30, np.uint8)
        line1 = (f"{self.key}  +{t - self.t[0]:.1f}s  frame {self.idx + 1}/{len(self.t)}  "
                 f"{'PLAY' if self.playing else 'PAUSE'} x{self.speed:g}   "
                 f"[l] labels {'on' if self.show_labels else 'off'}  [d] dropped {'on' if self.show_dropped else 'off'}")
        cv2.putText(img, line1, (8, 18), FONT, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        if wi is None:
            line2, col = "no window here (session edge)", (160, 160, 160)
        else:
            k = int(self.g[wi].argmax())
            ctx = self.classes[k]
            engaged = self.score[wi] >= self.tau
            lab = self.wlabel[wi]
            share = f"{self.share[wi]:.2f}" if np.isfinite(self.share[wi]) else "-"
            line2 = (f"window {wi + 1}/{len(self.mid)}  "
                     f"activity {ctx} ({self.g[wi, k]:.2f}) true {self.key.exp_id}{'' if ctx == self.key.exp_id else ' WRONG'}  |  "
                     f"score {self.score[wi]:.2f} {'>=' if engaged else '<'} {self.tau:g} -> {'ENGAGED' if engaged else 'not'}  |  "
                     f"labelled share {share}, label {'-' if lab is None else ('engaged' if lab else 'not')}")
            col = (120, 230, 120) if (lab is None or bool(lab) == engaged) else (90, 140, 255)
        cv2.putText(img, line2, (8, 38), FONT, 0.42, col, 1, cv2.LINE_AA)
        return img

    def x_of(self, t: float) -> int:
        return LEFT + int((t - self.t[0]) / max(self.t[-1] - self.t[0], 1e-9) * self.plot_w)

    def timeline(self, wi: int | None) -> np.ndarray:
        img = np.full((PLOT_H + 2 * STRIP_H + 18, self.vw, 3), 22, np.uint8)
        y_of = lambda v: 6 + int((1 - v) * (PLOT_H - 12))
        if wi is not None:
            cv2.rectangle(img, (self.x_of(self.t0w[wi]), 4), (self.x_of(self.t1w[wi]), PLOT_H - 4), (55, 55, 55), -1)
        for v in (0, 0.5, 1):
            cv2.putText(img, f"{v:g}", (8, y_of(v) + 4), FONT, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.line(img, (LEFT, y_of(v)), (LEFT + self.plot_w, y_of(v)), (45, 45, 45), 1)
        for x in range(LEFT, LEFT + self.plot_w, 12):                                   # dashed cut-off
            cv2.line(img, (x, y_of(self.tau)), (min(x + 6, LEFT + self.plot_w), y_of(self.tau)), (200, 200, 200), 1)
        xs = np.array([self.x_of(m) for m in self.mid])
        ok = np.isfinite(self.share)
        if ok.sum() > 1:
            pts = np.stack([xs[ok], [y_of(v) for v in self.share[ok]]], 1).astype(np.int32)
            cv2.polylines(img, [pts], False, (170, 170, 170), 1, cv2.LINE_AA)
        if len(xs) > 1:
            pts = np.stack([xs, [y_of(v) for v in self.score]], 1).astype(np.int32)
            cv2.polylines(img, [pts], False, (60, 60, 235), 2, cv2.LINE_AA)
        true_col = ACTIVITY[self.classes.index(self.key.exp_id) % len(ACTIVITY)]
        y = PLOT_H
        for i in range(len(self.mid)):
            a = self.x_of(self.mid[i] - (self.mid[1] - self.mid[0]) / 2 if len(self.mid) > 1 else self.t0w[i])
            b = self.x_of(self.mid[i] + (self.mid[1] - self.mid[0]) / 2 if len(self.mid) > 1 else self.t1w[i])
            cv2.rectangle(img, (a, y), (b, y + STRIP_H - 1), ACTIVITY[int(self.g[i].argmax()) % len(ACTIVITY)], -1)
        cv2.rectangle(img, (LEFT, y + STRIP_H), (LEFT + self.plot_w, y + 2 * STRIP_H - 1), true_col, -1)
        cv2.putText(img, "gate", (4, y + STRIP_H - 1), FONT, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, "true", (4, y + 2 * STRIP_H - 1), FONT, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
        x0 = LEFT
        for k, c in enumerate(self.classes):
            cv2.rectangle(img, (x0, y + 2 * STRIP_H + 4), (x0 + 10, y + 2 * STRIP_H + 13), ACTIVITY[k % len(ACTIVITY)], -1)
            cv2.putText(img, c, (x0 + 13, y + 2 * STRIP_H + 13), FONT, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            x0 += 40
        cv2.putText(img, "red: score  grey: labelled share  dashed: cut-off", (x0 + 10, y + 2 * STRIP_H + 13),
                    FONT, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        x = self.x_of(float(self.t[self.idx]))
        cv2.line(img, (x, 2), (x, y + 2 * STRIP_H), (255, 255, 255), 1)
        self.timeline_y0 = HEADER_H + self.vh
        return img

    # -- interaction
    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and y >= getattr(self, "timeline_y0", 1 << 30) and x >= LEFT:
            frac = (x - LEFT) / self.plot_w
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
                cv2.resizeWindow(WIN, img.shape[1], min(img.shape[0], 1000))
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
            elif key == ord("l"):
                self.show_labels = not self.show_labels
            elif key == ord("d"):
                self.show_dropped = not self.show_dropped
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", nargs="?", default="group_01")
    ap.add_argument("experiment", nargs="?", default="experiment_03")
    ap.add_argument("camera", nargs="?", default="context_cam_01")
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--models", help="model bundle (default: trained under this config, else the newest)")
    ap.add_argument("--start", type=float, default=0.0, help="start this many seconds into the session")
    ap.add_argument("--scale", type=float, default=0.75, help="video display scale")
    ap.add_argument("--snapshot", help="render the frame at --start to this PNG and exit (no window)")
    args = ap.parse_args()
    viewer = OutputViewer(args, load_config(args.config, parse_sets(args.set)))
    if args.snapshot:
        cv2.imwrite(args.snapshot, viewer.render())
        print(f"wrote {args.snapshot}")
        return
    viewer.run()


if __name__ == "__main__":
    main()

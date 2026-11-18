"""Experiment 09 pilot: which pose-model input gives usable head keypoints on DIPSER's classroom
cameras, and at what cost? Runs the head_pose unit (src/bfmu/head_pose.py) in several variants on a
sample of frames from a few sessions, on the tracks already cached by experiment 06 (no new frame pass).

  python -m scripts.analysis.head_pose_pilot [--frames 40] [--out outputs/experiments/09_vfoa/pilot]

Per variant: seconds per frame of the pose step; over all (frame, track) pairs, the share with a
matched pose, with a yaw value, with a head-height value; mean face confidence; yaw and head-height
percentiles. Overlays (boxes, keypoints, yaw) for two frames per session and variant, for eyeballing.
Writes summary.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.run_pipeline import session_stem, tracks_dir
from src.bfmu.head_pose import HeadPose
from src.core.config import load_config
from src.core.store import load_track
from src.core.types import SessionKey, TrackState
from src.datasets import DipserIndex
from src.ingestion.source import ZipFrameSource

SESSIONS = ["group_01/experiment_03/context_cam_01", "group_02/experiment_04/context_cam_02",
            "group_03/experiment_01/context_cam_03"]
VARIANTS = {
    "n_frame640": {"weights": "models/yolo11n-pose.pt", "mode": "frame", "imgsz": 640},
    "n_frame1280": {"weights": "models/yolo11n-pose.pt", "mode": "frame", "imgsz": 1280},
    "n_crops256": {"weights": "models/yolo11n-pose.pt", "mode": "crops", "crop_size": 256},
    "s_frame1280": {"weights": "models/yolo11s-pose.pt", "mode": "frame", "imgsz": 1280},
    "s_crops256": {"weights": "models/yolo11s-pose.pt", "mode": "crops", "crop_size": 256},
}


def states_at(tracklets, t: float, tol: float = 1e-3) -> list[TrackState]:
    out = []
    for tr in tracklets:
        i = np.searchsorted(tr.t, t)
        for j in (i - 1, i):
            if 0 <= j < len(tr.t) and abs(tr.t[j] - t) < tol:
                out.append(TrackState(tr.tid, tr.xyxy[j], float(tr.conf[j]), tr.velocity[j]))
                break
    return out


def overlay(image, tracks, sig, path: Path) -> None:
    import cv2
    img = image.copy()
    for s in tracks:
        x1, y1, x2, y2 = map(int, s.xyxy)
        v = sig.get(s.tid, {})
        ok = v.get("hp.matched", 0) == 1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0) if ok else (0, 0, 220), 2)
        yaw, hh = v.get("hp.yaw", np.nan), v.get("hp.head_height", np.nan)
        cv2.putText(img, f"y{yaw:+.2f} h{hh:.2f} f{v.get('hp.face_conf', np.nan):.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="docs/experiments/06_full_pipeline_and_labels.json")
    ap.add_argument("--frames", type=int, default=40, help="frames sampled per session")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--out", default="outputs/experiments/09_vfoa/pilot")
    args = ap.parse_args()
    cfg = load_config(args.config)
    index = DipserIndex.from_config(cfg)
    out = Path(args.out)

    samples = []                                   # (stem, frame, tracks)
    for s in SESSIONS:
        key = SessionKey(*s.split("/"))
        st = load_track(tracks_dir(cfg) / f"{session_stem(key)}.npz")
        src = ZipFrameSource(index.frames_zip(key))
        idx = np.linspace(0, len(src.timestamps) - 1, args.frames).astype(int)
        for i in idx:
            frame = src.read(int(i))
            if frame is None:
                continue
            tracks = states_at(st.tracklets, frame.t)
            if tracks:
                samples.append((session_stem(key), frame, tracks))
    print(f"{len(samples)} frames with tracks from {len(SESSIONS)} sessions")

    summary = {}
    for name in args.variants:
        unit = HeadPose(**VARIANTS[name])
        unit.frame_signals(samples[0][1], samples[0][2])           # load + warm up, not timed
        rows, secs, shown = [], 0.0, {}
        for stem, frame, tracks in samples:
            t0 = time.perf_counter()
            sig = unit.frame_signals(frame, tracks)
            secs += time.perf_counter() - t0
            rows += list(sig.values())
            if shown.get(stem, 0) < 2:
                overlay(frame.image, tracks, sig, out / name / f"{stem}_{frame.index}.jpg")
                shown[stem] = shown.get(stem, 0) + 1
        def col(k):
            return np.array([r[k] for r in rows], float)
        yaw, hh = col("hp.yaw"), col("hp.head_height")
        summary[name] = {
            "frames": len(samples), "track_frames": len(rows), "s_per_frame": secs / len(samples),
            "matched": float(np.mean(col("hp.matched"))), "yaw_defined": float(np.mean(np.isfinite(yaw))),
            "head_height_defined": float(np.mean(np.isfinite(hh))),
            "face_conf_mean": float(np.nanmean(col("hp.face_conf"))),
            "yaw_p10_50_90": [float(x) for x in np.nanpercentile(yaw, [10, 50, 90])] if np.isfinite(yaw).any() else None,
            "head_height_p10_50_90": [float(x) for x in np.nanpercentile(hh, [10, 50, 90])] if np.isfinite(hh).any() else None,
        }
        print(f"  {name}: done")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n| Variant | s / frame | matched | yaw defined | head height defined | face conf | yaw p10 / 50 / 90 |")
    print("|---|---|---|---|---|---|---|")
    for name, m in summary.items():
        yq = " / ".join(f"{x:+.2f}" for x in m["yaw_p10_50_90"]) if m["yaw_p10_50_90"] else "-"
        print(f"| {name} | {m['s_per_frame']:.3f} | {m['matched']:.2f} | {m['yaw_defined']:.2f} | "
              f"{m['head_height_defined']:.2f} | {m['face_conf_mean']:.2f} | {yq} |")
    print(f"\nwrote {out}/summary.json and overlays per variant")


if __name__ == "__main__":
    main()

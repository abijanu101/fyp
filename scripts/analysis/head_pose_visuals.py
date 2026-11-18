"""Quick visuals of what the head-pose unit (src/bfmu/head_pose.py) sees, for eyeballing experiment 09.

  python -m scripts.analysis.head_pose_visuals [--frames 3] [--out outputs/experiments/09_vfoa/visuals]

For a few frames from one session per group and activity mix, on the tracks experiment 06 cached:
  frames/<stem>_<i>.jpg     the frame twice, side by side: the nano pose model at 640 (left) and the
                            small model at 1280 (right, the one experiment 09 uses). Per tracked
                            student: the box (green = pose matched, red = not), the face and
                            shoulder keypoints (bright = confident, dim = below 0.5), an arrow from
                            the ears' (or shoulders') midpoint through the nose showing the yaw, and
                            the yaw / head height values.
  heads/<stem>_<i>.jpg      every tracked student's head and shoulders, cut out and enlarged 3x with
                            the small model's keypoints, so the small heads can actually be seen.
  distributions.png         yaw, head height and face confidence over every sampled student-frame,
                            both models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from scripts.analysis.head_pose_pilot import states_at
from scripts.run_pipeline import session_stem, tracks_dir
from src.bfmu.head_pose import head_geometry, match_poses
from src.core.config import load_config
from src.core.store import load_track
from src.core.types import SessionKey
from src.datasets import DipserIndex
from src.ingestion.source import ZipFrameSource

SESSIONS = ["group_01/experiment_03/context_cam_01", "group_01/experiment_06/context_cam_02",
            "group_02/experiment_04/context_cam_02", "group_02/experiment_01/context_cam_03",
            "group_03/experiment_01/context_cam_03", "group_03/experiment_05/context_cam_01"]
MODELS = {"nano @640": ("models/yolo11n-pose.pt", 640), "small @1280": ("models/yolo11s-pose.pt", 1280)}
KP = {0: "nose", 1: "l_eye", 2: "r_eye", 3: "l_ear", 4: "r_ear", 5: "l_sho", 6: "r_sho"}
KP_COLOR = {0: (0, 0, 255), 1: (255, 0, 255), 2: (255, 0, 255), 3: (255, 255, 0), 4: (255, 255, 0),
            5: (0, 255, 255), 6: (0, 255, 255)}


def poses(model, image, imgsz):
    r = model.predict(image, imgsz=imgsz, conf=0.1, verbose=False, half=True)[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return np.empty((0, 4)), np.empty((0, 17, 2)), np.empty((0, 17))
    return r.boxes.xyxy.cpu().numpy(), r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()


def draw(image, tracks, matched, kxy, kc, title, scale=1.0):
    img = image.copy()
    for t, s in enumerate(tracks):
        x1, y1, x2, y2 = (s.xyxy * scale).astype(int)
        p = matched.get(t)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0) if p is not None else (0, 0, 220), 2)
        if p is None:
            continue
        xy, c = kxy[p] * scale, kc[p]
        for k, col in KP_COLOR.items():
            dim = c[k] < 0.5
            cv2.circle(img, tuple(xy[k].astype(int)), max(2, int(3 * scale)),
                       tuple(int(v * (0.35 if dim else 1)) for v in col), -1)
        g = head_geometry(kxy[p], c)
        ref = (3, 4) if c[3] >= 0.5 and c[4] >= 0.5 else (5, 6)
        if c[0] >= 0.5 and c[ref[0]] >= 0.5 and c[ref[1]] >= 0.5:
            mid = (xy[ref[0]] + xy[ref[1]]) / 2
            mid = np.array([mid[0], xy[0][1]])                      # level with the nose
            tip = mid + (xy[0] - mid) * 3
            cv2.arrowedLine(img, tuple(mid.astype(int)), tuple(tip.astype(int)), (255, 255, 255), 2, tipLength=0.3)
        cv2.putText(img, f"y{g['hp.yaw']:+.2f} h{g['hp.head_height']:.2f}", (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def head_grid(image, tracks, matched, kxy, kc, zoom=3):
    tiles = []
    for t, s in enumerate(tracks):
        x1, y1, x2, y2 = s.xyxy
        h = (y2 - y1) * 0.6
        cx1, cy1, cx2, cy2 = (max(0, int(x1)), max(0, int(y1 - 0.1 * h)),
                              min(image.shape[1], int(x2)), min(image.shape[0], int(y1 + h)))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            continue
        crop = cv2.resize(image[cy1:cy2, cx1:cx2], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        p = matched.get(t)
        if p is not None:
            for k, col in KP_COLOR.items():
                x, y = (kxy[p][k] - [cx1, cy1]) * zoom
                if 0 <= x < crop.shape[1] and 0 <= y < crop.shape[0]:
                    cv2.circle(crop, (int(x), int(y)), 5, col if kc[p][k] >= 0.5 else tuple(v // 3 for v in col), -1)
            g = head_geometry(kxy[p], kc[p])
            label = f"y{g['hp.yaw']:+.2f} h{g['hp.head_height']:.2f} f{g['hp.face_conf']:.2f}"
        else:
            label = "no pose"
        cv2.putText(crop, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(crop, (300, 260)))
    if not tiles:
        return None
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="docs/experiments/06_full_pipeline_and_labels.json")
    ap.add_argument("--frames", type=int, default=3, help="frames per session")
    ap.add_argument("--out", default="outputs/experiments/09_vfoa/visuals")
    args = ap.parse_args()
    from ultralytics import YOLO
    cfg = load_config(args.config)
    index = DipserIndex.from_config(cfg)
    out = Path(args.out)
    for d in ("frames", "heads"):
        (out / d).mkdir(parents=True, exist_ok=True)
    models = {name: (YOLO(w), imgsz) for name, (w, imgsz) in MODELS.items()}
    dist = {name: {"yaw": [], "head_height": [], "face_conf": []} for name in MODELS}
    for s in SESSIONS:
        key = SessionKey(*s.split("/"))
        path = tracks_dir(cfg) / f"{session_stem(key)}.npz"
        if not path.exists():
            print(f"  {s}: no cached tracks, skipped")
            continue
        st = load_track(path)
        src = ZipFrameSource(index.frames_zip(key))
        for i in np.linspace(len(src.timestamps) * 0.15, len(src.timestamps) * 0.85, args.frames).astype(int):
            frame = src.read(int(i))
            if frame is None:
                continue
            tracks = states_at(st.tracklets, frame.t)
            if not tracks:
                continue
            boxes = np.stack([t.xyxy for t in tracks])
            panels = []
            for name, (model, imgsz) in models.items():
                pb, kxy, kc = poses(model, frame.image, imgsz)
                matched = match_poses(pb, boxes, 0.3)
                panels.append(draw(frame.image, tracks, matched, kxy, kc, name))
                for p in matched.values():
                    g = head_geometry(kxy[p], kc[p])
                    for k in ("yaw", "head_height", "face_conf"):
                        dist[name][k].append(g[f"hp.{k}"])
                if name == "small @1280":
                    grid = head_grid(frame.image, tracks, matched, kxy, kc)
                    if grid is not None:
                        cv2.imwrite(str(out / "heads" / f"{session_stem(key)}_{i}.jpg"), grid)
            cv2.imwrite(str(out / "frames" / f"{session_stem(key)}_{i}.jpg"),
                        cv2.resize(np.hstack(panels), None, fx=0.75, fy=0.75))
        print(f"  {s}: done")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, k, rng in zip(axes, ("yaw", "head_height", "face_conf"), ((-1.5, 1.5), (-0.6, 1.0), (0, 1))):
        for (name, d), col in zip(dist.items(), ("#eb6834", "#2a78d6")):
            x = np.array(d[k], float)
            x = x[np.isfinite(x)]
            ax.hist(x, bins=40, range=rng, alpha=0.55, color=col, label=f"{name} (n {len(x)})")
        ax.set_title(k.replace("_", " "))
        ax.legend(fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Head-pose signals over the sampled student-frames")
    fig.tight_layout()
    fig.savefig(out / "distributions.png", dpi=120)
    print(f"wrote {out}/frames, {out}/heads, {out}/distributions.png")


if __name__ == "__main__":
    main()

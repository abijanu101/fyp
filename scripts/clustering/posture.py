"""Body posture from the pose model's body keypoints, on the cached tracks (no re-tracking). Experiment
10/11 follow-up: the only untested information that is not facial.

  python -m scripts.clustering.posture [--config docs/experiments/10_clustering.json] [--set k=v ...]
         [--every 1] [--imgsz 1280] [--device 0] [--max-frames 0] [--overlays 0]
         [--out outputs/experiments/10_clustering_v2]

For every original recording, the pose model (`detector.weights`, a *-pose model: experiment 09's
yolo11s-pose) runs on the full frame at --imgsz, and its poses are matched one-to-one to the cached
tracks of that frame by box IoU (head_pose's frame mode, `match_poses`). From the matched pose, BODY
keypoints only (shoulders, elbows, wrists, hips; never the nose, eyes or ears), per track and frame:
  po.sho_height    shoulder midpoint below the box top, over the box height (larger = lower: slumped)
  po.sho_tilt      left minus right shoulder height, over shoulder width
  po.lean          shoulder midpoint minus hip midpoint, horizontally, over shoulder width (hips seen)
  po.torso         hip midpoint below the shoulder midpoint, over shoulder width (shorter when leaning
                   toward the camera; hips seen)
  po.wrist_h       wrists above the shoulder line, over shoulder width (mean of the visible wrists)
  po.hand_up       1 if a wrist is above the shoulder line
  po.hand_head     1 if a wrist is above the shoulder line and between the shoulders (a hand at the head)
  po.arms_crossed  1 if both wrists are on the opposite side of the shoulder midpoint
  po.elbow_spread  elbow distance over shoulder width
  po.visible       share of the 8 body keypoints above kp_conf
  po.matched       1 if a pose was matched to the track
Lengths are over shoulder width, so distance to the camera does not scale them (experiment 10's seat
check found the existing features do vary with it). NaN where the needed keypoints are below kp_conf.

Written to <features cache>/../posture/<tracks hash>-<settings hash>/<stem>.npz (tid, t, one array per
signal); a recording already written is skipped (resumable). --overlays N saves N annotated frames per
recording for checking. Window features (mean, and SD of sho_height and wrist_h) for any rows with
(stem, tid, t) come from window_features().
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12
BODY = [L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP]
SIGNALS = ("po.sho_height", "po.sho_tilt", "po.lean", "po.torso", "po.wrist_h", "po.hand_up", "po.hand_head",
           "po.arms_crossed", "po.elbow_spread", "po.visible", "po.matched")
SD_OF = ("po.sho_height", "po.wrist_h")


def posture(xy: np.ndarray, conf: np.ndarray, box: np.ndarray, kp_conf: float = 0.5) -> dict[str, float]:
    """Posture from one person's COCO keypoints (xy (17, 2) pixels, conf (17,)) and the track box
    (x1, y1, x2, y2). Image y grows downward. Body keypoints only."""
    ok = conf >= kp_conf
    out = {k: np.nan for k in SIGNALS}
    out["po.matched"] = 1.0
    out["po.visible"] = float(np.mean(ok[BODY]))
    if not (ok[L_SHO] and ok[R_SHO]):
        return out
    sw = abs(xy[L_SHO, 0] - xy[R_SHO, 0])
    if sw <= 1:
        return out
    smx, smy = (xy[L_SHO, 0] + xy[R_SHO, 0]) / 2, (xy[L_SHO, 1] + xy[R_SHO, 1]) / 2
    bh = max(box[3] - box[1], 1.0)
    out["po.sho_height"] = float((smy - box[1]) / bh)
    out["po.sho_tilt"] = float((xy[L_SHO, 1] - xy[R_SHO, 1]) / sw)
    if ok[L_HIP] and ok[R_HIP]:
        hmx, hmy = (xy[L_HIP, 0] + xy[R_HIP, 0]) / 2, (xy[L_HIP, 1] + xy[R_HIP, 1]) / 2
        out["po.lean"] = float((smx - hmx) / sw)
        out["po.torso"] = float((hmy - smy) / sw)
    wr = [w for w in (L_WRI, R_WRI) if ok[w]]
    if wr:
        above = [(smy - xy[w, 1]) / sw for w in wr]
        out["po.wrist_h"] = float(np.mean(above))
        out["po.hand_up"] = float(any(a > 0 for a in above))
        lo, hi = min(xy[L_SHO, 0], xy[R_SHO, 0]), max(xy[L_SHO, 0], xy[R_SHO, 0])
        out["po.hand_head"] = float(any(a > 0 and lo <= xy[w, 0] <= hi for a, w in zip(above, wr)))
    if ok[L_WRI] and ok[R_WRI]:
        l_side = np.sign(xy[L_SHO, 0] - smx)                         # which side the left shoulder is on
        out["po.arms_crossed"] = float(np.sign(xy[L_WRI, 0] - smx) == -l_side and
                                       np.sign(xy[R_WRI, 0] - smx) == l_side)
    if ok[L_ELB] and ok[R_ELB]:
        out["po.elbow_spread"] = float(abs(xy[L_ELB, 0] - xy[R_ELB, 0]) / sw)
    return out


def unmatched() -> dict[str, float]:
    out = {k: np.nan for k in SIGNALS}
    out["po.matched"] = 0.0
    return out


def settings_hash(s: dict) -> str:
    return hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:8]


def cache_dir(cfg: dict, settings: dict) -> Path:
    from scripts.run_pipeline import features_dir, tracks_dir

    return features_dir(cfg).parent.parent / "posture" / f"{tracks_dir(cfg).name}-{settings_hash(settings)}"


def frame_tracks(st, t: float, tol: float = 1e-3):
    """The cached tracks present at frame time t: [(tid, box)]."""
    out = []
    for tr in st.tracklets:
        i = np.searchsorted(tr.t, t)
        for j in (i - 1, i):
            if 0 <= j < len(tr.t) and abs(tr.t[j] - t) < tol:
                out.append((tr.tid, np.asarray(tr.xyxy[j], float)))
                break
    return out


def extract_recording(cfg: dict, key, settings: dict, model, out_path: Path, overlays: int = 0) -> int:
    """Posture signals for every tracked student in every --every-th frame of one recording."""
    from scripts.run_pipeline import session_stem, tracks_dir
    from src.bfmu.head_pose import match_poses
    from src.core.store import load_track
    from src.datasets import DipserIndex
    from src.ingestion.source import ZipFrameSource

    st = load_track(tracks_dir(cfg) / f"{session_stem(key)}.npz")
    src = ZipFrameSource(DipserIndex.from_config(cfg).frames_zip(key))
    track_t = np.unique(np.concatenate([tr.t for tr in st.tracklets])) if st.tracklets else np.zeros(0)
    idx = [i for i in range(len(src.timestamps))][::settings["every"]]
    if settings["max_frames"]:
        idx = idx[:settings["max_frames"]]
    rows, shown = [], 0
    for i in idx:
        frame = src.read(int(i))
        if frame is None:
            continue
        j = np.searchsorted(track_t, frame.t)
        cand = [track_t[k] for k in (j - 1, j) if 0 <= k < len(track_t)]
        if not cand or min(abs(c - frame.t) for c in cand) > 1e-3:
            continue
        tracks = frame_tracks(st, frame.t)
        if not tracks:
            continue
        kw = {"conf": settings["conf"], "imgsz": settings["imgsz"], "verbose": False}
        if settings.get("device") is not None:
            kw["device"] = settings["device"]
        res = model.predict(frame.image, **kw)[0]
        sig = {tid: unmatched() for tid, _ in tracks}
        if res.keypoints is not None and len(res.boxes):
            boxes = np.stack([b for _, b in tracks])
            kxy, kc = res.keypoints.xy.cpu().numpy(), res.keypoints.conf.cpu().numpy()
            for ti, p in match_poses(res.boxes.xyxy.cpu().numpy(), boxes, settings["min_iou"]).items():
                tid, box = tracks[ti]
                sig[tid] = posture(kxy[p], kc[p], box, settings["kp_conf"])
                if shown < overlays:
                    _overlay(frame.image, kxy[p], kc[p], box, sig[tid], out_path.with_name(f"{out_path.stem}_{i}_{tid}.jpg"),
                             settings["kp_conf"])
                    shown += 1
        rows += [{"tid": tid, "t": frame.t, **v} for tid, v in sig.items()]
    df = pd.DataFrame(rows, columns=["tid", "t", *SIGNALS])
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, tid=df["tid"].to_numpy(int), t=df["t"].to_numpy(float),
                        **{k: df[k].to_numpy(float) for k in SIGNALS})
    tmp.replace(out_path)
    return len(df)


def _overlay(image, xy, conf, box, sig, path: Path, kp_conf: float) -> None:
    import cv2
    img = image.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
    for k in BODY:
        if conf[k] >= kp_conf:
            cv2.circle(img, (int(xy[k, 0]), int(xy[k, 1])), 4, (0, 140, 255), -1)
    txt = f"sh{sig['po.sho_height']:.2f} wr{sig['po.wrist_h']:.2f} up{sig['po.hand_up']:.0f} vis{sig['po.visible']:.2f}"
    cv2.putText(img, txt, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    pad = 60
    crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop if crop.size else img)


def load_signals(cdir: Path, stem: str) -> pd.DataFrame | None:
    p = cdir / f"{stem}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return pd.DataFrame({"tid": z["tid"], "t": z["t"], **{k: z[k] for k in SIGNALS}})


def window_features(cdir: Path, rows: pd.DataFrame, length_s: float) -> pd.DataFrame:
    """Per row (columns stem, tid, t): the mean of every posture signal over the track's frames within
    length_s/2 of t, and the SD of sho_height and wrist_h; columns trk.po.<name>[.sd]. NaN where the
    recording has no posture cache (augmented copies) or the track no frame there."""
    names = [f"trk.{s}" for s in SIGNALS] + [f"trk.{s}.sd" for s in SD_OF]
    M = np.full((len(rows), len(names)), np.nan)
    half = length_s / 2
    sd_idx = [SIGNALS.index(s) for s in SD_OF]
    stem_a, tid_a, t_a = rows["stem"].to_numpy(), rows["tid"].to_numpy(), rows["t"].to_numpy(float)
    for stem in np.unique(stem_a):
        sig = load_signals(cdir, stem)
        if sig is None or sig.empty:
            continue
        in_stem = np.flatnonzero(stem_a == stem)
        for tid, g in sig.groupby("tid"):
            pos = in_stem[tid_a[in_stem] == tid]
            if len(pos) == 0:
                continue
            g = g.sort_values("t")
            tg, V = g["t"].to_numpy(), g[list(SIGNALS)].to_numpy(float)
            lo = np.searchsorted(tg, t_a[pos] - half, side="left")
            hi = np.searchsorted(tg, t_a[pos] + half, side="right")
            for k, a_, b_ in zip(pos, lo, hi):
                if b_ > a_:
                    w = V[a_:b_]
                    with np.errstate(invalid="ignore"):
                        M[k, :len(SIGNALS)] = np.nanmean(w, 0)
                        for c, si in enumerate(sd_idx):
                            col = w[:, si]
                            M[k, len(SIGNALS) + c] = np.nanstd(col) if np.isfinite(col).sum() >= 2 else np.nan
    return pd.DataFrame(M, index=rows.index, columns=names)


def main() -> None:
    import warnings

    from scripts.clustering.common import base_parser, load
    from scripts.run_pipeline import session_stem
    from src.datasets import DipserIndex

    warnings.filterwarnings("ignore", message="Mean of empty slice")
    ap = base_parser(__doc__)
    ap.add_argument("--every", type=int, default=1, help="every N-th frame")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--kp-conf", type=float, default=0.5)
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--device", default=None, help="e.g. 0 for the first GPU; default: ultralytics' choice")
    ap.add_argument("--max-frames", type=int, default=0, help="per recording (0: all): for a pilot")
    ap.add_argument("--overlays", type=int, default=0, help="annotated frames saved per recording")
    args = ap.parse_args()
    cfg, out = load(args)
    from ultralytics import YOLO

    settings = {"weights": cfg["detector"]["weights"], "every": args.every, "imgsz": args.imgsz, "conf": args.conf,
                "kp_conf": args.kp_conf, "min_iou": args.min_iou, "max_frames": args.max_frames,
                "device": args.device}
    cdir = cache_dir(cfg, {k: v for k, v in settings.items() if k != "device"})
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
    keys = DipserIndex.from_config(cfg).sessions()
    if args.sessions:
        keys = [k for k in keys if session_stem(k) in set(args.sessions)]
    model = YOLO(settings["weights"])
    todo = [k for k in keys if not (cdir / f"{session_stem(k)}.npz").exists()]
    print(f"{len(keys)} recordings ({len(keys) - len(todo)} done already) -> {cdir}", flush=True)
    for i, k in enumerate(todo, 1):
        t0 = time.time()
        n = extract_recording(cfg, k, settings, model, cdir / f"{session_stem(k)}.npz", args.overlays)
        print(f"  [{i}/{len(todo)}] {session_stem(k)}: {n} track-frames in {time.time() - t0:.0f} s", flush=True)
    print(f"wrote {cdir}")


if __name__ == "__main__":
    main()

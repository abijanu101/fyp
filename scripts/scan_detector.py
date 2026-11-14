"""Does the detector's confidence threshold drop students? A tracker-free check against the manual boxes.

  python -m scripts.scan_detector [--thresholds 0.05 0.1 ...] [--set key=value ...]

Uses the floor-confidence detections that `tune_tracker` caches (outputs/cache/detections/, every
frame of every in-scope session at tuning.tracker.det_conf_floor), so it needs no GPU and runs in
about a minute. For each threshold, each frame's boxes with conf >= threshold are matched one-to-one
(Hungarian on IoU, >= association.min_iou) to the manual student boxes, the same rule the tracker is
scored with. Per student, recall = share of frames in which they are matched.

The floor is the reference: a student missed even at the floor is not lost to the threshold (the
detector does not see them, or their manual box does not fit them). What a threshold costs is the
drop from the floor.

Writes outputs/tuning/detector_scan.csv (one row per threshold) and detector_scan_students.csv
(one row per student and threshold).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.run_pipeline import parse_sets
from scripts.tune_tracker import SessionDetections
from src.core.config import load_config
from src.datasets import DipserIndex
from src.tracking.association import iou_matrix

DEFAULT_THRESHOLDS = [0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


def session_recall(sd: SessionDetections, subjects: dict[str, np.ndarray], thresholds: list[float],
                   min_iou: float) -> dict[str, np.ndarray]:
    """{sid: (len(thresholds),) share of frames in which the student is matched}."""
    sids = sorted(subjects)
    boxes = np.stack([subjects[s] for s in sids])
    hits = np.zeros((len(sids), len(thresholds)))
    for d in sd.dets:
        if not len(d):
            continue
        iou = iou_matrix(boxes, d.xyxy)                                   # (students, detections)
        for k, thr in enumerate(thresholds):
            keep = d.conf >= thr
            if not keep.any():
                continue
            m = iou[:, keep]
            r, c = linear_sum_assignment(-m)
            hits[r[m[r, c] >= min_iou], k] += 1
    n = max(len(sd.dets), 1)
    return {s: hits[i] / n for i, s in enumerate(sids)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--thresholds", nargs="*", type=float, default=DEFAULT_THRESHOLDS)
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    floor = cfg["tuning"]["tracker"]["det_conf_floor"]
    thresholds = sorted(set([floor] + [t for t in args.thresholds if t >= floor]))
    min_iou = cfg["association"]["min_iou"]
    index = DipserIndex.from_config(cfg)

    def no_detector():
        raise SystemExit("no cached detections for a session; run `python -m scripts.tune_tracker --profile full` "
                         "(it caches every session's detections first) or point --set output_dir at the cache")

    rows = []
    for i, key in enumerate(index.sessions(), 1):
        sd = SessionDetections.load_or_detect(cfg, index, key, None, floor, no_detector)
        subjects = index.annotations(key)
        for sid, rec in session_recall(sd, subjects, thresholds, min_iou).items():
            rows += [{"session": str(key), "group": key.group, "student": sid, "threshold": t, "recall": r}
                     for t, r in zip(thresholds, rec)]
        print(f"\r[{i}/{len(index.sessions())}] {key}", end="", flush=True)
    print()
    per = pd.DataFrame(rows)
    ref = per[per.threshold == floor].set_index(["session", "student"])["recall"]
    per["drop_vs_floor"] = ref.reindex(pd.MultiIndex.from_frame(per[["session", "student"]])).to_numpy() - per["recall"]

    def summary(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "students": len(g),
            "mean_recall": g.recall.mean(),
            "p10_recall": g.recall.quantile(0.10),
            "min_recall": g.recall.min(),
            "never_detected": int((g.recall == 0).sum()),
            "below_50pct": int((g.recall < 0.5).sum()),
            "max_drop_vs_floor": g.drop_vs_floor.max(),
            "lost_10pts_vs_floor": int((g.drop_vs_floor > 0.10).sum()),
            "lost_25pts_vs_floor": int((g.drop_vs_floor > 0.25).sum()),
        })
    table = per.groupby("threshold").apply(summary).reset_index()
    out = Path(cfg["output_dir"]) / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "detector_scan.csv", index=False)
    per.to_csv(out / "detector_scan_students.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    conf = cfg["detector"]["conf"]
    at = per[np.isclose(per.threshold, conf)]
    if len(at):
        worst = at.sort_values("drop_vs_floor", ascending=False).head(5)
        print(f"\nlargest losses at detector.conf={conf} against the floor {floor}:")
        print(worst[["session", "student", "recall", "drop_vs_floor"]].to_string(index=False))
    print(f"-> {out / 'detector_scan.csv'}")


if __name__ == "__main__":
    main()

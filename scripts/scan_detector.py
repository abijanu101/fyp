"""Which detector confidence threshold keeps every student? A tracker-free check against the manual boxes.

  python -m scripts.scan_detector                       # binary search for the optimal threshold
  python -m scripts.scan_detector --grid [--thresholds 0.05 0.1 ...]   # full table instead
  python -m scripts.scan_detector --augment [COPY] --detect            # the same on augmented data

Uses the floor-confidence detections that `tune_tracker` caches (outputs/cache/detections/, every
frame of every in-scope session at tuning.tracker.det_conf_floor). For a threshold, each frame's
boxes with conf >= threshold are matched one-to-one (Hungarian on IoU >= association.min_iou) to
the manual student boxes, the rule the tracker is scored with. Per student, recall = share of
frames in which they are matched. The floor is the reference: a student missed even at the floor
is not lost to the threshold.

Binary search (default). Recall can only fall as the threshold rises, so "keeps every student" is
monotone and bisection finds the highest threshold that does, in about 8 passes: the optimal
threshold in the sense of the fewest extra boxes (people who are not students, false positives)
without losing anyone. A threshold keeps every student if no student is undetected, none is
detected in fewer than half their frames, and none loses more than --max-drop (0.10) of recall
against the floor. Each probe also reports `unmatched_share`: kept boxes that match no annotated
student, the price of a lower threshold.

--augment [COPY] runs on augmented copy COPY of the data (scripts/augment_dataset.py; written if
missing, boxes mirrored where flipped). Augmented data has its own detection cache; --detect fills
missing caches with the detector (GPU).

Writes to outputs/tuning/: detector_search[_<aug tag>].csv (every probe) and .json (the result), or
with --grid detector_scan[_<aug tag>].csv and detector_scan_students[...].csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.run_pipeline import parse_sets
from scripts.tune_tracker import SessionDetections
from src.core.config import load_config
from src.core.pipeline import Pipeline
from src.datasets import DipserIndex
from src.ingestion import aug_tag, augmented_root
from src.tracking.association import iou_matrix

DEFAULT_THRESHOLDS = [0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


def session_recall(sd: SessionDetections, subjects: dict[str, np.ndarray], thresholds: list[float],
                   min_iou: float) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """({sid: (len(thresholds),) share of frames in which the student is matched},
    (len(thresholds),) boxes kept, (len(thresholds),) kept boxes matched to a student)."""
    sids = sorted(subjects)
    boxes = np.stack([subjects[s] for s in sids])
    hits = np.zeros((len(sids), len(thresholds)))
    kept, matched = np.zeros(len(thresholds)), np.zeros(len(thresholds))
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
            ok = m[r, c] >= min_iou
            hits[r[ok], k] += 1
            kept[k] += keep.sum()
            matched[k] += ok.sum()
    n = max(len(sd.dets), 1)
    return {s: hits[i] / n for i, s in enumerate(sids)}, kept, matched


def evaluate(sessions, thresholds: list[float], floor_recall: dict | None, min_iou: float):
    """Per-student rows and a summary per threshold, for the loaded sessions."""
    rows, totals = [], {t: [0.0, 0.0] for t in thresholds}
    for sd, subjects in sessions:
        rec, kept, matched = session_recall(sd, subjects, thresholds, min_iou)
        for sid, r in rec.items():
            rows += [{"session": str(sd.key), "group": sd.key.group, "student": sid, "threshold": t, "recall": v}
                     for t, v in zip(thresholds, r)]
        for k, t in enumerate(thresholds):
            totals[t][0] += kept[k]
            totals[t][1] += matched[k]
    per = pd.DataFrame(rows)
    if floor_recall is None:
        floor_recall = per[per.threshold == thresholds[0]].set_index(["session", "student"])["recall"]
    per["drop_vs_floor"] = floor_recall.reindex(pd.MultiIndex.from_frame(per[["session", "student"]])).to_numpy() - per["recall"]

    def summary(g: pd.DataFrame) -> pd.Series:
        kept, matched = totals[g.name]
        return pd.Series({
            "students": len(g), "mean_recall": g.recall.mean(), "p10_recall": g.recall.quantile(0.10),
            "min_recall": g.recall.min(), "never_detected": int((g.recall == 0).sum()),
            "below_50pct": int((g.recall < 0.5).sum()), "max_drop_vs_floor": g.drop_vs_floor.max(),
            "lost_10pts_vs_floor": int((g.drop_vs_floor > 0.10).sum()),
            "lost_25pts_vs_floor": int((g.drop_vs_floor > 0.25).sum()),
            "unmatched_share": 1 - matched / kept if kept else np.nan})
    return per, per.groupby("threshold").apply(summary).reset_index(), floor_recall


def keeps_everyone(row: pd.Series, max_drop: float) -> bool:
    return row.never_detected == 0 and row.below_50pct == 0 and row.max_drop_vs_floor <= max_drop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--grid", action="store_true", help="full table over --thresholds instead of the binary search")
    ap.add_argument("--thresholds", nargs="*", type=float, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--max-drop", type=float, default=0.10, help="largest recall loss against the floor per student")
    ap.add_argument("--tolerance", type=float, default=0.005, help="binary search stops at this threshold width")
    ap.add_argument("--augment", nargs="?", type=int, const=1, metavar="COPY", help="augmented copy COPY (default 1)")
    ap.add_argument("--detect", action="store_true", help="run the detector (GPU) where detections are not cached")
    ap.add_argument("--workers", type=int, default=3, help="--augment: parallel writers for missing copies")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    floor = cfg["tuning"]["tracker"]["det_conf_floor"]
    min_iou = cfg["association"]["min_iou"]
    keys = DipserIndex.from_config(cfg).sessions()
    suffix = ""
    if args.augment:
        from scripts.augment_dataset import ensure_many
        ensure_many(cfg, [(k, args.augment) for k in keys], args.workers, log=lambda m: print(m, flush=True))
        suffix = f"_{aug_tag(cfg['augmentation'], args.augment)}"
        cfg = {**cfg, "dataset": {**cfg["dataset"], "root": str(augmented_root(cfg["augmentation"], args.augment))}}
    index = DipserIndex.from_config(cfg)

    pipe = Pipeline({**cfg, "detector": {**cfg["detector"], "conf": floor}}) if args.detect else None

    def detector():
        if pipe is None:
            raise SystemExit("no cached detections for a session: add --detect to run the detector, or run "
                             "`python -m scripts.tune_tracker --profile full` (it caches every session first)")
        return pipe.detector

    sessions = []
    for i, key in enumerate(keys, 1):
        sessions.append((SessionDetections.load_or_detect(cfg, index, key, None, floor, detector), index.annotations(key)))
        print(f"\rloaded [{i}/{len(keys)}] {key}", end="", flush=True)
    print()
    out = Path(cfg["output_dir"]) / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    fmt = {"display.width": 220, "display.max_columns": 20, "display.float_format": "{:.3f}".format}

    if args.grid:
        thresholds = sorted(set([floor] + [t for t in args.thresholds if t >= floor]))
        per, table, _ = evaluate(sessions, thresholds, None, min_iou)
        table.to_csv(out / f"detector_scan{suffix}.csv", index=False)
        per.to_csv(out / f"detector_scan_students{suffix}.csv", index=False)
        with pd.option_context(*[x for kv in fmt.items() for x in kv]):
            print(table.to_string(index=False))
        print(f"-> {out / f'detector_scan{suffix}.csv'}")
        return

    # binary search: the highest threshold that keeps every student
    _, base, floor_recall = evaluate(sessions, [floor], None, min_iou)
    probes = [base.iloc[0]]
    lo, hi = floor, 0.95
    while hi - lo > args.tolerance:
        mid = round((lo + hi) / 2, 4)
        _, t, _ = evaluate(sessions, [mid], floor_recall, min_iou)
        row = t.iloc[0]
        ok = keeps_everyone(row, args.max_drop)
        probes.append(row)
        print(f"  threshold {mid:.4f}: {'keeps everyone' if ok else 'loses students'}  mean recall {row.mean_recall:.3f}  "
              f"min {row.min_recall:.3f}  max drop {row.max_drop_vs_floor:.3f}  unmatched {row.unmatched_share:.3f}",
              flush=True)
        lo, hi = (mid, hi) if ok else (lo, mid)
    table = pd.DataFrame(probes).sort_values("threshold")
    table.to_csv(out / f"detector_search{suffix}.csv", index=False)
    best = table[np.isclose(table.threshold, lo)].iloc[0] if lo > floor else table.iloc[0]
    conf = cfg["detector"]["conf"]
    _, at_conf, _ = evaluate(sessions, [conf], floor_recall, min_iou)
    result = {"dataset": suffix.lstrip("_") or "original", "criterion": {"never_detected": 0, "below_50pct": 0,
              "max_drop_vs_floor": args.max_drop}, "floor": floor, "optimal_threshold": lo,
              "at_optimal": best.to_dict(), "configured_threshold": conf, "at_configured": at_conf.iloc[0].to_dict(),
              "configured_keeps_everyone": bool(keeps_everyone(at_conf.iloc[0], args.max_drop))}
    (out / f"detector_search{suffix}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"\nhighest threshold keeping every student: {lo:.4f} (configured {conf}: "
          f"{'keeps every student' if result['configured_keeps_everyone'] else 'fails the criterion'}) "
          f"-> {out / f'detector_search{suffix}.json'}")


if __name__ == "__main__":
    main()

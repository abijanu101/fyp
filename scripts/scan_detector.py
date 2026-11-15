"""Which detector confidence threshold keeps every student? A tracker-free check against the manual boxes.

  python -m scripts.scan_detector                      # binary search for the optimal threshold
  python -m scripts.scan_detector --grid [--thresholds 0.05 0.1 ...]   # full table instead
  python -m scripts.scan_detector --augment [--variants 5 10]          # under augmentation (hours, resumable)

For a threshold, each frame's boxes with conf >= threshold are matched one-to-one (Hungarian on
IoU >= association.min_iou) to the manual student boxes, the rule the tracker is scored with. Per
student, recall = share of frames in which they are matched. The detections at the confidence
floor (tuning.tracker.det_conf_floor) are the reference: a student missed even at the floor is
not lost to the threshold.

Binary search (default). Recall only falls as the threshold rises, so "keeps every student" is
monotone and bisection finds the highest threshold that does, in about 8 passes: the fewest extra
boxes (people who are not students, false positives) without losing anyone. A threshold keeps
every student if none is undetected, none is detected in fewer than half their frames, and none
loses more than --max-drop (0.10) of recall against the floor. Each probe also reports
`unmatched_share`: kept boxes that match no annotated student, the price of a lower threshold.

Original data: uses the floor detections `tune_tracker` caches; --detect runs the detector where
they are missing.

--augment: the detector is the single point of failure, so this mode tests it hard. Every frame of
every in-scope session is augmented on the fly K times (K drawn per frame from --variants, 5..10),
each variant with its own independently drawn parameters from config `augmentation` (colour,
brightness, gamma, noise, flip, masks), and each variant is detected at the floor. Student boxes
are mirrored for flipped variants. Nothing is written but the detections themselves, one file per
session (outputs/cache/detections_augmented/<settings>/), which makes the run resumable and lets
the bisection re-score thresholds without re-running the detector. Roughly (frames x mean K) x
(detector + augmentation time): several hours for all 52 sessions; --stride N uses every N-th frame.

Writes to outputs/tuning/: detector_search[_augmented].csv (every probe) and .json (the result), or
with --grid detector_scan[_augmented].csv and detector_scan_students[...].csv.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from scripts.run_pipeline import parse_sets, session_stem
from scripts.tune_tracker import SessionDetections
from src.core.config import config_hash, load_config
from src.core.pipeline import Pipeline
from src.core.types import Frame
from src.datasets import DipserIndex
from src.ingestion import ZipFrameSource, hflip_boxes, sample_augmentation
from src.ingestion.augment import AUG_VERSION, augment_frame
from src.tracking.association import iou_matrix

DEFAULT_THRESHOLDS = [0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


# One detected image: (xyxy (N, 4), conf (N,), flipped, frame width). Original frames are unflipped.
def original_variants(sd: SessionDetections) -> list[tuple]:
    return [(d.xyxy, d.conf, False, 0) for d in sd.dets]


def augmented_variants(cfg: dict, index: DipserIndex, key, detector, variants: tuple[int, int], stride: int,
                       cache_dir: Path) -> list[tuple]:
    """Every (stride-th) frame of the session, augmented K times (K in `variants`, drawn per frame),
    each variant with independent parameters, detected at the floor. Cached per session (atomic)."""
    path = cache_dir / f"{session_stem(key)}.npz"
    if not path.exists():
        aug = cfg["augmentation"]
        rng = np.random.default_rng([aug.get("seed", 0), zlib.crc32(str(key).encode())])
        xyxy, conf, counts, flips, widths = [], [], [], [], []
        source, t0 = ZipFrameSource(index.frames_zip(key), stride=stride), time.monotonic()
        for n, frame in enumerate(source, 1):
            print(f"    {key}: frame {n}/{len(source.timestamps)}, {(time.monotonic() - t0) / 60:.1f} min", flush=True)
            for j in range(int(rng.integers(variants[0], variants[1] + 1))):
                p = sample_augmentation(aug, f"{key}@{frame.index}", j)          # independent per variant
                img = augment_frame(frame.image, p, frame.index)
                d = detector.detect(Frame(frame.index, frame.t, img))
                xyxy.append(d.xyxy)
                conf.append(d.conf)
                counts.append(len(d.conf))
                flips.append(p["hflip"])
                widths.append(img.shape[1])
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, counts=np.array(counts), flips=np.array(flips), widths=np.array(widths),
                            xyxy=np.concatenate(xyxy) if xyxy else np.zeros((0, 4)),
                            conf=np.concatenate(conf) if conf else np.zeros(0))
        tmp.replace(path)
    with np.load(path) as z:
        counts, flips, widths, xyxy, conf = z["counts"], z["flips"], z["widths"], z["xyxy"], z["conf"]
    b = np.cumsum(np.concatenate([[0], counts]))
    return [(xyxy[s:e], conf[s:e], bool(f), int(w)) for s, e, f, w in zip(b[:-1], b[1:], flips, widths)]


_worker_pipe = {}


def _augment_session(cfg: dict, key, variants: tuple[int, int], stride: int, cache_dir: Path) -> str:
    """Worker: build this process's detector once, write one session's augmented detections."""
    if "p" not in _worker_pipe:
        floor = cfg["tuning"]["tracker"]["det_conf_floor"]
        _worker_pipe["p"] = Pipeline({**cfg, "detector": {**cfg["detector"], "conf": floor}})
    augmented_variants(cfg, DipserIndex.from_config(cfg), key, _worker_pipe["p"].detector, variants, stride, cache_dir)
    return str(key)


def session_recall(images: list[tuple], subjects: dict[str, np.ndarray], thresholds: list[float],
                   min_iou: float) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """({sid: (len(thresholds),) share of images in which the student is matched},
    (len(thresholds),) boxes kept, (len(thresholds),) kept boxes matched to a student)."""
    sids = sorted(subjects)
    boxes = np.stack([subjects[s] for s in sids])
    hits = np.zeros((len(sids), len(thresholds)))
    kept, matched = np.zeros(len(thresholds)), np.zeros(len(thresholds))
    for xyxy, conf, flipped, width in images:
        if not len(conf):
            continue
        iou = iou_matrix(hflip_boxes(boxes, width) if flipped else boxes, xyxy)   # (students, detections)
        for k, thr in enumerate(thresholds):
            keep = conf >= thr
            if not keep.any():
                continue
            m = iou[:, keep]
            r, c = linear_sum_assignment(-m)
            ok = m[r, c] >= min_iou
            hits[r[ok], k] += 1
            kept[k] += keep.sum()
            matched[k] += ok.sum()
    n = max(len(images), 1)
    return {s: hits[i] / n for i, s in enumerate(sids)}, kept, matched


def evaluate(sessions, thresholds: list[float], floor_recall, min_iou: float):
    """Per-student rows and a summary per threshold, for the loaded sessions [(key, images, subjects)]."""
    rows, totals = [], {t: [0.0, 0.0] for t in thresholds}
    for key, images, subjects in sessions:
        rec, kept, matched = session_recall(images, subjects, thresholds, min_iou)
        for sid, r in rec.items():
            rows += [{"session": str(key), "group": key.group, "student": sid, "threshold": t, "recall": v}
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
    ap.add_argument("--augment", action="store_true", help="augment every frame on the fly, K variants per frame")
    ap.add_argument("--variants", nargs=2, type=int, default=[5, 10], metavar=("MIN", "MAX"),
                    help="--augment: variants per frame, drawn per frame from MIN..MAX")
    ap.add_argument("--stride", type=int, default=1, help="--augment: use every N-th frame")
    ap.add_argument("--workers", type=int, default=4,
                    help="--augment: sessions processed in parallel (each with its own detector)")
    ap.add_argument("--detect", action="store_true", help="original data: run the detector where not cached")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    floor = cfg["tuning"]["tracker"]["det_conf_floor"]
    min_iou = cfg["association"]["min_iou"]
    index = DipserIndex.from_config(cfg)
    keys = index.sessions()
    pipe = Pipeline({**cfg, "detector": {**cfg["detector"], "conf": floor}})

    def detector():
        if not (args.detect or args.augment):
            raise SystemExit("no cached detections for a session: add --detect to run the detector, or run "
                             "`python -m scripts.tune_tracker --profile full` (it caches every session first)")
        return pipe.detector

    suffix = "_augmented" if args.augment else ""
    if args.augment:
        settings = {"detector": {**cfg["detector"], "conf": floor}, "augmentation":
                    {k: v for k, v in cfg["augmentation"].items() if k not in ("copies", "root")},
                    "variants": args.variants, "stride": args.stride, "version": AUG_VERSION}
        cache_dir = Path(cfg["output_dir"]) / "cache" / "detections_augmented" / config_hash(settings)[:12]
        print(f"augmenting every {'' if args.stride == 1 else f'{args.stride}th '}frame of {len(keys)} sessions, "
              f"{args.variants[0]}-{args.variants[1]} variants each; detections cached in {cache_dir}", flush=True)

    sessions, start = [], time.monotonic()
    if args.augment and args.workers > 1:            # write the per-session detections in parallel first
        from concurrent.futures import ProcessPoolExecutor, as_completed
        todo = [k for k in keys if not (cache_dir / f"{session_stem(k)}.npz").exists()]
        print(f"{len(todo)} sessions to augment and detect, {args.workers} worker processes", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_augment_session, cfg, k, tuple(args.variants), args.stride, cache_dir) for k in todo]
            for n, f in enumerate(as_completed(futures), 1):
                print(f"  [{n}/{len(todo)}] {f.result()} done, {(time.monotonic() - start) / 60:.0f} min elapsed", flush=True)
    for i, key in enumerate(keys, 1):
        if args.augment:
            images = augmented_variants(cfg, index, key, detector(), tuple(args.variants), args.stride, cache_dir)
        else:
            images = original_variants(SessionDetections.load_or_detect(cfg, index, key, None, floor, detector))
        sessions.append((key, images, index.annotations(key)))
        el = (time.monotonic() - start) / 60
        print(f"[{i}/{len(keys)}] {key}: {len(images)} images, {el:.0f} min elapsed", flush=True)
    out = Path(cfg["output_dir"]) / "tuning"
    out.mkdir(parents=True, exist_ok=True)

    if args.grid:
        thresholds = sorted(set([floor] + [t for t in args.thresholds if t >= floor]))
        per, table, _ = evaluate(sessions, thresholds, None, min_iou)
        table.to_csv(out / f"detector_scan{suffix}.csv", index=False)
        per.to_csv(out / f"detector_scan_students{suffix}.csv", index=False)
        with pd.option_context("display.width", 220, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
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
    result = {"data": "augmented on the fly" if args.augment else "original",
              **({"variants_per_frame": args.variants, "stride": args.stride,
                  "images": int(sum(len(s[1]) for s in sessions))} if args.augment else {}),
              "criterion": {"never_detected": 0, "below_50pct": 0, "max_drop_vs_floor": args.max_drop},
              "floor": floor, "optimal_threshold": lo, "at_optimal": best.to_dict(),
              "at_floor": base.iloc[0].to_dict(), "configured_threshold": conf,
              "at_configured": at_conf.iloc[0].to_dict(),
              "configured_keeps_everyone": bool(keeps_everyone(at_conf.iloc[0], args.max_drop))}
    (out / f"detector_search{suffix}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"\nhighest threshold keeping every student: {lo:.4f} (configured {conf}: "
          f"{'keeps every student' if result['configured_keeps_everyone'] else 'fails the criterion'}) "
          f"-> {out / f'detector_search{suffix}.json'}")


if __name__ == "__main__":
    main()

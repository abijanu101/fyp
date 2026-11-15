"""Trivial baselines on DIPSER attention labels (docs/misc/benchmarks.md §1).

Scores a constant prediction of 3 (and of the target's mean) against each target the published
works use: the mean of the experts, the majority vote (with and without the self-report), and
every labeler's own ratings. Reads only the labels/*.json inside each subject ZIP.

  python -m scripts.label_baselines            # the published targets, four ways (benchmarks.md §1)
  python -m scripts.label_baselines --grid     # every target of docs/misc/targets.md, as a markdown table
  python -m scripts.label_baselines --grid --set window.length_s=4   # per-window rows at another window

Every metric is computed four ways, because papers differ on both choices and rarely say which:

  sampling   per second: the target is read once per second (labels hold until changed), so a
             score that holds for ten minutes counts 600 times;
             per change: each run of a constant target value counts once, however long it held.
  pooling    pooled: every sample from every subject in one pile;
             per subject: the metric per student-session, then the mean over them.

Each subject's grid runs from the latest first label of any of its labelers to the last label event.
Majority ties go to the value closest to 3, then the lower one. Not engaged = attention below 3.
"""

from __future__ import annotations

import argparse
import glob

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.labels.attention import SELF_LABELER, load_subject_labels


def majority(M: np.ndarray) -> np.ndarray:
    out = []
    for col in M.T.astype(int):
        c = np.bincount(col, minlength=6)[1:]
        best = np.flatnonzero(c == c.max()) + 1
        out.append(best[np.argmin(np.abs(best - 3) * 10 + best)])
    return np.array(out, float)


def runs(y: np.ndarray) -> np.ndarray:
    """One sample per run of equal consecutive values."""
    keep = np.r_[True, y[1:] != y[:-1]]
    return y[keep]


def load_targets():
    targets = {"mean of experts": [], "majority, all six incl. self": [], "majority, experts only": [],
               "each labeler, pooled": []}
    for z in sorted(glob.glob("data/DIPSER/DIPSEER/*/*/subjects/*.zip")):
        L = {k: v for k, v in load_subject_labels(z).items() if len(v.t)}
        experts = [k for k in L if k != SELF_LABELER]
        if not experts:
            continue
        t0, t1 = max(v.t[0] for v in L.values()), max(v.t[-1] for v in L.values())
        ts = np.arange(np.ceil(t0), t1 + 1, 1.0)
        if not len(ts):
            continue
        E = np.array([[L[k].at(t) for t in ts] for k in experts])
        A = np.array([[L[k].at(t) for t in ts] for k in L])
        targets["mean of experts"].append(E.mean(0))
        targets["majority, all six incl. self"].append(majority(A))
        targets["majority, experts only"].append(majority(E))
        targets["each labeler, pooled"].append([row for row in A])   # one series per labeler
    return targets


def metrics(y: np.ndarray, classes: bool) -> dict:
    p = np.full_like(y, 3.0)
    m = {"RMSE": np.sqrt(np.mean((y - 3) ** 2)), "MAE": np.mean(np.abs(y - 3)),
         "RMSE mean": y.std(), "MAE mean": np.mean(np.abs(y - y.mean()))}   # per subject: its own mean
    if classes:
        yi, pi = y.astype(int), p.astype(int)
        m |= {"acc": accuracy_score(yi, pi),
              "wF1": f1_score(yi, pi, average="weighted", zero_division=0),
              "mF1": f1_score(yi, pi, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0),
              "within1": np.mean(np.abs(yi - 3) <= 1),
              "engaged acc": np.mean(yi >= 3),
              "notF1 const-not": 2 * np.mean(yi < 3) / (1 + np.mean(yi < 3))}
    return m


# ----------------------------------------------------------------------------- the target grid
# unit x aggregation over labelers x self-report (docs/misc/targets.md)

UNITS = ("per label", "per second", "per frame", "per window")
SELF_MODES = ("excl.", "only", "incl.")
SELF_NAMES = {"excl.": "experts", "only": "self", "incl.": "experts + self"}
AGGS = ("median", "mean", "majority", "all")


def aggregate(M: np.ndarray, agg: str) -> np.ndarray:
    """M: (labelers, samples). 'all' keeps every labeler's value as its own sample (NaNs dropped);
    majority rounds non-integer values (window means) first."""
    with np.errstate(all="ignore"):
        if agg == "all":
            return M[np.isfinite(M)]
        ok = np.isfinite(M).any(0)
        M = M[:, ok]
        if agg == "median":
            return np.nanmedian(M, 0)
        if agg == "mean":
            return np.nanmean(M, 0)
    out = []
    for col in np.round(M.T):
        col = col[np.isfinite(col)].astype(int)
        c = np.bincount(col, minlength=6)[1:]
        best = np.flatnonzero(c == c.max()) + 1
        out.append(best[np.argmin(np.abs(best - 3) * 10 + best)])
    return np.array(out, float)


def unit_matrix(L: dict, names: list[str], unit: str, frame_t: np.ndarray, window: dict) -> np.ndarray:
    """(labelers, samples) values of `names` at the unit's sampling points, over the span from the
    latest first label to the last label event of the included labelers. frame_t: the context
    camera's frame times (per frame)."""
    t0, t1 = max(L[k].t[0] for k in names), max(L[k].t[-1] for k in names)
    if unit in ("per second", "per frame"):                  # 1 Hz clock, or the camera's real frames
        ts = np.arange(np.ceil(t0), t1 + 1, 1.0) if unit == "per second" else frame_t[(frame_t >= t0) & (frame_t <= t1)]
        return np.array([[L[k].at(t) for t in ts] for k in names])
    if unit == "per label":                                  # every event of an included labeler
        ts = np.unique(np.concatenate([L[k].t for k in names]))
        ts = ts[(ts >= t0) & (ts <= t1)]
        return np.array([[L[k].at(t) for t in ts] for k in names])
    length = float(window["length_s"])                       # per window: the config's windows,
    step = length * (1 - float(window["overlap"]))           # time-weighted means
    starts = np.arange(t0, t1 - length + 1e-9, step)
    return np.array([[L[k].mean_over(a, a + length) for a in starts] for k in names])


def camera_frames(group: str, experiment: str, subject: str, cache: dict) -> np.ndarray:
    """Frame times of the first context camera (01-03) that annotates this subject, else camera 01."""
    from src.datasets import DipserIndex
    from src.ingestion import ZipFrameSource
    from src.core.types import SessionKey
    index = DipserIndex("data/DIPSER", [group], [experiment], ["context_cam_01", "context_cam_02", "context_cam_03"])
    keys = index.sessions()
    key = next((k for k in keys if subject in index.annotations(k)), keys[0] if keys else None)
    if key is None:
        return np.zeros(0)
    if key not in cache:
        cache[key] = np.asarray(ZipFrameSource(index.frames_zip(key)).timestamps, float)
    return cache[key]


def grid(window: dict) -> None:
    cells = {(u, a, s): [] for u in UNITS for a in AGGS for s in SELF_MODES}
    frames_cache: dict = {}
    for z in sorted(glob.glob("data/DIPSER/DIPSEER/*/*/subjects/*.zip")):
        L = {k: v for k, v in load_subject_labels(z).items() if len(v.t)}
        experts = [k for k in L if k != SELF_LABELER]
        if not experts:
            continue
        zp = Path(z)
        frame_t = camera_frames(zp.parts[-4], zp.parts[-3], zp.stem, frames_cache)
        for mode in SELF_MODES:
            names = {"incl.": list(L), "excl.": experts, "only": [SELF_LABELER] if SELF_LABELER in L else []}[mode]
            if not names:
                continue
            for u in UNITS:
                M = unit_matrix(L, names, u, frame_t, window)
                if M.size:
                    for a in AGGS if mode != "only" else ("all",):     # one labeler: every aggregation is the same
                        cells[(u, a, mode)].append(aggregate(M, a))
    print("| Unit | Aggregation | Labelers | Samples | Mean | SD | Exactly 3 | RMSE | MAE | Accuracy | Macro F1 | Share >= 3 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (u, a, mode), parts in cells.items():
        if not parts:
            continue
        y = np.concatenate(parts)
        integer = np.allclose(y, np.round(y))
        # classes: non-integer targets are rounded to the nearest rating, halves toward 3 (marked *)
        half = np.isclose(y % 1, 0.5)
        r = np.where(half, np.where(y < 3, np.ceil(y), np.floor(y)), np.round(y)).astype(int)
        mark = "" if integer else "*"
        acc = f"{np.mean(r == 3):.3f}{mark}"
        mf1 = f"{f1_score(r, np.full(len(r), 3), average='macro', labels=[1, 2, 3, 4, 5], zero_division=0):.3f}{mark}"
        print(f"| {u} | {a if mode != 'only' else 'n/a'} | {SELF_NAMES[mode]} | {len(y):,} | {y.mean():.2f} | {y.std():.2f} | "
              f"{np.mean(np.isclose(y, 3)):.1%} | {np.sqrt(np.mean((y - 3) ** 2)):.3f} | {np.mean(np.abs(y - 3)):.3f} | "
              f"{acc} | {mf1} | {np.mean(y >= 3 - 1e-9):.3f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", action="store_true", help="every target of docs/misc/targets.md as a markdown table")
    ap.add_argument("--config", default="configs/deliverable3.json", help="--grid: window.length_s / overlap from here")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    if args.grid:
        from scripts.run_pipeline import parse_sets
        from src.core.config import load_config
        grid(load_config(args.config, parse_sets(args.set))["window"])
        return
    targets = load_targets()
    print(f"subjects: {len(targets['mean of experts'])}")
    for name, series in targets.items():
        classes = name != "mean of experts"
        if name == "each labeler, pooled":                  # flatten per labeler, keep subject grouping
            subj = [[s for s in sub] for sub in series]
            sec = [np.concatenate(sub) for sub in subj]
            chg = [np.concatenate([runs(s) for s in sub]) for sub in subj]
        else:
            sec, chg = series, [runs(s) for s in series]
        print(f"\n{name}")
        for sampling, data in (("per second", sec), ("per change", chg)):
            pooled = metrics(np.concatenate(data), classes)
            per = [metrics(d, classes) for d in data if len(d)]
            per = {k: np.mean([m[k] for m in per]) for k in per[0]}
            n = sum(len(d) for d in data)
            for pooling, m in (("pooled", pooled), ("per subject", per)):
                print(f"  {sampling:10s} {pooling:11s} n={n:7d} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))


if __name__ == "__main__":
    main()

"""Inspect attention labels before choosing label thresholds (design §8.1, §11 Labels).

  python -m scripts.data_visualization.inspect_labels [--set window.length_s=8 ...]

Thresholds are chosen from the label distribution and agreement, not by optimising model scores
(that would let the model pick its own target). Reports, over in-scope session-cameras:
  * attention value distribution (external labelers vs self-report)
  * per-experiment student consensus distribution
  * window class balance and coverage over a (tau_att, tau_share) grid
  * Krippendorff's ordinal alpha per experiment, with and without the self-report; units are
    (student, window midpoint) items, each labeler contributing the attention value in force
Needs only the frame timestamps from each ZIP's index, not extracted features.
Writes <output_dir>/labels/report.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.run_pipeline import parse_sets
from src.aggregation import make_windows
from src.core.config import load_config
from src.datasets import DipserIndex
from src.evaluation import krippendorff_alpha_ordinal
from src.ingestion import ZipFrameSource, timing_audit
from src.labels import label_session
from src.labels.attention import SELF_LABELER
from src.labels.session import _load

GRID_TAU_ATT = (2.5, 3, 3.5, 4)
GRID_TAU_SHARE = (0.5, 0.6, 0.7, 0.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    index = DipserIndex.from_config(cfg)
    lcfg, win = cfg["labels"], cfg["window"]

    windows = {}
    for key in index.sessions():
        a = timing_audit(ZipFrameSource(index.frames_zip(key)).timestamps)
        windows[key] = make_windows(a["t_first"], a["t_last"], win["length_s"], win["overlap"])

    # Distribution + agreement: once per (group, experiment), all labeled students.
    value_counts = {"external": np.zeros(5), "self": np.zeros(5)}
    consensus = defaultdict(list)
    units = defaultdict(lambda: {"with_self": [], "external_only": []})
    done = set()
    for key, ws in windows.items():
        if (key.group, key.experiment) in done:
            continue
        done.add((key.group, key.experiment))
        for p in index.subject_zips(key).values():
            labelers = _load(p, float(lcfg["offset_s"]))
            for w in ws:
                mid = (w.t_start + w.t_end) / 2
                vals = {n: tr.at(mid) for n, tr in labelers.items()}
                for n, v in vals.items():
                    if np.isfinite(v) and 1 <= v <= 5:
                        value_counts["self" if n == SELF_LABELER else "external"][int(v) - 1] += 1
                units[key.exp_id]["with_self"].append(list(vals.values()))
                units[key.exp_id]["external_only"].append([v for n, v in vals.items() if n != SELF_LABELER])
                means = [m for m in (tr.mean_over(w.t_start, w.t_end) for tr in labelers.values()) if np.isfinite(m)]
                if means:
                    consensus[key.exp_id].append(float(np.median(means)))

    report = {
        "n_sessions": len(windows),
        "attention_share_1_to_5": {k: (v / max(v.sum(), 1)).round(3).tolist() for k, v in value_counts.items()},
        "consensus_by_exp": {e: {"mean": float(np.mean(c)), **{f"share_ge_{t}": float(np.mean(np.array(c) >= t))
                                                              for t in GRID_TAU_ATT}}
                             for e, c in sorted(consensus.items())},
        "krippendorff_alpha_ordinal": {e: {k: krippendorff_alpha_ordinal(u) for k, u in d.items()}
                                       for e, d in sorted(units.items())},
        "grid": [],
    }
    for ta in GRID_TAU_ATT:
        for ts in GRID_TAU_SHARE:
            by_exp, n_win = defaultdict(list), 0
            for key, ws in windows.items():
                n_win += len(ws)
                labs = label_session(index, key, ws, {**lcfg, "tau_att": ta, "tau_share": ts})
                by_exp[key.exp_id] += [l.engaged for l in labs if l.engaged is not None]
            pooled = [v for vs in by_exp.values() for v in vs]
            report["grid"].append({"tau_att": ta, "tau_share": ts, "coverage": len(pooled) / max(n_win, 1),
                                   "pos_rate": float(np.mean(pooled)) if pooled else None,
                                   "pos_rate_by_exp": {e: round(float(np.mean(v)), 3) for e, v in sorted(by_exp.items())}})

    out = Path(cfg["output_dir"]) / "labels"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("attention share 1..5:", report["attention_share_1_to_5"])
    print("\nKrippendorff alpha (ordinal):")
    for e, a in report["krippendorff_alpha_ordinal"].items():
        print(f"  exp {e}: with self {a['with_self']:.3f}   external only {a['external_only']:.3f}")
    print("\nwindow class balance (pos rate pooled | by experiment):")
    for g in report["grid"]:
        pr = "n/a" if g["pos_rate"] is None else f"{g['pos_rate']:.2f}"
        print(f"  tau_att={g['tau_att']:<4} tau_share={g['tau_share']:<4} coverage={g['coverage']:.2f} "
              f"pos={pr}  {g['pos_rate_by_exp']}")
    print(f"\nwrote {out / 'report.json'}")


if __name__ == "__main__":
    main()

"""Compare two per-labeler results files, e.g. the mixture of experts against a single decoder.

  python -m scripts.compare_results outputs/results/<A>.json outputs/results/<B>.json
         [--names MoE single] [--context classified] [--prediction own_labeler]

For each track (expert entries, self entries): the summary metrics of A and B on the configured
variant, fold by fold (the folds are the same groups in both runs), and per activity (each
activity is scored in the fold that holds it out). Baselines (constant 3, training mean) are shown
for reference. Both files must come from engagement.labelers = "per_labeler".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ("auc", "spearman", "balanced_acc", "macro_f1", "mae")


def load(path: str) -> dict:
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    if r.get("mode") != "per_labeler":
        raise SystemExit(f"{path}: not a per-labeler results file")
    return r


def row(name: str, values: dict) -> str:
    return f"| {name} | " + " | ".join("-" if v is None or not np.isfinite(v) else f"{v:.3f}"
                                       for v in (values.get(m) for m in METRICS)) + " |"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--context", default="classified", choices=["classified", "oracle"])
    ap.add_argument("--prediction", default="own_labeler", choices=["own_labeler", "median", "majority"])
    args = ap.parse_args()
    runs = dict(zip(args.names, (load(args.a), load(args.b))))
    head = "| | " + " | ".join(METRICS) + " |\n|---|" + "---|" * len(METRICS)
    key = f"configured/{args.context}/{args.prediction}"
    for track in ("experts", "self"):
        print(f"\n## {track} entries, {key}\n\n{head}")
        for name, r in runs.items():
            ev = r["evaluation"]
            print(row(name, {m: ev[f"{track}/{key}"]["summary"][m]["mean"] for m in METRICS}))
        first = next(iter(runs.values()))["evaluation"]
        for base in ("constant_3", "train_mean"):
            print(row(base, {m: first[f"{track}/{base}"]["summary"][m]["mean"] for m in METRICS}))

        print(f"\nBy fold (AUC; difference {args.names[0]} - {args.names[1]}):\n")
        folds = {n: {f["held_out"]: f for f in r["evaluation"][f"{track}/{key}"]["folds"]} for n, r in runs.items()}
        a, b = (folds[n] for n in args.names)
        print("| held-out group | " + " | ".join(args.names) + " | difference |\n|---|---|---|---|")
        for g in sorted(set(a) & set(b)):
            print(f"| {g} | {a[g]['auc']:.3f} | {b[g]['auc']:.3f} | {a[g]['auc'] - b[g]['auc']:+.3f} |")

        print(f"\nBy activity (AUC / Spearman):\n")
        print("| activity | " + " | ".join(args.names) + " |\n|---|---|---|")
        acts = sorted({c for f in a.values() for c in f.get("per_experiment", {})})
        for c in acts:
            cells = []
            for n in args.names:
                m = next((f["per_experiment"][c] for f in folds[n].values() if c in f.get("per_experiment", {})), {})
                cells.append(f"{m.get('auc', np.nan):.3f} / {m.get('spearman', np.nan):.3f}" if m else "-")
            print(f"| {c} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

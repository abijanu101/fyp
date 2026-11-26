"""Trivial baselines (the floor) on the official OUC-CGE split: scores of models that never look at video.

  python -m scripts.ouc_cge.floor --split-dir <OUC-CGE-main> --out <dir>

Fitted on train, scored on val and test: majority class, uniform random, stratified random
(random runs averaged over 100 seeds). Metrics as in the main benchmark: accuracy, balanced accuracy,
macro-F1, per-class F1, confusion matrix. Writes <out>/floor.json. Needs only numpy and scikit-learn (no torch).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from src.datasets.ouc_cge import read_splits

CLASSES = [0, 1, 2]                    # low, mid, high
NAMES = ["low", "mid", "high"]


def scores(y, p) -> dict:
    return {"accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, labels=CLASSES, average="macro", zero_division=0),
            "f1_per_class": dict(zip(NAMES, f1_score(y, p, labels=CLASSES, average=None, zero_division=0).tolist())),
            "confusion": confusion_matrix(y, p, labels=CLASSES).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=100)
    a = ap.parse_args()
    rows = read_splits(a.split_dir)
    y = {s: np.array([r.label for r in rows if r.split == s]) for s in ("train", "val", "test")}
    X = {s: np.zeros((len(v), 1)) for s, v in y.items()}
    result = {"n": {s: int(len(v)) for s, v in y.items()},
              "train_counts": dict(zip(NAMES, np.bincount(y["train"], minlength=3).tolist()))}
    for name in ("most_frequent", "uniform", "stratified"):
        runs = [0] if name == "most_frequent" else range(a.seeds)
        per = {s: [] for s in ("val", "test")}
        for seed in runs:
            m = DummyClassifier(strategy=name, random_state=seed).fit(X["train"], y["train"])
            for s in per:
                per[s].append(scores(y[s], m.predict(X[s])))
        out = {}
        for s, lst in per.items():
            out[s] = {k: (float(np.mean([d[k] for d in lst])) if not isinstance(lst[0][k], (dict, list)) else lst[0][k])
                      for k in lst[0]}
            if len(lst) > 1:
                out[s]["macro_f1_std"] = float(np.std([d["macro_f1"] for d in lst]))
                out[s].pop("confusion")
                out[s]["f1_per_class"] = {c: float(np.mean([d["f1_per_class"][c] for d in lst])) for c in NAMES}
        result[name] = out
    path = Path(a.out)
    path.mkdir(parents=True, exist_ok=True)
    (path / "floor.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{'baseline':<14}{'split':<6}{'acc':>7}{'bal_acc':>9}{'macro_f1':>10}")
    for name in ("most_frequent", "uniform", "stratified"):
        for s in ("val", "test"):
            d = result[name][s]
            print(f"{name:<14}{s:<6}{d['accuracy']:>7.3f}{d['balanced_accuracy']:>9.3f}{d['macro_f1']:>10.3f}")


if __name__ == "__main__":
    main()

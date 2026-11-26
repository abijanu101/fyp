"""Shortcut check: how well can the label be predicted from file properties alone, without any pixels?

  python -m scripts.ouc_cge.metadata_baseline --clips <audit out>/clips.csv --out <dir>

Uses the audit table (scripts.ouc_cge.audit). Arms, each a random forest and a logistic regression
fitted on train and scored on val/test (repo pipeline: impute -> standardise -> classifier):
  duration            clip length in seconds
  format              fps, width, height (camera and recording settings: a proxy for the session)
  duration+format
  bitrate             bytes per second (compressed size tracks how much the picture changes, a crude
                      motion signal, and also the camera / encoder: part signal, part shortcut)
  all                 everything above
If these beat the floor clearly, the official split carries recording identity across train and
test, and the pixel-based scores must be read against these numbers, not only against the floor.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.core.classifiers import build_classifier

ARMS = {"duration": ["duration_s"], "format": ["fps", "width", "height"],
        "duration+format": ["duration_s", "fps", "width", "height"], "bitrate": ["bitrate"],
        "all": ["duration_s", "fps", "width", "height", "bitrate"]}
MODELS = {"random_forest": {"n_estimators": 300, "min_samples_leaf": 5, "class_weight": "balanced", "n_jobs": -1},
          "logreg": {"class_weight": "balanced"}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    d = pd.read_csv(a.clips)
    d = d[d["readable"].fillna(False).astype(bool)].copy()
    d["bitrate"] = d["bytes"] / d["duration_s"].clip(lower=0.1)
    tr = d[d["split"] == "train"]
    res = {"n": d["split"].value_counts().to_dict(), "runs": []}
    print(f"{'model':<15}{'arm':<17}{'split':<6}{'acc':>7}{'bal_acc':>9}{'macro_f1':>10}")
    for arm, cols in ARMS.items():
        for name, params in MODELS.items():
            m = build_classifier(name, seed=0, **params).fit(tr[cols].to_numpy(float), tr["label"])
            for s in ("val", "test"):
                te = d[d["split"] == s]
                p = m.predict(te[cols].to_numpy(float))
                r = {"model": name, "arm": arm, "split": s, "n": int(len(te)),
                     "accuracy": float(accuracy_score(te["label"], p)),
                     "balanced_accuracy": float(balanced_accuracy_score(te["label"], p)),
                     "macro_f1": float(f1_score(te["label"], p, average="macro"))}
                res["runs"].append(r)
                print(f"{name:<15}{arm:<17}{s:<6}{r['accuracy']:>7.3f}{r['balanced_accuracy']:>9.3f}{r['macro_f1']:>10.3f}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata_baseline.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

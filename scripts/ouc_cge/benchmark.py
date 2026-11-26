"""Classifiers on the extracted OUC-CGE clip features, scored on the official split.

  python -m scripts.ouc_cge.benchmark [--config configs/ouc_cge.json] [--set key=value ...]

Reads <output_dir>/cache/features/<feature hash>/ouc_cge/*.csv (scripts.ouc_cge.extract), fits each
model in `ouc_cge.models` on the train clips for each feature-family arm in `ouc_cge.arms` (families:
`ouc_cge.families`, column prefixes), and scores val and test. The floor (majority class) is
included. Every model is the repo's impute -> standardise -> classifier pipeline
(src/core/classifiers.py), fitted on training clips only. Clips whose features are missing are
reported; the test set is never used to choose anything.

Clips shorter than `ouc_cge.min_duration_s` are dropped from every split before fitting (default 1 s:
116 high clips in the authors' archives decode to 4 frames or not at all; see the benchmark doc).
Set it to 3.0 to also drop the 341 clips under 3 s (305 of them low), which the tracker leaves with no
students (tracklets.min_visible_s) and which otherwise let "no students" stand in for "low".
The `duration` family (`clip.duration_s`, the clip length) is the shortcut check: on its own it should
score near the floor; if it does not, the split carries recording identity.

Writes outputs/ouc_cge/results/<config hash>.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from scripts.ouc_cge.extract import out_dir
from scripts.run_pipeline import parse_sets
from src.core.classifiers import build_classifier
from src.core.config import config_hash, load_config
from src.core.pipeline import feature_hash
from src.datasets.ouc_cge import LEVEL_NAMES, read_splits

CLASSES = [0, 1, 2]


def scores(y, p) -> dict:
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)),
            "f1_per_class": dict(zip(LEVEL_NAMES, f1_score(y, p, labels=CLASSES, average=None,
                                                           zero_division=0).round(4).tolist())),
            "confusion": confusion_matrix(y, p, labels=CLASSES).tolist()}


def load_features(cfg: dict) -> pd.DataFrame:
    d = out_dir(cfg)
    files = sorted(p for p in d.glob("ouc_cge__*.csv") if not p.name.endswith(".tracks.csv"))
    if not files:
        raise SystemExit(f"no clip features in {d}: run scripts.ouc_cge.extract first")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    labels = pd.DataFrame([vars(c) for c in read_splits(cfg["dataset"]["split_dir"], verbose=False)])
    df = df.merge(labels[["clip_id", "label", "split"]], on="clip_id", how="inner")
    df.insert(1, "clip.duration_s", df["t_end"] - df["t_start"])
    return df


def columns(df: pd.DataFrame, fam: dict, arm: list[str]) -> list[str]:
    prefixes = [p for f in arm for p in fam[f]]
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ouc_cge.json")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    cfg = load_config(a.config, parse_sets(a.set))
    oc = cfg["ouc_cge"]
    df = load_features(cfg)
    min_dur = float(oc.get("min_duration_s") or 0.0)
    dropped = df[df["clip.duration_s"] < min_dur]
    df = df[df["clip.duration_s"] >= min_dur]
    n_split = df["split"].value_counts().to_dict()
    expected = pd.Series([c.split for c in read_splits(cfg["dataset"]["split_dir"], verbose=False)]).value_counts().to_dict()
    part = {s: (int(n_split.get(s, 0)), int(expected.get(s, 0))) for s in ("train", "val", "test")}
    tr, sets = df[df["split"] == "train"], {s: df[df["split"] == s] for s in ("val", "test")}
    if tr.empty:
        raise SystemExit("no training clips extracted yet")

    rows, result = [], {"config_hash": config_hash(cfg), "feature_hash": feature_hash(cfg),
                        "clips_extracted_vs_split": part, "min_duration_s": min_dur,
                        "dropped_short": {s: dropped["label"][dropped["split"] == s].value_counts().sort_index().to_dict()
                                          for s in ("train", "val", "test")},
                        "no_tracks_share": float((df.get("vec.n_tracks", pd.Series(0.0)) <= 0).mean()),
                        "runs": []}
    major = int(tr["label"].mode()[0])
    for s, d in sets.items():
        if len(d):
            sc = scores(d["label"].to_numpy(), np.full(len(d), major))
            result["runs"].append({"model": "majority", "arm": "-", "split": s, **sc})
    for arm in oc["arms"]:
        cols = columns(df, oc["families"], arm)
        if not cols:
            print(f"arm {'+'.join(arm)}: no columns, skipped")
            continue
        for name, params in oc["models"].items():
            m = build_classifier(name, seed=cfg["seed"], **params).fit(tr[cols].to_numpy(float), tr["label"].to_numpy())
            for s, d in sets.items():
                if len(d):
                    sc = scores(d["label"].to_numpy(), m.predict(d[cols].to_numpy(float)))
                    result["runs"].append({"model": name, "arm": "+".join(arm), "n_features": len(cols),
                                           "split": s, **sc})
    out = Path(cfg["output_dir"]) / "ouc_cge" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{result['config_hash']}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"clips used / in split: {part}  (partial runs are not the benchmark); "
          f"dropped < {min_dur:g} s: {len(dropped)}")
    print(f"{'model':<15}{'arm':<36}{'split':<6}{'n':>6}{'acc':>7}{'bal_acc':>9}{'macro_f1':>10}")
    for r in result["runs"]:
        print(f"{r['model']:<15}{r['arm']:<36}{r['split']:<6}{r['n']:>6}{r['accuracy']:>7.3f}"
              f"{r['balanced_accuracy']:>9.3f}{r['macro_f1']:>10.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

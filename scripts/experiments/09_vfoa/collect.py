"""All of experiment 09's arms in one table (held-out, own labeler, mean over folds), from outputs/experiments/09_vfoa/arms.log.
Writes outputs/experiments/09_vfoa/all_arms.csv. Run: PYTHONPATH=. python scripts/experiments/09_vfoa/collect.py"""
import json
import re
from pathlib import Path

import pandas as pd

from scripts.reporting.compare_results import load, resolve

COLS = ("auc", "spearman", "mae", "mae_levels", "rmse", "pred_mean", "pred_sd", "label_sd", "sd_ratio",
        "share_pred_3", "macro_f1", "accuracy", "balanced_acc")
arms = {}
for line in Path("outputs/experiments/09_vfoa/arms.log").read_text(encoding="utf-8").splitlines():
    m = re.match(r"^([a-z0-9]+_c[0-9p]+_\S+) ([0-9a-f]{12})\b", line)
    if m:
        arms[m.group(1)] = m.group(2)
rows = []
for name, h in arms.items():
    model, comp, inp, *rest = name.split("_")
    r = load(resolve(h, "outputs/results"))
    ev = r["evaluation"]
    for track in ("experts", "self"):
        for split, key in (("test", f"{track}/configured/classified/own_labeler"),
                           ("train", f"{track}/configured/train/own_labeler")):
            s = ev[key]["summary"]
            row = {"arm": name, "hash": h, "model": model, "comparison": comp, "input": int(inp),
                   "ctx": name.endswith("_ctx"), "track": track, "split": split}
            row.update({c: s[c]["mean"] if c in s else float("nan") for c in COLS})
            rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("outputs/experiments/09_vfoa/all_arms.csv", index=False)
print(len(arms), "arms;", df.groupby(["model", "comparison"]).arm.nunique().to_dict())

"""Experiment 09's result tables (markdown), from outputs/experiments/09_vfoa/all_arms.csv (run collect.py first).
Held-out (test) group, own labeler, mean over the three folds. Run: PYTHONPATH=. python scripts/experiments/09_vfoa/tables.py"""
import json

import pandas as pd

d = pd.read_csv("outputs/experiments/09_vfoa/all_arms.csv")
t = d[d.split == "test"]
CEIL = {s: json.load(open(f"outputs/labels/{f}"))["ceiling"]["subsets"]
        for s, f in (("1-5", "agreement.json"), ("lmh", "agreement_lmh.json"))}
MODELS = {"lr": "LR", "rf": "RF", "knn": "k-NN 500", "knn5": "k-NN 5"}
INPUTS = {1: "1. 08's inputs", 2: "2. + head pose", 3: "3. head pose alone", 4: "4. + class-relative"}


def ceil(scale, track, col):
    p = CEIL[scale][f"entry/{track}"]["predictors"]
    return [p[f"{w} of other experts, step"][col] for w in ("mean", "median")]


def f3(x):
    return f"{x:.3f}"


def c1_table(track):
    x = t[(t.comparison == "c1") & (t.track == track)]
    out = ["| Input | Context | " + " | ".join(MODELS.values()) + " |", "|---|---|" + "---|" * len(MODELS)]
    for i in INPUTS:
        for ctx in (False, True):
            cells = []
            for m in MODELS:
                r = x[(x.model == m) & (x.input == i) & (x.ctx == ctx)].iloc[0]
                cells.append(f"{f3(r.auc)} / {f3(r.spearman)}")
            out.append(f"| {INPUTS[i] if not ctx else ''} | {'on' if ctx else 'off'} | " + " | ".join(cells) + " |")
    a, s = ceil("1-5", track, "auc"), ceil("1-5", track, "spearman")
    out.append(f"| *Human ceiling (mean / median of experts)* | | AUC {f3(a[0])} / {f3(a[1])}, Spearman {f3(s[0])} / {f3(s[1])} | | | |")
    return "\n".join(out)


STRATS = (("c1", "Comparison 1 (reference)"), ("c2", "2. Scale shrinking"),
          ("c3p26", "3. Filtering, p = 0.26"), ("c3p50", "3. Filtering, p = 0.5"))


def strat_table(track, models=("lr", "rf", "knn", "knn5"), names=MODELS, inp=2, ctx=False):
    """One input arm: every strategy x model, ranking and spread."""
    x = t[(t.track == track) & (t.input == inp) & (t.ctx == ctx)]
    out = ["| Model | Strategy | AUC | Spearman | Pred. SD / label SD | Share predicted mid | Macro F1 | MAE |",
           "|---|---|---|---|---|---|---|---|"]
    for m in models:
        for c, cname in STRATS:
            r = x[(x.model == m) & (x.comparison == c)]
            if r.empty:
                continue
            r = r.iloc[0]
            mae = f"{f3(r.mae)} ({f3(r.mae_levels)} levels)" if c == "c2" else f3(r.mae)
            out.append(f"| {names[m] if c == 'c1' else ''} | {cname} | {f3(r.auc)} | {f3(r.spearman)} | "
                       f"{r.sd_ratio:.2f} | {r.share_pred_3:.2f} | {f3(r.macro_f1)} | {mae} |")
    return "\n".join(out)


def knn_table(track):
    rows = (("knn5", "Classifier, k = 5"), ("knnr5", "Regression, k = 5"), ("knnck50", "Classifier, k = 50"),
            ("knnrk50", "Regression, k = 50"), ("knn", "Classifier, k = 500"), ("lr", "*LR, for reference*"))
    x = t[(t.track == track) & (t.comparison == "c1")]
    out = ["| k-NN | AUC | Spearman | MAE | Pred. SD / label SD | Share predicted 3 |", "|---|---|---|---|---|---|"]
    for m, name in rows:
        g = x[(x.model == m) & x.input.isin([1, 2, 4])]          # the arms with 08's inputs, mean over 6
        out.append(f"| {name} | {f3(g.auc.mean())} | {f3(g.spearman.mean())} | {f3(g.mae.mean())} | "
                   f"{g.sd_ratio.mean():.2f} | {g.share_pred_3.mean():.2f} |")
    return "\n".join(out)


if __name__ == "__main__":
    for tr in ("experts", "self"):
        print(f"\n## C1 {tr}\n"); print(c1_table(tr))
        print(f"\n## strategies {tr}\n"); print(strat_table(tr))
        print(f"\n## knn {tr}\n"); print(knn_table(tr))
    for tr in ("experts", "self"):
        print(tr, "lmh ceiling mae_levels", [round(v/2, 3) for v in ceil("lmh", tr, "mae")], "macro_f1", ceil("lmh", tr, "macro_f1"))


def refs(track):
    """Reference rows for the strategy tables: constant 3 / MID and the human ceiling, on both scales."""
    from scripts.reporting.compare_results import load, resolve
    out = []
    for scale, h, label in (("1-5", "b1c2d485d3db", "1–5"), ("lmh", "5190274f3a29", "LOW / MID / HIGH")):
        s = load(resolve(h, "outputs/results"))["evaluation"][f"{track}/constant_3"]["summary"]
        lv = f" ({f3(s['mae_levels']['mean'])} levels)" if "mae_levels" in s else ""
        out.append(f"| *Constant {'3' if scale == '1-5' else 'MID'}* | *{label}* | 0.500 | – | 0 | 1.00 | "
                   f"{f3(s['macro_f1']['mean'])} | {f3(s['mae']['mean'])}{lv} |")
        p = CEIL[scale][f"entry/{track}"]["predictors"]
        for w in ("mean", "median"):
            q = p[f"{w} of other experts, step"]
            lv = f" ({f3(q['mae'] / 2)} levels)" if scale == "lmh" else ""
            out.append(f"| *Human ceiling, {w} of experts* | *{label}* | {f3(q['auc'])} | {f3(q['spearman'])} | – | – | "
                       f"{f3(q['macro_f1'])} | {f3(q['mae'])}{lv} |")
    return "\n".join(out)


def c1_ref(track):
    from scripts.reporting.compare_results import load, resolve
    s = load(resolve("b1c2d485d3db", "outputs/results"))["evaluation"][f"{track}/constant_3"]["summary"]
    p = CEIL["1-5"][f"entry/{track}"]["predictors"]
    m, md = (p[f"{w} of other experts, step"] for w in ("mean", "median"))
    return (f"MAE: constant 3 {f3(s['mae']['mean'])}; human ceiling {f3(m['mae'])} (mean of experts) / "
            f"{f3(md['mae'])} (median)")


def mae_line(track, inp=2, ctx=False):
    x = t[(t.track == track) & (t.comparison == "c1") & (t.input == inp) & (t.ctx == ctx)]
    return ", ".join(f"{MODELS[m]} {f3(x[x.model == m].mae.iloc[0])}" for m in MODELS)

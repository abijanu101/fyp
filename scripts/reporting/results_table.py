"""The standard result tables of per-labeler runs (markdown), as used in docs/experiments/ from 06 on.

  python -m scripts.reporting.results_table single=c5341cb6385e pooled=fdb140e9477e rf=5df1987449dd
         [--track experts] [--variant configured] [--results outputs/results] [--by-labeler]

One table: predictors as columns (the runs, then constant 3 and, unless --no-train-mean, the
training mean), rows under two bold headings. Every data cell is "train / test":
  Profile: mean, median and SD of the predictions, and of the labels that column is scored on.
  Metrics: MAE, RMSE, AUC*, Spearman, macro F1**, accuracy**.
    *  uses a threshold: AUC ranks the raw prediction against labels split at tau_att (engaged =
       rating >= 3), so it depends on where the labels are cut.
    ** uses discretisation: the prediction is rounded to a whole rating (halves toward 3) first.
  Unmarked metrics use the raw, continuous prediction.

Test (held-out) = each fold's model on its held-out group (classified context, own labeler); train = the
same model on its own training entries (originals only). Values are means over the held-out groups.
Constant 3 and the training mean follow, scored the same way.

--by-labeler: per run, the same two tables for each labeler's own entries (perspectivism: every
labeler scored separately), then the macro mean over labelers and the pooled score over all entries.
"""

from __future__ import annotations

import argparse

import numpy as np

from scripts.reporting.compare_results import load, resolve

PROFILE = (("pred_mean", "Mean"), ("pred_median", "Median"), ("pred_sd", "SD"))
LABEL_PROFILE = ("label_mean", "label_median", "label_sd")
METRICS = (("mae", "MAE"), ("rmse", "RMSE"), ("auc", "AUC*"), ("spearman", "Spearman"),
           ("macro_f1", "Macro F1**"), ("accuracy", "Accuracy**"))
LEGEND = ("\\* threshold (AUC: labels split at rating >= 3); \\*\\* discretisation (prediction rounded "
          "to a whole rating, halves toward 3); unmarked: the raw prediction.")


def value(ev: dict, key: str, col: str, labs: list[str] | None = None) -> float:
    """A metric's mean over folds; with `labs`, the macro mean over those labelers' keys."""
    if labs is None:
        s = ev.get(key, {}).get("summary", {})
        return s[col]["mean"] if col in s else np.nan
    xs = [value(ev, f"{key}/by_labeler/{l}", col) for l in labs]
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else np.nan


def fmt(x: float) -> str:
    return f"{x:.3f}" if np.isfinite(x) else "-"


def tables(cols: list[tuple[str, dict, str, str, list[str] | None]], label_ref: tuple | None = None,
           ceilings: list[tuple[str, dict]] | None = None,
           references: list[tuple[str, dict]] | None = None) -> None:
    """Two tables, one row per predictor (the runs, then the constants), every data cell "train / test".
    cols: (row name, evaluation, held-out key, train key, macro labelers or None).
      Profile: mean, median and SD of each predictor's predictions. `references` adds rows with a
        single value per cell (the self-report ratings, the median of the expert ratings). When the
        rows are scored on different entries (per labeler), each statistic gets a predicted and a
        labels column instead.
      Metrics: MAE, RMSE, AUC*, Spearman, macro F1**, accuracy**; `ceilings` adds rows with a single
        value per cell (the other experts' mean and median predicting the same labels).
    `label_ref` is accepted for compatibility and unused."""
    def cell(c: str, ev, hk, tk, labs) -> str:                 # every data cell: train / test
        return f"{fmt(value(ev, tk, c, labs))} / {fmt(value(ev, hk, c, labs))}"

    shared = len({tuple(cell(lc, *col[1:]) for lc in LABEL_PROFILE) for col in cols}) == 1
    print("**Profile** (train / test)\n")
    if shared:
        print("| | " + " | ".join(n for _, n in PROFILE) + " |\n|---|" + "---|" * len(PROFILE))
        for name, *rest in cols:
            print(f"| {name} | " + " | ".join(cell(c, *rest) for c, _ in PROFILE) + " |")
        for name, m in references or []:
            print(f"| *{name}* | " + " | ".join(fmt(m.get(k, np.nan)) for k in ("mean", "median", "sd")) + " |")
    else:
        heads = [f"{n} ({w})" for n in (n for _, n in PROFILE) for w in ("predicted", "labels")]
        print("| | " + " | ".join(heads) + " |\n|---|" + "---|" * len(heads))
        for name, *rest in cols:
            print(f"| {name} | " + " | ".join(cell(k, *rest) for c, lc in zip((c for c, _ in PROFILE), LABEL_PROFILE)
                                               for k in (c, lc)) + " |")
    print("\n**Metrics** (train / test)\n")
    print("| | " + " | ".join(n for _, n in METRICS) + " |\n|---|" + "---|" * len(METRICS))
    for name, *rest in cols:
        print(f"| {name} | " + " | ".join(cell(c, *rest) for c, _ in METRICS) + " |")
    for name, m in ceilings or []:
        print(f"| *{name}* | " + " | ".join(fmt(m.get(c, np.nan)) for c, _ in METRICS) + " |")
    note = ""
    if references or ceilings:
        note = (" Rows in italics are references with one value per cell, not train / test, on all entries of the"
                " track (`outputs/labels/agreement.json`, experiment 06 Part 1): the self-report ratings and the"
                " median of the expert ratings (profile); the other experts' mean and median predicting the same"
                " labels (metrics).")
    print(f"\n{LEGEND} Every cell: training groups / held-out (test) group.{note}")


CEILING_PREDICTORS = (("Mean of experts", "mean of other experts, step"),
                      ("Median of experts", "median of other experts, step"))


def human_ceilings(track: str, path: str = "outputs/labels/agreement.json") -> list[tuple[str, dict]]:
    """The human-ceiling references for a track: the other experts' mean and median predicting every
    entry, scored like the model (scripts.labels.label_agreement), on all entries of the track."""
    import json
    from pathlib import Path
    f = Path(path)
    if not f.exists():
        return []
    sub = json.loads(f.read_text(encoding="utf-8"))["ceiling"]["subsets"].get(f"entry/{track}", {})
    preds = sub.get("predictors", {})
    return [(head, preds[key]) for head, key in CEILING_PREDICTORS if key in preds]


def label_references(track: str, path: str = "outputs/labels/agreement.json") -> list[tuple[str, dict]]:
    """Profile references: the self-report ratings (all self entries) and the median of the expert
    ratings at this track's entries (the other experts' median curve), each as mean / median / SD."""
    import json
    from pathlib import Path
    f = Path(path)
    if not f.exists():
        return []
    subs = json.loads(f.read_text(encoding="utf-8"))["ceiling"]["subsets"]
    out = []
    med = subs.get("entry/self", {}).get("predictors", {}).get("median of other experts, step", {})
    if "label_mean" in med:
        out.append(("Self-report labels", {k: med[f"label_{k}"] for k in ("mean", "median", "sd")}))
    med = subs.get(f"entry/{track}", {}).get("predictors", {}).get("median of other experts, step", {})
    if "pred_mean" in med:
        out.append(("Median of expert labels", {k: med[f"pred_{k}"] for k in ("mean", "median", "sd")}))
    return out


def constants(ev: dict, t: str, labs=None, suffix: str = "", train_mean: bool = True) -> list:
    bases = (("constant_3", "Constant 3"),) + ((("train_mean", "Training mean"),) if train_mean else ())
    return [(label, ev, f"{t}/{base}{suffix}", f"{t}/{base}/train{suffix}", labs) for base, label in bases]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="name=<results file or config-hash prefix>")
    ap.add_argument("--track", default="experts", choices=["experts", "self"])
    ap.add_argument("--variant", default="configured", choices=["configured", "shared_expert"])
    ap.add_argument("--results", default="outputs/results")
    ap.add_argument("--no-train-mean", action="store_true", help="leave out the training-mean column")
    ap.add_argument("--no-ceiling", action="store_true", help="leave out the human-ceiling columns")
    ap.add_argument("--by-labeler", action="store_true",
                    help="per run: each labeler's entries, then the macro and pooled aggregates")
    args = ap.parse_args()
    runs = {}
    for item in args.runs:
        name, _, ref = item.partition("=")
        runs[name] = load(resolve(ref or name, args.results))
    t, v = args.track, args.variant
    held = f"{t}/{v}/classified/own_labeler"
    train = f"{t}/{v}/train/own_labeler"
    if not args.by_labeler:
        rows = [(f"{n} (`{r['config_hash'][:12]}`)", r["evaluation"], held, train, None) for n, r in runs.items()]
        tables(rows + constants(next(iter(runs.values()))["evaluation"], t, train_mean=not args.no_train_mean),
               ceilings=[] if args.no_ceiling else human_ceilings(t),
               references=[] if args.no_ceiling else label_references(t))
        return
    for name, r in runs.items():
        ev = r["evaluation"]
        labs = sorted({k.rsplit("/", 1)[1] for k in ev if k.startswith(held + "/by_labeler/")})
        if not labs:
            print(f"\n{name}: no per-labeler scores stored (results written before 18 Nov 2026; retrain)")
            continue
        for pred, h, tr in (("model", held, train), ("constant 3", f"{t}/constant_3", f"{t}/constant_3/train")):
            print(f"\n### {name} (`{r['config_hash'][:12]}`), {t} entries by labeler: {pred}\n")
            cols = [(lab, ev, f"{h}/by_labeler/{lab}", f"{tr}/by_labeler/{lab}", None) for lab in labs]
            cols += [("**macro**", ev, h, tr, labs), ("**pooled**", ev, h, tr, None)]
            tables(cols, (ev, held, train, None))


if __name__ == "__main__":
    main()

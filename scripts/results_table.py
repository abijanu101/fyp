"""The standard result tables of per-labeler runs (markdown), as used in docs/experiments/ from 06 on.

  python -m scripts.results_table single=c5341cb6385e pooled=fdb140e9477e rf=5df1987449dd
         [--track experts] [--variant configured] [--results outputs/results] [--by-labeler]

One table: predictors as columns (the runs, then constant 3 and the training mean, then the labels'
own profile), metrics as rows under four bold heading rows (profile held-out / train, metrics
held-out / train):
  Prediction profile: mean, median and SD of the predictions.
  Metrics: MAE, RMSE, AUC*, Spearman, macro F1**, accuracy**.
    *  uses a threshold: AUC ranks the raw prediction against labels split at tau_att (engaged =
       rating >= 3), so it depends on where the labels are cut.
    ** uses discretisation: the prediction is rounded to a whole rating (halves toward 3) first.
  Unmarked metrics use the raw, continuous prediction.

Held-out = each fold's model on its held-out group (classified context, own labeler); train = the
same model on its own training entries (originals only). Values are means over the held-out groups.
Constant 3 and the training mean follow, scored the same way.

--by-labeler: per run, the same two tables for each labeler's own entries (perspectivism: every
labeler scored separately), then the macro mean over labelers and the pooled score over all entries.
"""

from __future__ import annotations

import argparse

import numpy as np

from scripts.compare_results import load, resolve

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


def tables(cols: list[tuple[str, dict, str, str, list[str] | None]], label_ref: tuple | None = None) -> None:
    """One table, metrics as rows and predictors as columns. cols: (header, evaluation, held-out key,
    train key, macro labelers or None). A last "Labels" column gives the labels' own profile, taken
    from `label_ref` (a column tuple without the header) or else from the first column. Bold heading
    rows split the table: prediction profile held-out / train, then metrics held-out / train."""
    ref = label_ref or cols[0][1:]
    print("| | " + " | ".join(h for h, *_ in cols) + " | Labels |")
    print("|---|" + "---|" * (len(cols) + 1))
    blank = " | ".join("" for _ in range(len(cols) + 1))
    for section, rows in (("Prediction profile", PROFILE), ("Metrics", METRICS)):
        for split, i in (("held-out", 1), ("train", 2)):
            print(f"| **{section}, {split}** | {blank} |")
            for c, name in rows:
                vals = [fmt(value(ev, hk if i == 1 else tk, c, labs)) for _, ev, hk, tk, labs in cols]
                if section == "Prediction profile":
                    ev, hk, tk, labs = ref
                    lab = fmt(value(ev, hk if i == 1 else tk, LABEL_PROFILE[[p for p, _ in PROFILE].index(c)], labs))
                else:
                    lab = ""
                print(f"| {name} | " + " | ".join(vals) + f" | {lab} |")
    print(f"\n{LEGEND}")


def constants(ev: dict, t: str, labs=None, suffix: str = "") -> list:
    return [(label, ev, f"{t}/{base}{suffix}", f"{t}/{base}/train{suffix}", labs)
            for base, label in (("constant_3", "Constant 3"), ("train_mean", "Training mean"))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="name=<results file or config-hash prefix>")
    ap.add_argument("--track", default="experts", choices=["experts", "self"])
    ap.add_argument("--variant", default="configured", choices=["configured", "shared_expert"])
    ap.add_argument("--results", default="outputs/results")
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
        tables(rows + constants(next(iter(runs.values()))["evaluation"], t))
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

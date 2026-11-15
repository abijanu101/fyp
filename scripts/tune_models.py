"""Tune SPU gate and engagement-expert hyperparameters with nested leave-one-group-out.

  python -m scripts.tune_models [--component gate|expert|both] [--trials N] [--set key=value ...]

Objectives:  gate   -> 6-way macro-F1 of the configured gate model
             expert -> window AUC of the configured mixture under the ORACLE context, which isolates
                       expert quality from gate errors (design §8 oracle condition)
Uses features cached by `run_pipeline extract` and labels from the current `labels.*` config, so
fix the label thresholds (scripts.dipser.inspect_labels) before tuning experts.

Writes <output_dir>/tuning/<component>_<model>.json with the selected params (config patch), their
plain LOGO score, and the nested-LOGO estimate (the honest number, since selection never saw the
outer test group). With 3 groups the inner loop has 2 folds of 1 training group each, so the inner
ranking is noisy; treat small differences between trials as ties.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from scripts.run_pipeline import parse_sets
from scripts.train_models import engagement_rows, load_table, mixture_factory
from src.core.config import load_config
from src.core.pipeline import SCENE_PREFIX, VEC_PREFIX
from src.evaluation import binary_metrics, context_metrics
from src.experts import fit_engagement, predict_windows
from src.spu import GATES, context_probs
from src.tuning import nested_logo_search


def gate_scorer(cfg, Xg, exp):
    name, classes = cfg["gate"]["name"], cfg["classes"]

    def score(params, tr, te):
        gate = GATES.build(name, classes=classes, seed=cfg["seed"], **params).fit(Xg[tr], exp[tr])
        pred = np.array(classes)[gate.predict_proba(Xg[te]).argmax(1)]
        return context_metrics(exp[te], pred, classes)["macro_f1"]
    return score


def expert_scorer(cfg, rows, y, exp, labeled):
    """Window AUC under oracle context; rows are early (per window) or late (per student) per config."""
    classes, mcfg = cfg["classes"], cfg["mixture"]

    def score(params, tr, te):
        c = copy.deepcopy(cfg)
        c["expert"]["params"] = params
        mix, fill = fit_engagement(c["engagement"], mixture_factory(c, rows, None, mcfg["expert_scope"], mcfg["mode"]),
                                   rows, tr)
        te = te[labeled[te]]
        p = predict_windows(mix, rows, te, context_probs("oracle", classes, exp[te]), fill)
        return binary_metrics(y[te], p, mcfg["tau"])["auc"]
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--component", choices=["gate", "expert", "both"], default="both")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    tune = cfg["tuning"]

    df = load_table(cfg)
    vec_cols = sorted(c for c in df if c.startswith(VEC_PREFIX))
    gate_cols = sorted(c for c in df if c.startswith(SCENE_PREFIX)) + (vec_cols if cfg["gate_uses_vec"] else [])
    groups, exp = df["group"].to_numpy(), df["exp"].to_numpy()
    labeled = df["y"].notna().to_numpy()
    y = df["y"].fillna(0).to_numpy(int)
    if args.component in ("expert", "both") and len(np.unique(y[labeled])) < 2:
        raise SystemExit("Window labels have a single class under the current labels.* thresholds; "
                         "run scripts.dipser.inspect_labels and set labels.tau_att / labels.tau_share first.")

    jobs = []
    if args.component in ("gate", "both"):
        jobs.append(("gate", cfg["gate"], gate_scorer(cfg, df[gate_cols].to_numpy(float), exp)))
    if args.component in ("expert", "both"):
        jobs.append(("expert", cfg["expert"],
                     expert_scorer(cfg, engagement_rows(cfg, df, vec_cols), y, exp, labeled)))

    out = Path(cfg["output_dir"]) / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    for component, model_cfg, score in jobs:
        name = model_cfg["name"]
        space = tune[component]["space"].get(name)
        if not space:
            print(f"{component}: no search space for '{name}', skipping")
            continue
        print(f"{component} ({name}): nested LOGO search, {args.trials or tune[component]['n_trials']} trials ...")
        res = nested_logo_search(groups, space, args.trials or tune[component]["n_trials"], tune["seed"],
                                 score, baseline=model_cfg.get("params", {}))
        res["config_patch"] = {f"{component}.params": res["selected"]}
        (out / f"{component}_{name}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
        print(f"  selected {res['selected']}  LOGO {res['selected_cv_score']:.3f}  "
              f"nested {res['nested_mean']:.3f}+/-{res['nested_std']:.3f}")


if __name__ == "__main__":
    main()

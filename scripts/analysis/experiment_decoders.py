"""Experiment 2: every classical decoder, trained on groups 1-2 clean or with augmented copies,
tested on group 3, clean and augmented (reported separately).

  python -m scripts.analysis.experiment_decoders --extract     # long (hours), resumable: features for every
                                                      # session, then augmented copies of groups 1-2
  python -m scripts.analysis.experiment_decoders               # evaluation from the caches (resumable)

Training sets (`--copies`, default 0 1 2): groups 1-2 clean, + 1 augmented copy, + 2 copies.
Splits (training never contains group 3 in any form):
  cv:group_01, cv:group_02   development estimate: train on the other dev group (+ its copies),
                             score its clean footage
  test:group_03              train on groups 1-2 (+ copies), score group 3 as recorded
  test_aug:group_03          the same models, scored on group 3's augmented copies (robustness)
Decoders:
  gates    random_forest, svm, logreg, knn_k1/k3/k5 (6-way activity from scene features)
  experts  logreg, random_forest, svm, knn_k1/k3/k5, hand_weighted; scope per_experiment or shared;
           inputs "student+context" (trk.* + Vec, the default) or "student" (trk.* only);
           soft mixture, scored under the true activity (oracle) and under each gate's prediction;
           labelers (config engagement.labelers): consensus labels, or an ensemble of one mixture per
           expert labeler on that labeler's own labels, averaged over the labelers who rated each student
Baselines: majority (training engaged rate) and "-flow" (minus class-mean residual flow as the curve).

Labels use the provisional thresholds tau_att=3, tau_share=0.6 (set below), late aggregation.
Output: <output_dir>/experiments/decoders/<hash>/{engagement.csv, gates.csv, summary.md}
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_pipeline import cmd_extract, parse_sets
from scripts.train_models import engagement_rows, expert_factory, load_table, with_augmented
from src.core.config import config_hash, load_config
from src.core.pipeline import SCENE_PREFIX, TRACK_PREFIX, VEC_PREFIX
from src.evaluation import binary_metrics, context_metrics
from src.experts import EXPERTS, EngagementRows, ExpertMixture, decision_threshold, fit_engagement, predict_windows
from src.spu import GATES, context_probs

DEV, TEST = ["group_01", "group_02"], "group_03"
EXPERIMENT = {"labels.tau_att": 3, "labels.tau_share": 0.6, "engagement.aggregate": "late",
              "engagement.late_context": True, "augmentation.copies": 2}
# decoder name -> (registered model, parameters). k-NN at k = 1, 3, 5 only: the feature space is
# sparse, so larger neighbourhoods mostly reach into other sessions.
GATES_TRIED = {"random_forest": ("random_forest", {"n_estimators": 300, "min_samples_leaf": 5, "n_jobs": -1}),
               "svm": ("svm", {"C": 1.0}),
               "logreg": ("logreg", {"C": 1.0}),
               **{f"knn_k{k}": ("knn", {"n_neighbors": k}) for k in (1, 3, 5)}}
EXPERTS_TRIED = {"logreg": ("logreg", {"C": 0.1}),
                 "random_forest": ("random_forest", {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20,
                                                     "n_jobs": -1}),
                 "svm": ("svm", {"C": 1.0, "max_train_rows": 4000}),       # SVC cost ~ rows^2: capped
                 **{f"knn_k{k}": ("knn", {"n_neighbors": k, "max_train_rows": 20000}) for k in (1, 3, 5)},
                 "hand_weighted": ("hand_weighted", None)}
# config engagement.labelers -> experts tried in that mode. "ensemble" = one mixture per expert
# labeler (self-ratings excluded via engagement.ensemble_exclude), averaged over the rating labelers;
# tried for the two cheapest strong models (each costs one fit per labeler).
LABELER_MODES = {"consensus": list(EXPERTS_TRIED), "ensemble": ["logreg", "random_forest"]}


def experiment_config(args) -> dict:
    cfg = load_config(args.config, {**EXPERIMENT, **parse_sets(args.set)})
    return cfg


def with_copies(cfg: dict, n: int, groups: list[str] | None = None) -> dict:
    c = copy.deepcopy(cfg)
    c["augmentation"]["copies"] = n
    if groups is not None:
        c["dataset"]["groups"] = groups
    return c


def extract(cfg: dict) -> None:
    """Two resumable phases; rerun the same command after an interruption."""
    print("== phase 1/2: features of every in-scope session, clean", flush=True)
    cmd_extract(with_copies(cfg, 0), None, reuse=True)
    n = cfg["augmentation"]["copies"]
    aug = with_copies(cfg, n, DEV + [TEST])       # dev copies train; group-3 copies are a second test set
    print(f"== phase 2/2: features of {n} augmented copies of every {'/'.join(DEV + [TEST])} session "
          f"(augmented on the fly)", flush=True)
    cmd_extract(aug, None, reuse=True)


def make_mixture(cfg, rows, expert, scope):
    model, params = EXPERTS_TRIED[expert]
    if model == "hand_weighted":
        make = expert_factory(cfg, rows.cols, "hand_weighted")
    else:
        make = lambda exp_id: EXPERTS.build(model, seed=cfg["seed"], **params)
    return lambda: ExpertMixture(cfg["classes"], make, scope, "soft")


def student_only(rows: EngagementRows) -> EngagementRows:
    keep = [i for i, c in enumerate(rows.cols) if c.startswith(TRACK_PREFIX)]
    return EngagementRows(rows.X[:, keep], [rows.cols[i] for i in keep], rows.window, rows.exp, rows.y,
                          rows.y_by_labeler, rows.rated_by)


def splits(df: pd.DataFrame):
    """Training never contains group 3 in any form; dev-fold tests are clean footage only."""
    g, aug = df["group"].to_numpy(), df["aug"].to_numpy(bool)
    idx = np.arange(len(df))
    dev = np.isin(g, DEV)
    for held in DEV:
        yield f"cv:{held}", idx[dev & (g != held)], idx[(g == held) & ~aug]
    yield f"test:{TEST}", idx[dev], idx[(g == TEST) & ~aug]
    yield f"test_aug:{TEST}", idx[dev], idx[(g == TEST) & aug]


def evaluate(cfg: dict, copies_list: list[int]) -> Path:
    h = config_hash({**cfg, "augmentation": {**cfg["augmentation"], "copies": None}})[:12]
    out = Path(cfg["output_dir"]) / "experiments" / "decoders" / h
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    eng_path, gate_path = out / "engagement.csv", out / "gates.csv"
    done = set()
    if eng_path.exists():
        e = pd.read_csv(eng_path)
        done = set(zip(e.copies, e.split, e.labelers, e.inputs, e.expert, e.scope))
    classes = cfg["classes"]
    tau = decision_threshold("late", cfg["mixture"]["tau"], cfg["labels"]["tau_share"])
    base = load_table(cfg)                                          # originals, all groups in scope
    missing = sorted(set(DEV + [TEST]) - set(base["group"]))
    if missing:
        raise SystemExit(f"no features for {missing}; run with --extract first")

    n_all = cfg["augmentation"]["copies"]
    test = with_augmented(with_copies(cfg, n_all), base[base["group"] == TEST].reset_index(drop=True))
    for k in copies_list:
        ck = with_copies(cfg, k)
        dev = with_augmented(ck, base[base["group"].isin(DEV)].reset_index(drop=True))
        df = pd.concat([dev, test], ignore_index=True)      # test: group 3 clean + all its copies
        vec_cols = sorted(c for c in df if c.startswith(VEC_PREFIX))
        scene_cols = sorted(c for c in df if c.startswith(SCENE_PREFIX))
        Xg, exp, y = df[scene_cols].to_numpy(float), df["exp"].to_numpy(), df["y"].to_numpy(float)
        t0 = time.monotonic()
        full = engagement_rows(ck, df, vec_cols, per_labeler=True)  # late rows, Vec context, per-labeler labels
        inputs = {"student+context": full, "student": student_only(full)}
        print(f"\n== copies {k}: {len(df)} windows ({int(df['aug'].sum())} augmented), {len(full.y)} student rows "
              f"({time.monotonic() - t0:.0f}s to build)", flush=True)

        for split, tr, te in splits(df):
            lab = np.isfinite(y[te])
            gates = {}
            for name, (model, params) in GATES_TRIED.items():
                gate = GATES.build(model, classes=classes, seed=cfg["seed"], **params).fit(Xg[tr], exp[tr])
                gates[name] = gate.predict_proba(Xg[te])
                m = context_metrics(exp[te], np.array(classes)[gates[name].argmax(1)], classes)
                pd.DataFrame([{"copies": k, "split": split, "gate": name, "accuracy": m["accuracy"],
                               "macro_f1": m["macro_f1"]}]).to_csv(gate_path, mode="a", index=False,
                                                                   header=not gate_path.exists())
            contexts = {"oracle": context_probs("oracle", classes, exp[te]), **{f"gate:{n}": G for n, G in gates.items()}}
            records = []
            if (k, split, "-", "-", "majority", "-") not in done:
                trl = tr[np.isfinite(y[tr])]
                p = np.full(len(te), float(np.nanmean(y[trl])))
                flow = -df.iloc[te][[c for c in vec_cols if c.startswith("vec.rf.mag_") and c.endswith(".mean")]].mean(1).to_numpy()
                flow = np.nan_to_num(flow, nan=np.nanmedian(flow))
                for name, score in (("majority", p), ("-flow", flow)):
                    m = binary_metrics(y[te][lab].astype(int), score[lab], 0.5 if name == "majority" else np.inf)
                    records.append({"copies": k, "split": split, "labelers": "-", "inputs": "-", "expert": name,
                                    "scope": "-",
                                    "context": "-", "auc": m["auc"], "f1": m["f1"], "balanced_acc": m["balanced_acc"],
                                    "n": m["n"],
                                    "pos_rate": m["pos_rate"], "fit_s": 0.0})
            for labelers, experts in LABELER_MODES.items():          # config engagement.labelers
                cm = {**ck, "engagement": {**ck["engagement"], "labelers": labelers}}
                for inp, rows in inputs.items():
                    for expert in experts:
                        for scope in ("per_experiment", "shared"):
                            if (k, split, labelers, inp, expert, scope) in done:
                                continue
                            t1 = time.monotonic()
                            mix, fill = fit_engagement(cm["engagement"], make_mixture(cm, rows, expert, scope), rows, tr)
                            fit_s = time.monotonic() - t1
                            for ctx, G in contexts.items():
                                p = predict_windows(mix, rows, te, G, fill)
                                m = binary_metrics(y[te][lab].astype(int), p[lab], tau)
                                records.append({"copies": k, "split": split, "labelers": labelers, "inputs": inp,
                                                "expert": expert, "scope": scope, "context": ctx, "auc": m["auc"],
                                                "f1": m["f1"], "balanced_acc": m["balanced_acc"],
                                                "n": m["n"],
                                                "pos_rate": m["pos_rate"], "fit_s": fit_s})
                            pd.DataFrame(records).to_csv(eng_path, mode="a", index=False, header=not eng_path.exists())
                            records = []
                            print(f"  copies {k} {split:16s} {labelers:9s} {inp:16s} {expert:14s} {scope:15s} "
                                  f"fit {fit_s:5.1f}s", flush=True)
            if records:
                pd.DataFrame(records).to_csv(eng_path, mode="a", index=False, header=not eng_path.exists())
    summarize(out)
    return out


def summarize(out: Path) -> None:
    e = pd.read_csv(out / "engagement.csv").drop_duplicates(
        subset=["copies", "split", "labelers", "inputs", "expert", "scope", "context"], keep="last")
    g = pd.read_csv(out / "gates.csv").drop_duplicates(subset=["copies", "split", "gate"], keep="last")
    kind = lambda s: s.str.split(":").str[0]          # cv, test (group 3 clean), test_aug (group 3 augmented)
    e["set"], g["set"] = kind(e["split"]), kind(g["split"])
    lines = [f"# Experiment 2: decoders (`{out.name}`)", "",
             "Columns: set (cv = mean of the two development folds; test = group 3 as recorded; "
             "test_aug = group 3's augmented copies) and +k = augmented copies per training session. "
             "AUC needs no cut-off; read it first.", ""]

    def table(df, index, cols, value):
        """Markdown pivot: one row per `index` combination, one column per (set, copies)."""
        t = df.pivot_table(index=index, columns=cols, values=value, aggfunc="mean")
        head = "| " + " | ".join(t.index.names) + " | " + " | ".join(f"{s} +{c}" for s, c in t.columns) + " |"
        rows = [head, "|" + "---|" * (len(t.index.names) + len(t.columns))]
        for ix, r in t.iterrows():
            ix = ix if isinstance(ix, tuple) else (ix,)
            rows.append("| " + " | ".join(map(str, ix)) + " | " + " | ".join("" if pd.isna(v) else f"{v:.3f}" for v in r) + " |")
        return rows

    lines += ["## Gates (6-way accuracy, macro-F1)", ""]
    lines += table(g, ["gate"], ["set", "copies"], "accuracy") + [""]
    lines += table(g, ["gate"], ["set", "copies"], "macro_f1") + [""]
    for ctx in ["oracle", "gate:random_forest"]:
        sub = e[(e.context == ctx) | (e.context == "-")]
        for metric in ("auc",):
            lines += [f"## Engagement {metric}, context {ctx}", ""]
            lines += table(sub, ["labelers", "inputs", "expert", "scope"], ["set", "copies"], metric) + [""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsummary -> {out / 'summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--extract", action="store_true", help="run both extraction phases first (hours, resumable)")
    ap.add_argument("--copies", nargs="*", type=int, help="training sets to evaluate (default 0..augmentation.copies)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    cfg = experiment_config(args)
    if args.extract:
        extract(cfg)
    copies_list = args.copies if args.copies is not None else list(range(cfg["augmentation"]["copies"] + 1))
    if args.summary_only:
        h = config_hash({**cfg, "augmentation": {**cfg["augmentation"], "copies": None}})[:12]
        summarize(Path(cfg["output_dir"]) / "experiments" / "decoders" / h)
        return
    evaluate(cfg, copies_list)


if __name__ == "__main__":
    main()

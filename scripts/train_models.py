"""Train and evaluate the SPU gate and engagement experts under leave-one-group-out (design §7, §8, §11).

  python -m scripts.train_models [--config configs/deliverable3.json] [--set labels.tau_att=3 ...]

Reads features cached by `run_pipeline extract`, attaches window labels, and writes
  <output_dir>/results/<config hash>.json   every metric, tagged with the full config hash
  <output_dir>/models/<config hash>.joblib  gate + mixture fit on all groups, for `run_pipeline infer`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.run_pipeline import features_dir, parse_sets
from src.core.config import config_hash, load_config
from src.core.pipeline import SCENE_PREFIX, VEC_PREFIX
from src.core.types import SessionKey, Window
from src.datasets import DipserIndex
from src.evaluation import binary_metrics, context_metrics
from src.experts import EXPERTS, ExpertMixture, HandWeightedExpert
from src.labels import label_session
from src.spu import GATES, context_probs, logo_splits


def load_table(cfg: dict) -> pd.DataFrame:
    index = DipserIndex.from_config(cfg)
    parts = []
    for csv in sorted(features_dir(cfg).glob("*.csv")):
        key = SessionKey(*csv.stem.split("__"))
        df = pd.read_csv(csv)
        windows = [Window(a, b) for a, b in zip(df.t_start, df.t_end)]
        labels = label_session(index, key, windows, cfg["labels"])
        df["session"], df["group"], df["exp"] = str(key), key.group, key.exp_id
        df["y"] = [np.nan if l.engaged is None else l.engaged for l in labels]
        df["share"] = [l.share_engaged for l in labels]
        df["n_labeled"] = [l.n_labeled for l in labels]
        parts.append(df)
    if not parts:
        raise SystemExit(f"No cached features in {features_dir(cfg)}; run `run_pipeline extract` first.")
    return pd.concat(parts, ignore_index=True)


def expert_factory(cfg: dict, vec_cols: list[str], name: str | None = None):
    name = name or cfg["expert"]["name"]
    if name == "hand_weighted":
        return lambda exp_id: HandWeightedExpert(vec_cols, exp_id=exp_id, **cfg["hand_weighted"])
    return lambda exp_id: EXPERTS.build(name, seed=cfg["seed"], **cfg["expert"].get("params", {}))


def retention_check(df: pd.DataFrame, p: np.ndarray) -> float:
    """Mean per-session Pearson r between engagement_prob and share of labeled students engaged."""
    rs = []
    for _, idx in df.groupby("session").indices.items():
        a, b = p[idx], df["share"].to_numpy()[idx]
        if len(idx) > 2 and np.std(a) > 0 and np.nanstd(b) > 0:
            rs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(rs)) if rs else np.nan


def summarize(folds: list[dict]) -> dict:
    keys = [k for k in folds[0] if isinstance(folds[0][k], (int, float))]
    return {k: {"mean": float(np.nanmean([f[k] for f in folds])),
                "std": float(np.nanstd([f[k] for f in folds]))} for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    h, classes, mcfg = config_hash(cfg), cfg["classes"], cfg["mixture"]

    df = load_table(cfg)
    vec_cols = sorted(c for c in df if c.startswith(VEC_PREFIX))
    scene_cols = sorted(c for c in df if c.startswith(SCENE_PREFIX))
    gate_cols = scene_cols + (vec_cols if cfg["gate_uses_vec"] else [])
    Xv, Xg = df[vec_cols].to_numpy(float), df[gate_cols].to_numpy(float)
    exp, groups = df["exp"].to_numpy(), df["group"].to_numpy()
    labeled = df["y"].notna().to_numpy()
    y = df["y"].fillna(0).to_numpy(int)

    # Mixture variants: (name, expert name, scope, mode). The configured soft per-experiment one is the headline.
    variants = [("configured", None, mcfg["expert_scope"], mcfg["mode"]),
                ("hard_top1", None, mcfg["expert_scope"], "hard"),
                ("shared_expert", None, "shared", mcfg["mode"]),
                ("hand_weighted", "hand_weighted", "per_experiment", mcfg["mode"])]

    gate_folds, eng_folds = [], {}
    for g, tr, te in logo_splits(groups):
        gate = GATES.build(cfg["gate"]["name"], classes=classes, seed=cfg["seed"],
                           **cfg["gate"].get("params", {})).fit(Xg[tr], exp[tr])
        G = {"classified": gate.predict_proba(Xg[te]), "oracle": context_probs("oracle", classes, exp[te])}
        gm = context_metrics(exp[te], np.array(classes)[G["classified"].argmax(1)], classes)
        gate_folds.append({"held_out": g, "accuracy": gm["accuracy"], "macro_f1": gm["macro_f1"], "detail": gm})

        trl, tel = tr[labeled[tr]], labeled[te]
        if tel.sum() == 0 or len(trl) == 0:
            continue
        majority = float(y[trl].mean())
        eng_folds.setdefault("majority", []).append(
            {"held_out": g, **binary_metrics(y[te][tel], np.full(tel.sum(), majority), mcfg["tau"])})
        for name, expert_name, scope, mode in variants:
            mix = ExpertMixture(classes, expert_factory(cfg, vec_cols, expert_name), scope, mode)
            mix.fit(Xv[trl], y[trl], exp[trl])
            for ctx, Gc in G.items():
                p = mix.predict_proba(Xv[te], Gc)
                m = binary_metrics(y[te][tel], p[tel], mcfg["tau"])
                m["retention_r"] = retention_check(df.iloc[te], p)
                m["per_experiment"] = {c: binary_metrics(y[te][tel & (exp[te] == c)], p[tel & (exp[te] == c)],
                                                         mcfg["tau"]) for c in classes}
                eng_folds.setdefault(f"{name}/{ctx}", []).append({"held_out": g, **m})

    results = {
        "config_hash": h,
        "config": cfg,
        "data": {"n_sessions": int(df["session"].nunique()), "n_windows": int(len(df)),
                 "label_coverage": float(labeled.mean()),
                 "pos_rate": float(y[labeled].mean()) if labeled.any() else None,
                 "pos_rate_by_exp": {c: float(y[labeled & (exp == c)].mean()) if (labeled & (exp == c)).any()
                                     else None for c in classes}},
        "gate": {"summary": summarize(gate_folds), "folds": gate_folds},
        "engagement": {k: {"summary": summarize(v), "folds": v} for k, v in eng_folds.items()},
    }
    out = Path(cfg["output_dir"])
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / f"{h}.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    # Final models on all groups, for inference.
    gate = GATES.build(cfg["gate"]["name"], classes=classes, seed=cfg["seed"],
                       **cfg["gate"].get("params", {})).fit(Xg, exp)
    mix = ExpertMixture(classes, expert_factory(cfg, vec_cols), mcfg["expert_scope"], mcfg["mode"])
    mix.fit(Xv[labeled], y[labeled], exp[labeled])
    (out / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump({"config_hash": h, "gate": gate, "mixture": mix, "vec_cols": vec_cols, "gate_cols": gate_cols},
                out / "models" / f"{h}.joblib")

    print(f"config {h[:12]}  windows={len(df)}  label coverage={labeled.mean():.2f}")
    print("gate (classified) accuracy:", results["gate"]["summary"]["accuracy"])
    for k, v in results["engagement"].items():
        s = v["summary"]
        print(f"  {k:32s} F1 {s['f1']['mean']:.3f}+/-{s['f1']['std']:.3f}  AUC {s['auc']['mean']:.3f}  "
              f"bAcc {s['balanced_acc']['mean']:.3f}")


if __name__ == "__main__":
    main()

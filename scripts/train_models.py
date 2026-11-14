"""Train and evaluate the SPU gate and engagement experts under leave-one-group-out (design §7, §8, §11).

  python -m scripts.train_models [--config configs/deliverable3.json] [--set labels.tau_att=3 ...]

Reads features cached by `run_pipeline extract`, attaches labels, and writes
  <output_dir>/results/<config hash>.json   every metric, tagged with the full config hash
  <output_dir>/models/<config hash>.joblib  gate + mixture fit on all groups, for `run_pipeline infer`

Aggregation (config engagement.aggregate, see src/experts/levels.py):
  early  the experts see one row per window (Vec) and are trained on window labels;
  late   the experts see one row per (window, visible student): that student's features (+ the
         window's Vec if engagement.late_context) and are trained on that student's own label.
         Tracks are linked to students with SID-TID association (training only, design §4).
Either way the window score is the mean of its rows' probabilities, and every metric is computed
on window labels, so the two modes are directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.run_pipeline import features_dir, parse_sets, session_stem, tracks_dir
from src.core.config import config_hash, load_config
from src.core.pipeline import SCENE_PREFIX, VEC_PREFIX
from src.core.store import load_track
from src.core.types import SessionKey, Window
from src.datasets import DipserIndex
from src.evaluation import binary_metrics, context_metrics
from src.experts import (EXPERTS, EngagementRows, ExpertMixture, HandWeightedExpert, decision_threshold,
                         early_rows, fit_rows, late_rows, predict_windows, student_key)
from src.labels import label_session, student_labels_session
from src.spu import GATES, context_probs, logo_splits
from src.tracking import associate_sids


def window_csvs(cfg: dict) -> list[Path]:
    """The cached window CSVs of the sessions in scope (config `dataset`). The feature folder may
    hold more (it is shared by every config with the same feature hash), and those are ignored."""
    scope = {session_stem(k) for k in DipserIndex.from_config(cfg).sessions()}
    return sorted(p for p in features_dir(cfg).glob("*.csv")
                  if not p.name.endswith(".tracks.csv") and p.stem in scope)


def load_table(cfg: dict) -> pd.DataFrame:
    """One row per window over all cached sessions, with window labels attached."""
    index = DipserIndex.from_config(cfg)
    parts = []
    for csv in window_csvs(cfg):
        key = SessionKey(*csv.stem.split("__"))
        df = pd.read_csv(csv)
        windows = [Window(a, b) for a, b in zip(df.t_start, df.t_end)]
        labels = label_session(index, key, windows, cfg["labels"])
        df["session"], df["group"], df["exp"] = str(key), key.group, key.exp_id
        df["window_local"] = np.arange(len(df))
        df["y"] = [np.nan if l.engaged is None else l.engaged for l in labels]
        df["share"] = [l.share_engaged for l in labels]
        df["n_labeled"] = [l.n_labeled for l in labels]
        parts.append(df)
    if not parts:
        raise SystemExit(f"No cached features in {features_dir(cfg)}; run `run_pipeline extract` first.")
    return pd.concat(parts, ignore_index=True)


def engagement_rows(cfg: dict, df: pd.DataFrame, vec_cols: list[str]) -> EngagementRows:
    """The rows the experts see, per config engagement.aggregate."""
    mode = cfg["engagement"]["aggregate"]
    if mode == "early":
        X, cols, window = early_rows(df, vec_cols)
        return EngagementRows(X, cols, window, df["exp"].to_numpy(), df["y"].to_numpy(float))
    if mode != "late":
        raise SystemExit(f"engagement.aggregate must be 'early' or 'late', got {mode!r}")

    index = DipserIndex.from_config(cfg)
    assoc = cfg["association"]
    tables, ys = [], []
    for session, sdf in df.groupby("session", sort=False):
        key = SessionKey(*session.split("/"))
        stem = session_stem(key)
        tracks_csv = features_dir(cfg) / f"{stem}.tracks.csv"
        track_npz = tracks_dir(cfg) / f"{stem}.npz"
        if not tracks_csv.exists() or not track_npz.exists():
            raise SystemExit(f"{session}: per-student rows or frame-pass cache missing; "
                             f"run `run_pipeline extract --reuse` (recomputes them from cached tracks)")
        t = pd.read_csv(tracks_csv)
        tid_to_sid = associate_sids(load_track(track_npz).tracklets, index.annotations(key),
                                    assoc["min_iou"], assoc.get("min_match_share", 0.5))
        windows = [Window(a, b) for a, b in zip(sdf.t_start, sdf.t_end)]
        labels = student_labels_session(index, key, windows, cfg["labels"])
        local = t["window"].to_numpy(int)
        sids = [tid_to_sid.get(int(tid)) for tid in t["tid"]]
        ys.append(np.array([labels[s][w] if s in labels else np.nan for s, w in zip(sids, local)], float))
        t["window"] = sdf.index.to_numpy()[local]                   # local window -> global row of df
        tables.append(t)
    tracks = pd.concat(tables, ignore_index=True)
    X, cols, window = late_rows(df, tracks, context=cfg["engagement"].get("late_context", True))
    return EngagementRows(X, cols, window, df["exp"].to_numpy()[window], np.concatenate(ys))


def expert_factory(cfg: dict, feature_names: list[str], name: str | None = None):
    """make_expert(exp_id) for ExpertMixture; feature_names are the row columns (hand_weighted reads by name)."""
    name = name or cfg["expert"]["name"]
    if name == "hand_weighted":
        hw = dict(cfg["hand_weighted"])
        if cfg["engagement"]["aggregate"] == "late":     # weight each student's own feature, not the class mean
            hw["weights"] = {student_key(k): v for k, v in hw["weights"].items()}
            hw["phone_key"] = student_key(hw["phone_key"]) if hw.get("phone_key") else None
        return lambda exp_id: HandWeightedExpert(feature_names, exp_id=exp_id, **hw)
    return lambda exp_id: EXPERTS.build(name, seed=cfg["seed"], **cfg["expert"].get("params", {}))


def mixture_factory(cfg: dict, rows: EngagementRows, expert_name: str | None, scope: str, mode: str):
    return lambda: ExpertMixture(cfg["classes"], expert_factory(cfg, rows.cols, expert_name), scope, mode)


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
    tau = decision_threshold(cfg["engagement"]["aggregate"], mcfg["tau"], cfg["labels"]["tau_share"])

    df = load_table(cfg)
    vec_cols = sorted(c for c in df if c.startswith(VEC_PREFIX))
    scene_cols = sorted(c for c in df if c.startswith(SCENE_PREFIX))
    gate_cols = scene_cols + (vec_cols if cfg["gate_uses_vec"] else [])
    Xg = df[gate_cols].to_numpy(float)
    exp, groups = df["exp"].to_numpy(), df["group"].to_numpy()
    labeled = df["y"].notna().to_numpy()
    y = df["y"].fillna(0).to_numpy(int)
    rows = engagement_rows(cfg, df, vec_cols)

    # Mixture variants: (name, expert name, scope, mode). The configured one is the headline.
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
            {"held_out": g, **binary_metrics(y[te][tel], np.full(tel.sum(), majority), 0.5)})
        for name, expert_name, scope, mode in variants:
            mix, fill = fit_rows(mixture_factory(cfg, rows, expert_name, scope, mode), rows, tr)
            for ctx, Gc in G.items():
                p = predict_windows(mix, rows, te, Gc, fill)
                m = binary_metrics(y[te][tel], p[tel], tau)
                m["retention_r"] = retention_check(df.iloc[te], p)
                m["per_experiment"] = {c: binary_metrics(y[te][tel & (exp[te] == c)], p[tel & (exp[te] == c)],
                                                         tau) for c in classes}
                eng_folds.setdefault(f"{name}/{ctx}", []).append({"held_out": g, **m})

    row_labeled = np.isfinite(rows.y)
    results = {
        "config_hash": h,
        "config": cfg,
        "data": {"n_sessions": int(df["session"].nunique()), "n_windows": int(len(df)),
                 "label_coverage": float(labeled.mean()),
                 "pos_rate": float(y[labeled].mean()) if labeled.any() else None,
                 "pos_rate_by_exp": {c: float(y[labeled & (exp == c)].mean()) if (labeled & (exp == c)).any()
                                     else None for c in classes},
                 "aggregate": cfg["engagement"]["aggregate"], "n_rows": int(len(rows.y)),
                 "row_label_coverage": float(row_labeled.mean()),
                 "row_pos_rate": float(rows.y[row_labeled].mean()) if row_labeled.any() else None},
        "gate": {"summary": summarize(gate_folds), "folds": gate_folds},
        "engagement": {k: {"summary": summarize(v), "folds": v} for k, v in eng_folds.items()},
    }
    out = Path(cfg["output_dir"])
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / f"{h}.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    # Final models on all groups, for inference.
    gate = GATES.build(cfg["gate"]["name"], classes=classes, seed=cfg["seed"],
                       **cfg["gate"].get("params", {})).fit(Xg, exp)
    mix, fill = fit_rows(mixture_factory(cfg, rows, None, mcfg["expert_scope"], mcfg["mode"]), rows,
                         np.arange(len(df)))
    (out / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump({"config_hash": h, "gate": gate, "mixture": mix, "gate_cols": gate_cols,
                 "aggregate": cfg["engagement"]["aggregate"], "row_cols": rows.cols, "fill": fill, "tau": tau},
                out / "models" / f"{h}.joblib")

    print(f"config {h[:12]}  aggregate={cfg['engagement']['aggregate']}  windows={len(df)}  "
          f"rows={len(rows.y)}  window label coverage={labeled.mean():.2f}  row label coverage={row_labeled.mean():.2f}")
    print("gate (classified) accuracy:", results["gate"]["summary"]["accuracy"])
    for k, v in results["engagement"].items():
        s = v["summary"]
        print(f"  {k:32s} F1 {s['f1']['mean']:.3f}+/-{s['f1']['std']:.3f}  AUC {s['auc']['mean']:.3f}  "
              f"bAcc {s['balanced_acc']['mean']:.3f}")


if __name__ == "__main__":
    main()

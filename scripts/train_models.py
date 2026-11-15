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
                         early_rows, fit_engagement, late_rows, predict_windows, student_key)
from src.ingestion import aug_tag, hflip_boxes
from src.labels import label_session, labeler_labels_session, student_labels_session
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
        df["stem"], df["flip_width"], df["aug"] = csv.stem, 0, False
        parts.append(df)
    if not parts:
        raise SystemExit(f"No cached features in {features_dir(cfg)}; run `run_pipeline extract` first.")
    return pd.concat(parts, ignore_index=True)


def with_augmented(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Append the augmented copies (config augmentation.copies) of every session in `df`. A copy has
    the same timestamps, so the same windows and window labels as its original; `aug` marks it,
    `flip_width` is the frame width when it is mirrored (0 otherwise). Training only: callers must
    keep `aug` rows out of every test fold (`heldout_splits`)."""
    aug = cfg.get("augmentation", {})
    copies = aug.get("copies", 0)
    if not copies:
        return df
    parts = [df]
    for c in range(1, copies + 1):
        tag = aug_tag(aug, c)
        for session, sdf in df.groupby("session", sort=False):
            stem = f"{sdf['stem'].iloc[0]}.{tag}"
            csv, manifest = features_dir(cfg) / f"{stem}.csv", features_dir(cfg) / f"{stem}.manifest.json"
            if not csv.exists():
                raise SystemExit(f"{session}: augmented copy {c} not extracted ({csv.name}); run "
                                 f"`run_pipeline extract --reuse` with the same config")
            a = pd.read_csv(csv)
            if len(a) != len(sdf):
                raise SystemExit(f"{csv.name}: {len(a)} windows, original has {len(sdf)}")
            p = json.loads(manifest.read_text(encoding="utf-8"))["augmentation"]
            labels = sdf[["session", "group", "exp", "window_local", "y", "share", "n_labeled"]].reset_index(drop=True)
            a = pd.concat([a, labels], axis=1)
            a["session"] = a["session"] + f"#{tag}"
            a["stem"], a["flip_width"], a["aug"] = stem, p["frame_wh"][0] if p["hflip"] else 0, True
            parts.append(a)
    return pd.concat(parts, ignore_index=True)


def heldout_splits(df: pd.DataFrame):
    """Leave-one-group-out over the original sessions; augmented copies of the training groups train,
    and no augmented row is ever scored."""
    aug = df["aug"].to_numpy(bool)
    for g, tr, te in logo_splits(df["group"].to_numpy()):
        yield g, tr, te[~aug[te]]


def engagement_rows(cfg: dict, df: pd.DataFrame, vec_cols: list[str], per_labeler: bool = False) -> EngagementRows:
    """The rows the experts see, per config engagement.aggregate. per_labeler (late only): also
    attach each labeler's own label per row and whether that labeler rated the row's student."""
    mode = cfg["engagement"]["aggregate"]
    per_labeler = per_labeler or cfg["engagement"].get("labelers", "consensus") == "ensemble"
    if mode == "early" and per_labeler:
        raise SystemExit("engagement.labelers='ensemble' needs per-student rows: set engagement.aggregate='late'")
    if mode == "early":
        X, cols, window = early_rows(df, vec_cols)
        return EngagementRows(X, cols, window, df["exp"].to_numpy(), df["y"].to_numpy(float))
    if mode != "late":
        raise SystemExit(f"engagement.aggregate must be 'early' or 'late', got {mode!r}")

    index = DipserIndex.from_config(cfg)
    assoc = cfg["association"]
    tables, ys, per_session = [], [], []
    for session, sdf in df.groupby("session", sort=False):
        key = SessionKey(*session.split("#")[0].split("/"))         # "#<tag>" marks an augmented copy
        stem = sdf["stem"].iloc[0] if "stem" in sdf else session_stem(key)
        tracks_csv = features_dir(cfg) / f"{stem}.tracks.csv"
        track_npz = tracks_dir(cfg) / f"{stem}.npz"
        if not tracks_csv.exists() or not track_npz.exists():
            raise SystemExit(f"{session}: per-student rows or frame-pass cache missing; "
                             f"run `run_pipeline extract --reuse` (recomputes them from cached tracks)")
        t = pd.read_csv(tracks_csv)
        subjects = index.annotations(key)
        width = int(sdf["flip_width"].iloc[0]) if "flip_width" in sdf else 0
        if width:                                                   # mirrored copy: mirror the manual boxes
            subjects = {s: hflip_boxes(b[None], width)[0] for s, b in subjects.items()}
        tid_to_sid = associate_sids(load_track(track_npz).tracklets, subjects,
                                    assoc["min_iou"], assoc.get("min_match_share", 0.5))
        windows = [Window(a, b) for a, b in zip(sdf.t_start, sdf.t_end)]
        labels = student_labels_session(index, key, windows, cfg["labels"])
        local = t["window"].to_numpy(int)
        sids = [tid_to_sid.get(int(tid)) for tid in t["tid"]]
        ys.append(np.array([labels[s][w] if s in labels else np.nan for s, w in zip(sids, local)], float))
        if per_labeler:
            by = labeler_labels_session(index, key, windows, cfg["labels"])
            per_session.append({name: (np.array([v[s][w] if s in v else np.nan for s, w in zip(sids, local)], float),
                                       np.array([s in v for s in sids], bool)) for name, v in by.items()})
        t["window"] = sdf.index.to_numpy()[local]                   # local window -> global row of df
        tables.append(t)
    tracks = pd.concat(tables, ignore_index=True)
    X, cols, window = late_rows(df, tracks, context=cfg["engagement"].get("late_context", True))
    rows = EngagementRows(X, cols, window, df["exp"].to_numpy()[window], np.concatenate(ys))
    if per_labeler:                                                 # a labeler absent from a session: NaN / not rated
        names = sorted({n for s in per_session for n in s})
        sizes = [len(y) for y in ys]
        rows.y_by_labeler = {n: np.concatenate([s[n][0] if n in s else np.full(k, np.nan)
                                                for s, k in zip(per_session, sizes)]) for n in names}
        rows.rated_by = {n: np.concatenate([s[n][1] if n in s else np.zeros(k, bool)
                                            for s, k in zip(per_session, sizes)]) for n in names}
    return rows


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

    df = with_augmented(cfg, load_table(cfg))        # + augmented copies (training only), if configured
    orig = ~df["aug"].to_numpy(bool)
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
    for g, tr, te in heldout_splits(df):
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
            mix, fill = fit_engagement(cfg["engagement"], mixture_factory(cfg, rows, expert_name, scope, mode),
                                       rows, tr)
            for ctx, Gc in G.items():
                p = predict_windows(mix, rows, te, Gc, fill)
                m = binary_metrics(y[te][tel], p[tel], tau)
                m["retention_r"] = retention_check(df.iloc[te], p)
                m["per_experiment"] = {c: binary_metrics(y[te][tel & (exp[te] == c)], p[tel & (exp[te] == c)],
                                                         tau) for c in classes}
                eng_folds.setdefault(f"{name}/{ctx}", []).append({"held_out": g, **m})

    row_orig = orig[rows.window]
    row_labeled = np.isfinite(rows.y) & row_orig
    lab = labeled & orig                                  # data summary: original footage only
    results = {
        "config_hash": h,
        "config": cfg,
        "data": {"n_sessions": int(df.loc[orig, "session"].nunique()), "n_windows": int(orig.sum()),
                 "label_coverage": float(labeled[orig].mean()),
                 "pos_rate": float(y[lab].mean()) if lab.any() else None,
                 "pos_rate_by_exp": {c: float(y[lab & (exp == c)].mean()) if (lab & (exp == c)).any()
                                     else None for c in classes},
                 "aggregate": cfg["engagement"]["aggregate"], "n_rows": int(row_orig.sum()),
                 "row_label_coverage": float(row_labeled.sum() / max(row_orig.sum(), 1)),
                 "row_pos_rate": float(rows.y[row_labeled].mean()) if row_labeled.any() else None,
                 "augmentation_copies": int(cfg.get("augmentation", {}).get("copies", 0)),
                 "augmented_windows_for_training": int((~orig).sum()),
                 "augmented_row_label_coverage": (float(np.isfinite(rows.y[~row_orig]).mean())
                                                  if (~row_orig).any() else None)},
        "gate": {"summary": summarize(gate_folds), "folds": gate_folds},
        "engagement": {k: {"summary": summarize(v), "folds": v} for k, v in eng_folds.items()},
    }
    out = Path(cfg["output_dir"])
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / f"{h}.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    # Final models on all groups, for inference.
    gate = GATES.build(cfg["gate"]["name"], classes=classes, seed=cfg["seed"],
                       **cfg["gate"].get("params", {})).fit(Xg, exp)
    mix, fill = fit_engagement(cfg["engagement"], mixture_factory(cfg, rows, None, mcfg["expert_scope"], mcfg["mode"]), rows,
                         np.arange(len(df)))
    (out / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump({"config_hash": h, "gate": gate, "mixture": mix, "gate_cols": gate_cols,
                 "aggregate": cfg["engagement"]["aggregate"], "row_cols": rows.cols, "fill": fill, "tau": tau},
                out / "models" / f"{h}.joblib")

    d = results["data"]
    print(f"config {h[:12]}  aggregate={d['aggregate']}  windows={d['n_windows']}  rows={d['n_rows']}  "
          f"window label coverage={d['label_coverage']:.2f}  row label coverage={d['row_label_coverage']:.2f}"
          + (f"  | +{d['augmented_windows_for_training']} augmented windows for training "
             f"({d['augmentation_copies']} copies)" if d["augmentation_copies"] else ""))
    print("gate (classified) accuracy:", results["gate"]["summary"]["accuracy"])
    for k, v in results["engagement"].items():
        s = v["summary"]
        print(f"  {k:32s} F1 {s['f1']['mean']:.3f}+/-{s['f1']['std']:.3f}  AUC {s['auc']['mean']:.3f}  "
              f"bAcc {s['balanced_acc']['mean']:.3f}")


if __name__ == "__main__":
    main()

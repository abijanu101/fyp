"""Six-session run, scored with the metrics of docs/misc/benchmarks.md (experiments/04 §3.1).
  python scripts/analysis/e2e6/benchmark_metrics.py

Replays the leave-one-group-out loop of train_models for a few variants, keeps the held-out
predictions, and scores them at window level (labelled share, thresholded label) and at student
level (each linked student's own label), each per window and per change, pooled and per session.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from scripts.run_pipeline import parse_sets
from scripts.train_models import (engagement_rows, heldout_splits, load_table, mixture_factory,
                                  with_augmented)
from src.core.config import load_config
from src.core.pipeline import SCENE_PREFIX, VEC_PREFIX
from src.experts.levels import fit_engagement
from src.spu import GATES

import json

import scripts.train_models as tm


def merge(new, old):
    """The run's stored config wins; keys added to the config since then come from the current one."""
    if not (isinstance(new, dict) and isinstance(old, dict)):
        return old
    return {k: merge(new[k], old[k]) if k in old else new[k] for k in new} | {k: v for k, v in old.items() if k not in new}


RESULTS = Path("outputs/experiments/04_six_session/results/24387875c765b05d0a49d1c402d747c9ad7a263d43a3df273966aebd7cbc22d6.json")
BASE = merge(load_config("configs/deliverable3.json", parse_sets(['output_dir="tmp/e2e6"'])),
             json.loads(RESULTS.read_text())["config"])
# the run's caches, named by the feature/tracking hashes of that day
tm.features_dir = lambda cfg: Path("outputs/cache/e2e6/features/9a18772f0052")
tm.tracks_dir = lambda cfg: Path("outputs/cache/e2e6/tracks/300d595df270")
features_dir = tm.features_dir
TAU_WINDOW = BASE["labels"]["tau_share"]


def run_weights(keys: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 / length of the run of equal labels each sample belongs to (runs split by key)."""
    w = np.ones(len(y))
    start = 0
    for i in range(1, len(y) + 1):
        if i == len(y) or keys[i] != keys[start] or y[i] != y[start]:
            w[start:i] = 1.0 / (i - start)
            start = i
    return w


def cls_metrics(y, p, tau, w=None):
    y = np.asarray(y, int)
    pred = (np.asarray(p) >= tau).astype(int)
    both = len(np.unique(y)) == 2
    return {"AUC": roc_auc_score(y, p, sample_weight=w) if both else np.nan,
            "acc": accuracy_score(y, pred, sample_weight=w),
            "bal acc": balanced_accuracy_score(y, pred, sample_weight=w) if both else np.nan,
            "F1 engaged": f1_score(y, pred, sample_weight=w, zero_division=0),
            "F1 not engaged": f1_score(1 - y, 1 - pred, sample_weight=w, zero_division=0),
            "macro F1": f1_score(y, pred, average="macro", labels=[0, 1], sample_weight=w, zero_division=0),
            "weighted F1": f1_score(y, pred, average="weighted", sample_weight=w, zero_division=0)}


def views(y, p, tau, keys, session, order=None):
    """Per window / per change x pooled / per session. order: time order within each key (runs)."""
    order = np.arange(len(y)) if order is None else order
    w = np.empty(len(y))
    w[order] = run_weights(keys[order], y[order])
    out = {}
    for sampling, ww in (("per window", None), ("per change", w)):
        out[(sampling, "pooled")] = cls_metrics(y, p, tau, ww)
        per = [cls_metrics(y[m], p[m], tau, None if ww is None else ww[m])
               for s in np.unique(session) for m in [session == s]]
        out[(sampling, "per session")] = {k: np.nanmean([d[k] for d in per]) for k in per[0]}
    return out


def collect(cfg, variant):
    """Held-out window scores (and student-row scores) for one variant over the three folds."""
    df = with_augmented(cfg, load_table(cfg))
    vec_cols = sorted(c for c in df if c.startswith(VEC_PREFIX))
    scene_cols = sorted(c for c in df if c.startswith(SCENE_PREFIX))
    rows = engagement_rows(cfg, df, vec_cols)
    Xg, exp = df[scene_cols].to_numpy(float), df["exp"].to_numpy()
    p_win = np.full(len(df), np.nan)
    p_row = np.full(len(rows.window), np.nan)
    for g, tr, te in heldout_splits(df):
        if variant == "minus flow":
            flow = [c for c in vec_cols if c.startswith("vec.rf.mag_") and c.endswith(".mean")]
            p_win[te] = -df.iloc[te][flow].mean(1).to_numpy()
            continue
        if variant == "train mean":
            p_win[te] = df["share"].iloc[tr].mean()
            p_row[np.isin(rows.window, te)] = np.nanmean(rows.y[np.isin(rows.window, tr)])
            continue
        gate = GATES.build(cfg["gate"]["name"], classes=cfg["classes"], seed=cfg["seed"],
                           **cfg["gate"].get("params", {})).fit(Xg[tr], exp[tr])
        G = gate.predict_proba(Xg[te])
        expert, scope = {"shared expert": (None, "shared"), "shared expert, no context": (None, "shared"),
                         "hand weighted": ("hand_weighted", "per_experiment")}[variant]
        mix, fill = fit_engagement(cfg["engagement"], mixture_factory(cfg, rows, expert, scope,
                                                                       cfg["mixture"]["mode"]), rows, tr)
        pos = np.full(len(df), -1)
        pos[te] = np.arange(len(te))
        sel = np.isin(rows.window, te)
        pr = mix.predict_proba(rows.X[sel], G[pos[rows.window[sel]]])
        p_row[sel] = pr
        sums = np.bincount(rows.window[sel], weights=pr, minlength=len(df))
        cnt = np.bincount(rows.window[sel], minlength=len(df))
        p_win[te] = np.where(cnt[te] > 0, sums[te] / np.maximum(cnt[te], 1), fill)
    return df, rows, p_win, p_row


def main():
    no_ctx = copy.deepcopy(BASE)
    no_ctx["engagement"]["late_context"] = False
    variants = [("train mean", BASE), ("shared expert", BASE), ("shared expert, no context", no_ctx),
                ("hand weighted", BASE), ("minus flow", BASE)]
    win_rows, cont_rows, stu_rows = [], [], []
    for name, cfg in variants:
        df, rows, p_win, p_row = collect(cfg, name)
        ok = df["y"].notna().to_numpy()
        y, share, sess = df["y"].to_numpy(), df["share"].to_numpy(), df["session"].to_numpy()
        # thresholds: the pipeline's cut-off for calibrated scores; the median for raw feature scores
        tau = TAU_WINDOW if name != "minus flow" else np.nanmedian(p_win)
        for (sampling, pooling), m in views(y[ok].astype(int), p_win[ok], tau, sess[ok], sess[ok]).items():
            win_rows.append({"variant": name, "sampling": sampling, "pooling": pooling, **m})
        okc = np.isfinite(share) & np.isfinite(p_win)
        per_r = [np.corrcoef(p_win[okc & (sess == s)], share[okc & (sess == s)])[0, 1]
                 for s in np.unique(sess) if np.std(p_win[okc & (sess == s)]) > 0]
        calibrated = name in ("train mean", "shared expert", "shared expert, no context")
        cont_rows.append({"variant": name,
                          "MAE": np.mean(np.abs(p_win[okc] - share[okc])) if calibrated else np.nan,
                          "RMSE": np.sqrt(np.mean((p_win[okc] - share[okc]) ** 2)) if calibrated else np.nan,
                          "r pooled": np.corrcoef(p_win[okc], share[okc])[0, 1] if np.std(p_win[okc]) > 0 else np.nan,
                          "r within session": np.mean(per_r) if per_r else np.nan})
        if name != "minus flow":
            rs = np.isfinite(rows.y) & np.isfinite(p_row)
            tid = np.concatenate([pd.read_csv(tm.features_dir(cfg) / f"{st}.tracks.csv")["tid"].to_numpy()
                                  for st in df.groupby("session", sort=False)["stem"].first()])
            assert len(tid) == len(rows.window)
            keys = np.array([f"{a}/{t}" for a, t in zip(sess[rows.window], tid)])[rs]
            order = np.lexsort((rows.window[rs], keys))
            for (sampling, pooling), m in views(rows.y[rs].astype(int), p_row[rs], 0.5, keys,
                                                sess[rows.window][rs], order).items():
                stu_rows.append({"variant": name, "sampling": sampling, "pooling": pooling, **m})
    pd.set_option("display.width", 250)
    print("\n=== window level, thresholded label (engaged share >= tau_share)")
    print(pd.DataFrame(win_rows).round(3).to_string(index=False))
    print("\n=== window level, continuous labelled share")
    print(pd.DataFrame(cont_rows).round(3).to_string(index=False))
    print("\n=== student level, each linked student's own label (threshold 0.5)")
    print(pd.DataFrame(stu_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

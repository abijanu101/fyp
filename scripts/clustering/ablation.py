"""The clustering experiment's arms, on the same folds and models: task 10 of guide 15 §15.9 (the
test of §15.6).

  python -m scripts.clustering.ablation [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--candidates <dir>] [--arms null sync agglomerative hdbscan prototype sync+best]
         [--best agglomerative] [--prototypes 8] [--proto-set subset] [--no-window]
         [--context classified] [--out outputs/experiments/10_clustering]

Arms (each = the configured inputs, trk.* plus the class context only if
engagement.use_class_context, plus the candidate columns named):
  null           no class-relative features (the baseline)
  sync           option 0: deviation, synchrony and lag with the class (sync.*)
  agglomerative  option 1 in both spaces, with the window's chaos and fragmentation (agg.*)
  hdbscan        option 2, the variant of option 1 (hdb.*)
  prototype      option 4, a K-prototype codebook fitted per fold on the training groups, no labels
  sync+best      sync plus the best of agglomerative / hdbscan / prototype: --best names it; if not
                 given it is the one with the lowest held-out experts MAE, which uses the test folds
                 to choose, so freeze --best before a run is cited
--no-window drops the window-level cluster.* columns (the same for every student in a window, so
they can act as session identifiers, as the class context did in experiment 08).

Protocol: train_models' leave-one-group-out (augmented copies train, never scored), the configured
per-labeler model, each entry scored once with rating_metrics, held-out and on the training entries
(underfitting vs overfitting), next to constant 3 and the training mean.

The test (§15.6): an arm beats another when its mean held-out MAE is lower by more than the SD of
the per-fold differences (and the same for Spearman, higher). Clustering is justified only if
agglomerative, hdbscan, prototype or sync+best beats sync; if nothing beats null, class-relative
information is not visible in these features.

Writes to --out/ablation/: <config hash>-<settings hash>.json (every fold's metrics, the settings,
the candidates used), ablation.csv (mean and SD per arm, track, split) and ablation_verdict.csv
(each arm against null and against sync).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.clustering.common import (PER_WINDOW, base_columns, base_parser, candidate_columns, fit_model,
                                       fold_context, folds, label_table, latest_candidates, load, load_candidates,
                                       own_ratings, params_hash, score, with_fold_prototypes, write_table)
from src.core.config import config_hash
from src.labels.interp import to_rating

ARM_KEYS = {"sync": [".sync."], "agglomerative": [".agg."], "hdbscan": [".hdb."], "prototype": [".proto."]}
CLUSTER_ARMS = ["agglomerative", "hdbscan", "prototype"]
METRICS = ["mae", "rmse", "spearman", "auc", "c_index"]


def arm_cols(arm: str, cands: list[str], window: bool) -> list[str]:
    keys = [k for part in arm.split("+") for k in ARM_KEYS.get(part, [])]
    cols = [c for c in cands if any(k in c for k in keys)]
    return cols if window else [c for c in cols if not c.startswith(PER_WINDOW)]


def run_arm(cfg, r, cols, tr, te, G, tau) -> dict:
    model = fit_model(cfg, r, cols, tr)
    X = r.reindex(columns=cols).to_numpy(float)
    lab = r["labeler"].to_numpy()
    fit_rows = tr & ~r["aug"].to_numpy(bool)
    out = {f"{t}/heldout": m for t, m in score(r[te], own_ratings(model, X[te], G[te], lab[te]), tau).items()}
    out |= {f"{t}/train": m for t, m in score(r[fit_rows], own_ratings(model, X[fit_rows], G[fit_rows], lab[fit_rows]), tau).items()}
    return out


def baselines(r, tr, te, tau) -> dict:
    from scripts.clustering.common import SELF
    out = {}
    is_self = r["labeler"].to_numpy() == SELF
    for track, in_track in (("experts", ~is_self), ("self", is_self)):
        sel = te & in_track
        if not sel.any():
            continue
        mean = float(to_rating(r["target"].to_numpy(float)[tr & in_track].mean()))
        for name, v in (("constant_3", 3.0), ("train_mean", mean)):
            out[f"{track}/heldout/{name}"] = score(r[sel], np.full(int(sel.sum()), v), tau)[track]
    return out


def summarise(results: dict) -> pd.DataFrame:
    recs = []
    for arm, per_fold in results.items():
        keys = sorted({k for f in per_fold.values() for k in f})
        for k in keys:
            vals = [f[k] for f in per_fold.values() if k in f]
            rec = {"arm": arm, "track": k.split("/")[0], "split": "/".join(k.split("/")[1:]), "folds": len(vals),
                   "n_entries": int(np.sum([v.get("n", 0) for v in vals]))}
            for m in METRICS:
                x = np.array([v.get(m, np.nan) for v in vals], float)
                rec[m] = float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
                rec[f"{m}_sd"] = float(np.nanstd(x, ddof=1)) if np.isfinite(x).sum() > 1 else np.nan
            recs.append(rec)
    return pd.DataFrame(recs)


def verdict(results: dict, against: str) -> list[dict]:
    recs = []
    if against not in results:
        return recs
    for arm, per_fold in results.items():
        if arm == against:
            continue
        for track in ("experts", "self"):
            k = f"{track}/heldout"
            common = [g for g in per_fold if k in per_fold[g] and k in results[against].get(g, {})]
            if len(common) < 2:
                continue
            d_mae = np.array([per_fold[g][k]["mae"] - results[against][g][k]["mae"] for g in common])
            d_sp = np.array([per_fold[g][k]["spearman"] - results[against][g][k]["spearman"] for g in common], float)
            beats_mae = bool(d_mae.mean() < 0 and -d_mae.mean() > d_mae.std(ddof=1))
            beats_sp = bool(np.isfinite(d_sp).all() and d_sp.mean() > 0 and d_sp.mean() > d_sp.std(ddof=1))
            recs.append({"arm": arm, "against": against, "track": track, "folds": len(common),
                         "mae_delta": float(d_mae.mean()), "mae_delta_sd": float(d_mae.std(ddof=1)),
                         "spearman_delta": float(np.nanmean(d_sp)), "spearman_delta_sd": float(np.nanstd(d_sp, ddof=1)),
                         "beats_on_mae": beats_mae, "beats_on_spearman": beats_sp})
    return recs


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--arms", nargs="*", default=["null", "sync", "agglomerative", "hdbscan", "prototype", "sync+best"])
    ap.add_argument("--best", choices=CLUSTER_ARMS, default=None)
    ap.add_argument("--prototypes", type=int, default=8)
    ap.add_argument("--proto-set", default="subset")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--context", choices=["classified", "oracle"], default="classified")
    args = ap.parse_args()
    cfg, out = load(args)
    tau = float(cfg["labels"]["tau_att"])
    cdir = latest_candidates(out, args.candidates)
    rows, windows = label_table(cfg, with_windows=True)
    rows = load_candidates(rows, cdir)
    proto_set = args.proto_set if (out / "descriptor_subset.json").exists() or args.proto_set != "subset" else "all"
    arms = [a for a in args.arms if a != "sync+best"]
    need_proto = any("prototype" in a for a in args.arms) and args.prototypes > 0

    results: dict[str, dict] = {a: {} for a in arms}
    base_res: dict[str, dict] = {}
    fold_list = list(folds(rows))
    columns: dict[str, list[str]] = {}
    for g, tr, te in fold_list:
        r = with_fold_prototypes(cfg, rows, tr, proto_set, args.prototypes, out) if need_proto else rows
        G = fold_context(cfg, r, windows, g, args.context)
        base, cands = base_columns(cfg, r), candidate_columns(r)
        base_res[g] = baselines(r, tr, te, tau)
        for arm in arms:
            cols = base + (arm_cols(arm, cands, not args.no_window) if arm != "null" else [])
            columns[arm] = cols
            results[arm][g] = run_arm(cfg, r, cols, tr, te, G, tau)
        print(f"  fold {g}: {', '.join(f'{a} {len(columns[a])}' for a in arms)} columns", flush=True)

    chosen = None
    if "sync+best" in args.arms:
        chosen = args.best
        if chosen is None:
            scored = {a: np.mean([results[a][g]["experts/heldout"]["mae"] for g in results[a]])
                      for a in CLUSTER_ARMS if a in results}
            if not scored:
                raise SystemExit("sync+best: run at least one of agglomerative, hdbscan, prototype, or give --best")
            chosen = min(scored, key=scored.get)
            print(f"  sync+best: {chosen} chosen on held-out experts MAE (uses the test folds; freeze with --best {chosen})")
        arm = f"sync+{chosen}"
        results[arm] = {}
        for g, tr, te in fold_list:
            r = with_fold_prototypes(cfg, rows, tr, proto_set, args.prototypes, out) if chosen == "prototype" else rows
            G = fold_context(cfg, r, windows, g, args.context)
            cols = base_columns(cfg, r) + arm_cols(arm, candidate_columns(r), not args.no_window)
            columns[arm] = cols
            results[arm][g] = run_arm(cfg, r, cols, tr, te, G, tau)

    table = summarise(results)
    base_tab = summarise({"baseline": base_res})
    base_tab["arm"] = base_tab["split"].str.split("/").str[1]
    base_tab["split"] = "heldout"
    table = pd.concat([table, base_tab], ignore_index=True)
    ver = pd.DataFrame(verdict(results, "null") + verdict(results, "sync"))

    settings = {"candidates": cdir.name, "arms": args.arms, "best": chosen, "best_given": args.best is not None,
                "prototypes": args.prototypes, "proto_set": proto_set, "window_features": not args.no_window,
                "context": args.context}
    h = config_hash(cfg)
    adir = out / "ablation"
    adir.mkdir(parents=True, exist_ok=True)
    name = f"{h[:12]}-{params_hash(settings)}"
    (adir / f"{name}.json").write_text(json.dumps(
        {"config_hash": h, "settings": settings, "candidate_params": json.loads((cdir / "params.json").read_text()),
         "columns": {a: len(c) for a, c in columns.items()}, "arms": results, "baselines": base_res,
         "verdict": ver.to_dict("records")}, indent=2, default=float), encoding="utf-8")
    write_table(table, adir / "ablation")
    write_table(ver, adir / "ablation_verdict")

    show = table[table["split"].isin(["heldout", "train"])]
    for track in ("experts", "self"):
        t = show[show["track"] == track].sort_values(["split", "mae"])
        if t.empty:
            continue
        print(f"\n{track}: mean over {t['folds'].max()} folds ({args.context} context)")
        print(f"  {'arm':22s} {'split':8s} {'MAE':>12s} {'RMSE':>7s} {'Spearman':>16s} {'AUC':>6s}")
        for _, r in t.iterrows():
            print(f"  {r['arm']:22s} {r['split']:8s} {r['mae']:.3f} ±{r['mae_sd']:.3f} {r['rmse']:7.3f} "
                  f"{r['spearman']:+.3f} ±{r['spearman_sd']:.3f} {r['auc']:6.3f}")
    if len(ver):
        print("\nverdict (beats = better by more than the SD of the per-fold differences):")
        for _, v in ver.iterrows():
            print(f"  {v['arm']:22s} vs {v['against']:5s} {v['track']:8s} MAE {v['mae_delta']:+.4f} (sd {v['mae_delta_sd']:.4f})"
                  f"  Spearman {v['spearman_delta']:+.4f} (sd {v['spearman_delta_sd']:.4f})"
                  f"  {'BEATS' if v['beats_on_mae'] or v['beats_on_spearman'] else '-'}")
    print(f"\nwrote {adir}/{name}.json, ablation.csv, ablation_verdict.csv")


if __name__ == "__main__":
    main()

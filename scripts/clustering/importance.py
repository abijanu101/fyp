"""Permutation importance of the candidate features on held-out folds: task 9 of guide 15 §15.9.

  python -m scripts.clustering.importance [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--candidates <dir>] [--arms sync agg hdb proto] [--prototypes 8] [--by family]
         [--repeats 5] [--context oracle] [--gate] [--split both] [--out outputs/experiments/10_clustering]

The model: the configured per-labeler model (expert, e.g. `--set expert.name=...`, mixture and
labelers as in the config) on the configured inputs (trk.*, plus the class context only if
engagement.use_class_context) plus the candidates of the chosen arms. Per fold it is trained on the
training rows (augmented copies included) and scored on the held-out original entries (each entry
once): MAE and Spearman per track (experts, self). Two splits (--split; both by default):
  activity  train on 4 activities, test on the other 2, over all 15 pairs (same students in both)
  group     leave one group out (the test students are unseen)

Importance of a column or family (--by family: e.g. every sync.hp.yaw.* column together, so
correlated columns are not each judged with their partners still in place): shuffle it across the
held-out rows --repeats times and record the rise in MAE and the fall in Spearman. Positive = the
model uses it on unseen data. Base inputs are included, as a scale.

--context oracle (default) gives the experts the true activity, so only their inputs differ;
classified uses the configured gate, as train_models' headline numbers do (group split only: the
activity split always uses oracle, since a gate cannot name an activity it never saw).

--gate: also the activity gate (the configured SPU gate, a random forest in 06-09) on the centred
windows, on the group split only (it predicts the activity, so it cannot be tested on held-out
activities): held-out accuracy with scene.* alone against scene.* plus the window-level cluster.*
candidates, and the permutation importance of each cluster.* family on that accuracy.

Writes to --out (tables as .csv and .json): importance (per split, fold, track, feature: base score,
permuted score, drop), importance_summary (mean and SD over a split's folds), importance_<split>.png;
with --gate, importance_gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.clustering.common import (PER_STUDENT, PER_WINDOW, base_columns, base_parser, candidate_columns,
                                       fit_model, fold_context, folds, label_table, latest_candidates, load,
                                       load_candidates, own_ratings, plot_ready, score, splits, split_parser_arg,
                                       check_split, with_fold_prototypes, write_table)

ARM_KEYS = {"sync": ".sync.", "agg": ".agg.", "hdb": ".hdb.", "proto": ".proto."}


def family(col: str) -> str:
    """trk.cluster.sync.hp.yaw.corr -> sync.hp.yaw;  cluster.agg.desc.chaos -> win:agg.desc;
    trk.hp.yaw.mean -> base:trk.hp"""
    if col.startswith(PER_STUDENT):
        return col[len(PER_STUDENT):].rsplit(".", 1)[0]
    if col.startswith(PER_WINDOW):
        return "win:" + col[len(PER_WINDOW):].rsplit(".", 1)[0]
    return "base:" + ".".join(col.split(".")[:2])


def arm_columns(cols: list[str], arms: list[str]) -> list[str]:
    return [c for c in cols if any(ARM_KEYS[a] in c for a in arms)]


def permutation_drop(model, X: np.ndarray, G, rows_te: pd.DataFrame, groups: dict[str, list[int]], repeats: int,
                     tau: float, rng) -> list[dict]:
    lab = rows_te["labeler"].to_numpy()
    base = score(rows_te, own_ratings(model, X, G, lab), tau)
    recs = []
    for name, idx in groups.items():
        agg: dict[str, list] = {}
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, idx] = X[perm][:, idx]
            for track, m in score(rows_te, own_ratings(model, Xp, G, lab), tau).items():
                agg.setdefault(track, []).append(m)
        for track, ms in agg.items():
            b = base.get(track, {})
            mae = float(np.mean([m["mae"] for m in ms]))
            sp = float(np.nanmean([m["spearman"] for m in ms])) if any(np.isfinite(m["spearman"]) for m in ms) else np.nan
            recs.append({"feature": name, "track": track, "mae_base": b.get("mae"), "mae_permuted": mae,
                         "mae_rise": mae - b.get("mae", np.nan), "spearman_base": b.get("spearman"),
                         "spearman_permuted": sp, "spearman_fall": b.get("spearman", np.nan) - sp})
    return recs


def gate_importance(cfg: dict, rows: pd.DataFrame, repeats: int, rng) -> pd.DataFrame:
    from src.spu import GATES

    win = rows.drop_duplicates(["stem", "t_key"]).reset_index(drop=True)
    scene = sorted(c for c in win if c.startswith("scene."))
    clus = sorted(c for c in win if c.startswith(PER_WINDOW))
    fams: dict[str, list[int]] = {}
    for j, c in enumerate(clus):
        fams.setdefault(family(c), []).append(len(scene) + j)
    y, group, aug = win["exp"].to_numpy(), win["group"].to_numpy(), win["aug"].to_numpy(bool)
    recs = []
    for g in sorted(set(group.tolist())):
        tr, te = group != g, (group == g) & ~aug
        for name, cols in (("scene only", scene), ("scene + cluster", scene + clus)):
            X = win[cols].to_numpy(float)
            gate = GATES.build(cfg["gate"]["name"], classes=cfg["classes"], seed=cfg["seed"],
                               **cfg["gate"].get("params", {})).fit(X[tr], y[tr])
            acc = float((np.array(cfg["classes"])[gate.predict_proba(X[te]).argmax(1)] == y[te]).mean())
            recs.append({"fold": g, "feature": name, "accuracy": acc, "accuracy_drop": np.nan})
            if name == "scene + cluster":
                for fam, idx in fams.items():
                    drops = []
                    for _ in range(repeats):
                        Xp = X[te].copy()
                        Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
                        drops.append(acc - float((np.array(cfg["classes"])[gate.predict_proba(Xp).argmax(1)] == y[te]).mean()))
                    recs.append({"fold": g, "feature": fam, "accuracy": np.nan, "accuracy_drop": float(np.mean(drops))})
    return pd.DataFrame(recs)


def plot(summary: pd.DataFrame, split: str, context: str, path) -> None:
    plt = plot_ready()
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.28 * summary["feature"].nunique() + 2), sharey=True)
    order = summary[summary["track"] == "experts"].sort_values("spearman_fall", ascending=False)["feature"].tolist()
    for ax, track in zip(axes, ("experts", "self")):
        s = summary[summary["track"] == track].set_index("feature").reindex(order)
        y = np.arange(len(order))
        ax.barh(y, s["spearman_fall"], xerr=s["spearman_fall_sd"].fillna(0),
                color=["#bbbbbb" if f.startswith("base:") else "#2a78d6" for f in order], error_kw={"lw": 0.8})
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_yticks(y, order, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("fall in held-out Spearman when shuffled (mean, SD over folds)")
        ax.set_title(f"{split} split, {track} ({context} context; grey = base inputs)", fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--arms", nargs="*", default=["sync", "agg", "hdb", "proto"], choices=list(ARM_KEYS))
    ap.add_argument("--prototypes", type=int, default=8)
    ap.add_argument("--proto-set", default="subset")
    ap.add_argument("--by", choices=["family", "column"], default="family")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--context", choices=["oracle", "classified"], default="oracle")
    ap.add_argument("--gate", action="store_true")
    split_parser_arg(ap)
    args = ap.parse_args()
    cfg, out = load(args)
    for split in splits(args.split):
        check_split(cfg, split)
    rng = np.random.default_rng(cfg["seed"])
    tau = float(cfg["labels"]["tau_att"])
    cdir = latest_candidates(out, args.candidates)
    rows, windows = label_table(cfg, with_windows=True)
    rows = load_candidates(rows, cdir)
    proto_set = args.proto_set if (out / "descriptor_subset.json").exists() or args.proto_set != "subset" else "all"
    use_proto = "proto" in args.arms and args.prototypes > 0

    recs = []
    for split in splits(args.split):
        context = "oracle" if split == "activity" else args.context
        for g, tr, te in folds(rows, split):
            r = with_fold_prototypes(cfg, rows, tr, proto_set, args.prototypes, out) if use_proto else rows
            base = base_columns(cfg, r)
            extra = arm_columns(candidate_columns(r), args.arms)
            cols = base + extra
            groups: dict[str, list[int]] = {}
            for j, c in enumerate(cols):
                groups.setdefault(family(c) if args.by == "family" else c, []).append(j)
            model = fit_model(cfg, r, cols, tr)
            G = fold_context(cfg, r, windows, tr, context, split)
            rte = r[te].reset_index(drop=True)
            X = rte.reindex(columns=cols).to_numpy(float)
            recs += [{"split": split, "context": context, "fold": g, **x}
                     for x in permutation_drop(model, X, G[te], rte, groups, args.repeats, tau, rng)]
            print(f"  {split} fold {g}: {len(base)} base + {len(extra)} candidate columns, "
                  f"{len(groups)} features scored", flush=True)
    df = pd.DataFrame(recs)
    write_table(df, out / "importance")
    summary = df.groupby(["split", "context", "feature", "track"]).agg(
        mae_rise=("mae_rise", "mean"), mae_rise_sd=("mae_rise", "std"), spearman_fall=("spearman_fall", "mean"),
        spearman_fall_sd=("spearman_fall", "std"), folds=("fold", "nunique")).reset_index()
    summary = summary.sort_values(["split", "track", "spearman_fall"], ascending=[True, True, False])
    write_table(summary, out / "importance_summary")
    for (split, context), ss in summary.groupby(["split", "context"]):
        plot(ss, split, context, out / f"importance_{split}.png")
        print(f"\n{split} split: held-out permutation importance ({context} context, {args.repeats} repeats), experts:")
        print(ss[ss["track"] == "experts"].drop(columns=["split", "context"]).round(4).to_string(index=False))
    if args.gate:
        gi = gate_importance(cfg, rows, args.repeats, rng)
        write_table(gi, out / "importance_gate")
        acc = gi.dropna(subset=["accuracy"]).groupby("feature")["accuracy"].mean()
        print("\ngate (group split) held-out accuracy on the centred windows:", acc.round(3).to_dict())
        print(gi.dropna(subset=["accuracy_drop"]).groupby("feature")["accuracy_drop"].mean()
              .sort_values(ascending=False).round(4).to_string())
    print(f"\nwrote {out}/importance, importance_summary (.csv/.json), importance_<split>.png"
          + (", importance_gate" if args.gate else ""))


if __name__ == "__main__":
    main()

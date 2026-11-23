"""Which candidate class-relative features track the ratings, judged on training folds only:
task 8 of guide 15 §15.9.

  python -m scripts.clustering.feature_screen [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--candidates <dir>] [--prototypes 8] [--proto-set subset] [--with-base] [--per-labeler]
         [--split both] [--out outputs/experiments/10_clustering]

Rows: train_models' label rows (one per rating entry and linked track, originals only for the
statistics) with the candidates of `candidates.py` joined on (session, window centre, track), and
the prototype features of a codebook fitted per fold on the training sessions (--prototypes K; 0 = off).

Two splits (--split; both by default, reported separately):
  activity  train on 4 activities, test on the other 2, over all 15 pairs (same students in both)
  group     leave one group out (3 folds; the test students are unseen)

For each fold, on the TRAINING rows only:
  within-student Spearman rho  feature and rating both demeaned per student-session and labeler (as
                               in feature_study), so a feature cannot win by telling students apart,
                               only by moving when the rating moves
  mutual information           of the demeaned feature with the demeaned rating (k-NN estimator)
Summaries over a split's folds: mean and SD of rho, whether every fold has the same sign, mean MI.
The held-out rows' rho is reported beside it, for reading only: choosing features on it would leak
the test fold.

Writes to --out (tables as .csv and .json):
  feature_screen            per (split, candidate, track): the summaries; sorted by |mean rho| on experts
  feature_screen_folds      per (split, fold, candidate, track): train rho, MI, n, held-out rho
  feature_screen_<split>.png  mean within-student rho per candidate (bars) with the fold range
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.clustering.common import (EXPERTS, SELF, base_columns, base_parser, candidate_columns, demean, folds,
                                       label_table, latest_candidates, load, load_candidates, plot_ready, splits,
                                       split_parser_arg, with_fold_prototypes, write_table)


def spearman(x: np.ndarray, y: np.ndarray, min_n: int = 30) -> tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return np.nan, int(m.sum())
    a, b = pd.Series(x[m]).rank().to_numpy(), pd.Series(y[m]).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1]), int(m.sum())


def mutual_info(x: np.ndarray, y: np.ndarray, rng, max_n: int = 20000) -> float:
    from sklearn.feature_selection import mutual_info_regression

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50 or np.ptp(x[m]) == 0:
        return np.nan
    idx = np.flatnonzero(m)
    if len(idx) > max_n:
        idx = rng.choice(idx, size=max_n, replace=False)
    return float(mutual_info_regression(x[idx, None], y[idx], random_state=0)[0])


def tracks(rows: pd.DataFrame, per_labeler: bool) -> dict[str, np.ndarray]:
    t = {"experts": rows["labeler"].isin(EXPERTS).to_numpy(), "self": (rows["labeler"] == SELF).to_numpy()}
    if per_labeler:
        t |= {lab: (rows["labeler"] == lab).to_numpy() for lab in EXPERTS}
    return t


def same_sign(s: pd.Series) -> bool:
    s = s.dropna()
    return bool(len(s) > 1 and (np.all(s > 0) or np.all(s < 0)))


def screen_split(cfg, rows, split, args, proto_set, out, rng) -> list[dict]:
    recs = []
    for g, tr, te in folds(rows, split):
        r = with_fold_prototypes(cfg, rows, tr, proto_set, args.prototypes, out) if args.prototypes else rows
        cols = candidate_columns(r) + (base_columns(cfg, r) if args.with_base else [])
        orig = ~r["aug"].to_numpy(bool)
        y = r["rating"].astype(float)
        for track, m in tracks(r, args.per_labeler).items():
            trm, tem = m & tr & orig, m & te
            if not trm.any():
                continue
            ytr = demean(y[trm], r.loc[trm, "student"]).to_numpy()
            yte = demean(y[tem], r.loc[tem, "student"]).to_numpy() if tem.any() else None
            Xtr = demean(r.loc[trm, cols].astype(float), r.loc[trm, "student"])
            Xte = demean(r.loc[tem, cols].astype(float), r.loc[tem, "student"]) if tem.any() else None
            for c in cols:
                rho, n = spearman(Xtr[c].to_numpy(), ytr)
                rho_te = spearman(Xte[c].to_numpy(), yte)[0] if Xte is not None else np.nan
                recs.append({"split": split, "fold": g, "candidate": c, "track": track, "n_train": n,
                             "rho_train": rho, "mi_train": mutual_info(Xtr[c].to_numpy(), ytr, rng),
                             "rho_heldout": rho_te})
        print(f"  {split} fold {g}: {int((tr & orig).sum())} training rows, {len(cols)} columns", flush=True)
    return recs


def plot(summary: pd.DataFrame, top: list[str], split: str, path) -> None:
    plt = plot_ready()
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.25 * len(top) + 2), sharey=True)
    for ax, track in zip(axes, ("experts", "self")):
        s = summary[summary["track"] == track].set_index("candidate").reindex(top)
        y = np.arange(len(top))
        ax.barh(y, s["rho_mean"], color=np.where(s["same_sign"].fillna(False).astype(bool), "#2a78d6", "#bbbbbb"))
        ax.hlines(y, s["rho_min"], s["rho_max"], color="black", lw=0.8)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_yticks(y, [c.replace("trk.cluster.", "").replace("cluster.", "win:") for c in top], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("within-student Spearman rho, training folds (bar mean, line range)")
        ax.set_title(f"{split} split, {track} (blue = same sign in every fold)", fontsize=10)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--candidates", default=None, help="candidate folder (default: the newest under --out)")
    ap.add_argument("--prototypes", type=int, default=8, help="codebook size K per fold; 0 = no prototype features")
    ap.add_argument("--proto-set", default="subset", help="descriptor set for the codebook (falls back to all)")
    ap.add_argument("--with-base", action="store_true", help="screen the trk.* model inputs too, for reference")
    ap.add_argument("--per-labeler", action="store_true", help="also each expert labeler on their own")
    split_parser_arg(ap)
    args = ap.parse_args()
    cfg, out = load(args)
    rng = np.random.default_rng(cfg["seed"])
    cdir = latest_candidates(out, args.candidates)
    rows = load_candidates(label_table(cfg), cdir)
    proto_set = args.proto_set if (out / "descriptor_subset.json").exists() or args.proto_set != "subset" else "all"

    recs = []
    for split in splits(args.split):
        recs += screen_split(cfg, rows, split, args, proto_set, out, rng)
    df = pd.DataFrame(recs)
    write_table(df, out / "feature_screen_folds")

    summary = df.groupby(["split", "candidate", "track"]).agg(
        rho_mean=("rho_train", "mean"), rho_sd=("rho_train", "std"), rho_min=("rho_train", "min"),
        rho_max=("rho_train", "max"), same_sign=("rho_train", same_sign), mi_mean=("mi_train", "mean"),
        n_train=("n_train", "mean"), folds=("fold", "nunique"), rho_heldout_mean=("rho_heldout", "mean")).reset_index()
    parts = []
    print(f"\ncandidates {cdir.name}")
    for split, ss in summary.groupby("split", sort=False):
        order = ss[ss["track"] == "experts"].assign(a=lambda d: d["rho_mean"].abs()).sort_values("a", ascending=False)
        rank = {c: i for i, c in enumerate(order["candidate"])}
        parts.append(ss.assign(rank=ss["candidate"].map(rank)).sort_values(["rank", "track"]).drop(columns="rank"))
        plot(ss, order["candidate"].tolist()[:40], split, out / f"feature_screen_{split}.png")
        print(f"\n{split} split ({int(ss['folds'].max())} folds): strongest on the experts' training rows (within student):")
        for _, rr in order.head(15).iterrows():
            selfr = ss[(ss["candidate"] == rr["candidate"]) & (ss["track"] == "self")]
            sr = selfr["rho_mean"].iloc[0] if len(selfr) else np.nan
            print(f"  {rr['candidate']:50s} rho {rr['rho_mean']:+.3f} (sd {rr['rho_sd']:.3f}, "
                  f"{'same sign' if rr['same_sign'] else 'mixed'})  MI {rr['mi_mean']:.4f}  self {sr:+.3f}")
    write_table(pd.concat(parts, ignore_index=True), out / "feature_screen")
    print(f"\nwrote {out}/feature_screen.csv/.json, feature_screen_folds.csv/.json, feature_screen_<split>.png")


if __name__ == "__main__":
    main()

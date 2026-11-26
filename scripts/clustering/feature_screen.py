"""Which candidate class-relative features track the ratings, judged on training folds only:
task 8 of guide 15 §15.9.

  python -m scripts.clustering.feature_screen [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--candidates <dir>] [--prototypes 8] [--proto-set subset] [--with-base] [--per-labeler]
         [--split both] [--perms 200] [--workers 1] [--out outputs/experiments/10_clustering]

Folds run in parallel with --workers (each worker holds a copy of the label rows, ~0.5 GB on the full
data, so 2-4 workers on 16 GB) and are resumable: each finished fold is saved under
<out>/partial/feature_screen/<settings hash>/, and a rerun only computes the missing folds.

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
  block-null p-value           the same rho recomputed --perms times with each student's ratings
                               circularly shifted along that student's own time order (a random
                               offset per student): the shift keeps the ratings' autocorrelation and
                               each student's mean, and breaks only their alignment with the features,
                               so neighbouring, overlapping windows are not counted as independent
                               evidence. Two-sided, (1 + #|null| >= |rho|) / (1 + perms)
  q-value                      Benjamini-Hochberg over the candidates of one fold and track
Summaries over a split's folds: mean and SD of rho, whether every fold has the same sign, the median
p and q, and the share of folds with q < 0.05. The held-out rows' rho is reported beside it, for
reading only: choosing features on it would leak the test fold. (Mutual information is not reported:
the k-NN estimator breaks on the tied values that per-student demeaning creates, giving ~4 nats on
independent data.)

Missing values: the models fill them with the training median inside each fold (src/core/classifiers.py).
As a check that missingness does not stand in for the recording, `feature_screen_missingness.json`
reports how well the candidates' NaN pattern alone predicts the recording (5-fold accuracy of a
Bernoulli naive Bayes, against always guessing the largest recording).

Writes to --out (tables as .csv and .json):
  feature_screen            per (split, candidate, track): the summaries; sorted by |mean rho| on experts
  feature_screen_folds      per (split, fold, candidate, track): train rho, n, block-null p, q, held-out rho
  feature_screen_missingness.json  the NaN-pattern check
  feature_screen_<split>.png  mean within-student rho per candidate (bars) with the fold range
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.clustering.common import (EXPERTS, SELF, base_columns, base_parser, bh_qvalues, candidate_columns,
                                       demean, fold_jobs, fold_of, label_table, latest_candidates, load,
                                       load_candidates, plot_ready, rows_key, shared, splits, split_parser_arg,
                                       subset_text,
                                       with_fold_prototypes, write_table)
from src.core.config import config_hash


def spearman(x: np.ndarray, y: np.ndarray, min_n: int = 30) -> tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return np.nan, int(m.sum())
    a, b = pd.Series(x[m]).rank().to_numpy(), pd.Series(y[m]).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1]), int(m.sum())


def shift_within(order_keys: np.ndarray, n_perms: int, rng) -> np.ndarray:
    """(n_perms, n) index arrays: rows grouped by `order_keys` (already sorted by group, then time),
    each group's rows rotated by its own random offset in 1..len-1 (groups of one stay put)."""
    n = len(order_keys)
    starts = np.flatnonzero(np.r_[True, order_keys[1:] != order_keys[:-1]])
    lens = np.diff(np.r_[starts, n])
    group = np.repeat(np.arange(len(starts)), lens)
    pos = np.arange(n) - starts[group]
    out = np.empty((n_perms, n), int)
    for k in range(n_perms):
        off = np.where(lens > 1, rng.integers(1, np.maximum(lens, 2)), 0)
        out[k] = starts[group] + (pos + off[group]) % lens[group]
    return out


def rank_corr_with_null(X: pd.DataFrame, y: np.ndarray, perm_idx: np.ndarray, min_n: int = 30):
    """Per column: (rho, n, p). rho is the Spearman correlation of the column with y on the rows
    where both are finite (x ranked on those rows, y ranked once on all rows: the same order, and
    every permutation reuses the ranks); p is two-sided against the permuted y (rows of perm_idx).

    Memory: one float32 (perms x n) matrix of permuted y ranks and one of their squares; each column
    then costs three matrix-vector products, without copying the matrix per column."""
    ry = pd.Series(y).rank().to_numpy()
    fin_y = np.isfinite(ry)
    Yp = np.where(fin_y, ry, 0.0)[perm_idx].astype(np.float32)            # (P, n)
    Yp2 = Yp * Yp
    out = {}
    for c in X.columns:
        x = X[c].to_numpy(float)
        m = np.isfinite(x) & fin_y
        n = int(m.sum())
        if n < min_n or np.ptp(x[m]) == 0:
            out[c] = (np.nan, n, np.nan)
            continue
        a = pd.Series(x[m]).rank().to_numpy()
        a = (a - a.mean()) / (a.std() or 1.0)                              # mean 0, SD 1 on the mask
        b = ry[m]
        rho = float(a @ ((b - b.mean()) / (b.std() or 1.0)) / n)
        a_full = np.zeros(len(x), np.float32)
        a_full[m] = a
        w = m.astype(np.float32)
        s1, s2, cross = Yp @ w, Yp2 @ w, Yp @ a_full                       # per permutation, over the mask
        sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-12))
        null = cross / (n * sd)                                            # a has mean 0, so no mean term
        p = float((1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (1 + len(null)))
        out[c] = (rho, n, p)
    return out


def missingness_check(rows: pd.DataFrame, cols: list[str], seed: int) -> dict:
    """How well the NaN pattern of the candidates alone predicts the recording (original rows, one
    per window and student): 5-fold accuracy of a Bernoulli naive Bayes against the largest class."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.naive_bayes import BernoulliNB

    d = rows[~rows["aug"].to_numpy(bool)].drop_duplicates(["stem", "t_key", "tid"])
    if d["stem"].nunique() < 2 or len(d) < 50:
        return {"note": "fewer than 2 recordings: not computed"}
    M = d[cols].isna().to_numpy(float)
    y = d["stem"].to_numpy()
    counts = pd.Series(y).value_counts()
    k = int(min(5, counts.min()))
    if k < 2:
        return {"note": "a recording has fewer than 2 rows: not computed"}
    acc = cross_val_score(BernoulliNB(), M, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed)).mean()
    return {"rows": int(len(d)), "recordings": int(counts.size), "nan_share": float(M.mean()),
            "accuracy_from_nan_pattern": float(acc), "largest_recording_share": float(counts.iloc[0] / len(d)),
            "columns_ever_nan": int((M.sum(axis=0) > 0).sum())}


def tracks(rows: pd.DataFrame, per_labeler: bool) -> dict[str, np.ndarray]:
    t = {"experts": rows["labeler"].isin(EXPERTS).to_numpy(), "self": (rows["labeler"] == SELF).to_numpy()}
    if per_labeler:
        t |= {lab: (rows["labeler"] == lab).to_numpy() for lab in EXPERTS}
    return t


def same_sign(s: pd.Series) -> bool:
    s = s.dropna()
    return bool(len(s) > 1 and (np.all(s > 0) or np.all(s < 0)))


def screen_fold(split: str, i: int, seed) -> list[dict]:
    """One fold (worker process): every candidate's training-row statistics, per track."""
    cfg, a, out, proto_set = shared("cfg"), shared("args"), shared("out"), shared("proto_set")
    rows = shared("rows")
    rng = np.random.default_rng(seed)
    g, tr, te = fold_of(split, i)
    r = with_fold_prototypes(cfg, rows, tr, proto_set, a["prototypes"], out) if a["prototypes"] else rows
    cols = candidate_columns(r) + (base_columns(cfg, r) if a["with_base"] else [])
    orig = ~r["aug"].to_numpy(bool)
    y = r["rating"].astype(float)
    recs = []
    for track, m in tracks(r, a["per_labeler"]).items():
        trm, tem = m & tr & orig, m & te
        if not trm.any():
            continue
        sub = r[trm].sort_values(["student", "t"], kind="stable")     # each student's rows in time order
        ytr = demean(sub["rating"].astype(float), sub["student"]).to_numpy()
        yte = demean(y[tem], r.loc[tem, "student"]).to_numpy() if tem.any() else None
        Xtr = demean(sub[cols].astype(float), sub["student"]).reset_index(drop=True)
        Xte = demean(r.loc[tem, cols].astype(float), r.loc[tem, "student"]) if tem.any() else None
        perm_idx = shift_within(sub["student"].to_numpy(), a["perms"], rng)
        res = rank_corr_with_null(Xtr, ytr, perm_idx)
        q = dict(zip(res, bh_qvalues([v[2] for v in res.values()])))
        for c in cols:
            rho, n, pv = res[c]
            rho_te = spearman(Xte[c].to_numpy(), yte)[0] if Xte is not None else np.nan
            recs.append({"split": split, "fold": g, "candidate": c, "track": track, "n_train": n,
                         "rho_train": rho, "p_block_null": pv, "q": q[c], "rho_heldout": rho_te})
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
    ap = base_parser(__doc__, workers=True)
    ap.add_argument("--candidates", default=None, help="candidate folder (default: the newest under --out)")
    ap.add_argument("--prototypes", type=int, default=8, help="codebook size K per fold; 0 = no prototype features")
    ap.add_argument("--proto-set", default="subset", help="descriptor set for the codebook (falls back to all)")
    ap.add_argument("--with-base", action="store_true", help="screen the trk.* model inputs too, for reference")
    ap.add_argument("--per-labeler", action="store_true", help="also each expert labeler on their own")
    ap.add_argument("--perms", type=int, default=200, help="within-student circular shifts for the block null")
    split_parser_arg(ap)
    args = ap.parse_args()
    cfg, out = load(args)
    cdir = latest_candidates(out, args.candidates)
    rows = load_candidates(label_table(cfg), cdir)
    proto_set = args.proto_set if (out / "descriptor_subset.json").exists() or args.proto_set != "subset" else "all"
    import json
    miss = missingness_check(rows, candidate_columns(rows), cfg["seed"])
    (out / "feature_screen_missingness.json").write_text(json.dumps(miss, indent=1), encoding="utf-8")
    print("NaN-pattern check (does missingness predict the recording?):", miss, flush=True)

    a = {"prototypes": args.prototypes, "with_base": args.with_base, "per_labeler": args.per_labeler, "perms": args.perms}
    settings = {"config": config_hash(cfg), "candidates": cdir.name, "args": a, "proto_set": proto_set,
                "subset": subset_text(out), "rows": rows_key(cfg), "version": 2}
    recs = []
    for split, i, r in fold_jobs(screen_fold, rows, splits(args.split), args.workers, out, "feature_screen", settings,
                                 {"cfg": cfg, "args": a, "out": out, "proto_set": proto_set}, cfg["seed"]):
        recs += r
        print(f"  {split} fold {r[0]['fold'] if r else i}: done", flush=True)
    df = pd.DataFrame(recs)
    write_table(df, out / "feature_screen_folds")

    summary = df.groupby(["split", "candidate", "track"]).agg(
        rho_mean=("rho_train", "mean"), rho_sd=("rho_train", "std"), rho_min=("rho_train", "min"),
        rho_max=("rho_train", "max"), same_sign=("rho_train", same_sign),
        p_median=("p_block_null", "median"), q_median=("q", "median"),
        folds_q_below_05=("q", lambda v: float((v < 0.05).mean())),
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
                  f"{'same sign' if rr['same_sign'] else 'mixed'})  q median {rr['q_median']:.3f}, "
                  f"q<0.05 in {rr['folds_q_below_05']:.0%} of folds  self {sr:+.3f}")
    write_table(pd.concat(parts, ignore_index=True), out / "feature_screen")
    print(f"\nwrote {out}/feature_screen.csv/.json, feature_screen_folds.csv/.json, feature_screen_<split>.png")


if __name__ == "__main__":
    main()

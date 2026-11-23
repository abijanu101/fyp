"""How stable is each candidate clustering? Screen 6 of guide 15 §15.9.

  python -m scripts.clustering.stability [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--sets all subset head] [--channels hp.yaw rf.total] [--taus-desc 0.4 0.6 0.8 1.0 1.5]
         [--taus-sync 0.3 0.5 0.7 0.9] [--boot 10] [--max-windows 0] [--sessions <stem> ...]
         [--workers 1] [--out outputs/experiments/10_clustering]

For every regular-grid window with at least 4 visible students, every space (descriptor sets,
standardised per track; synchrony channels, correlation distance) and every method (agglomerative
at each threshold tau, HDBSCAN with min_cluster_size 2 and 3):

  subsample ARI  cluster 80% of the students again (--boot times) and compare with the full
                 clustering on the students both contain (adjusted Rand index; 1 = identical).
                 Subsampling, not the bootstrap, because duplicated students form fake clusters.
  temporal ARI   the clustering of the students two consecutive windows share (they overlap by
                 50%, so half their frames are the same): a stable grouping should mostly persist
  silhouette     where there are 2 to n-1 clusters
  degeneracy     the mean number of clusters per student, and the share of windows where everyone
                 is in one cluster, or where most students are alone (more clusters than n / 2, a
                 mean cluster size below 2): a setting that does either almost always gives
                 features with no variation, and a trivially high subsample ARI

With 7-20 students the silhouette is noisy; trust the two ARIs more (guide 15 R5).

Writes to --out:
  stability.csv   one row per (space, method, parameter): the measures above and `usable`
                  (not degenerate in more than 80% of windows)
  stability.png   subsample and temporal ARI against tau per space
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.clustering.common import (TrackStandardiser, available_channels, base_parser, cached_stems,
                                       descriptor_sets, load, load_session, map_sessions, plot_ready, series_matrix,
                                       write_table)
from src.cfmu.classrel import (MIN_CLUSTER_N, _rms_scale, agglomerative_labels, correlation_distance, hdbscan_labels,
                               pairwise_corr)


def ari(a, b) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def label(method: str, param: float, X=None, D=None) -> np.ndarray:
    if method == "agglomerative":
        return agglomerative_labels(param, X=X, D=D)
    lab, _ = hdbscan_labels(X=X, D=D, min_cluster_size=int(param))
    noise = lab < 0                                    # noise points are their own groups
    lab = lab.copy()
    lab[noise] = lab.max(initial=-1) + 1 + np.arange(noise.sum())
    return lab


def silhouette(lab: np.ndarray, X=None, D=None) -> float:
    from sklearn.metrics import silhouette_score

    k = len(np.unique(lab))
    if not 2 <= k <= len(lab) - 1:
        return np.nan
    if X is not None:
        return float(silhouette_score(_rms_scale(X), lab))
    return float(silhouette_score(D, lab, metric="precomputed"))


def subset(X, D, idx):
    return (X[idx] if X is not None else None), (D[np.ix_(idx, idx)] if D is not None else None)


def evaluate(spaces: dict, methods: list[tuple[str, str, float]], boot: int, rng) -> list[dict]:
    """spaces: {space: [(window index, tids, X or None, D or None), ...]} for one session."""
    recs = []
    for space, wins in spaces.items():
        kind = "descriptor" if space.startswith("desc:") else "synchrony"
        for mkind, method, param in methods:
            if mkind != kind:
                continue
            prev = None
            for wi, tids, X, D in wins:
                n = len(tids)
                lab = label(method, param, X, D)
                k = len(np.unique(lab))
                rec = {"space": space, "method": method, "param": param, "window": wi, "n": n,
                       "clusters_per_student": k / n, "one_cluster": float(k == 1), "mostly_alone": float(k > n / 2),
                       "silhouette": silhouette(lab, X, D)}
                scores = []
                for _ in range(boot):
                    idx = np.sort(rng.choice(n, size=max(MIN_CLUSTER_N, int(round(0.8 * n))), replace=False))
                    Xs, Ds = subset(X, D, idx)
                    scores.append(ari(lab[idx], label(method, param, Xs, Ds)))
                rec["subsample_ari"] = float(np.mean(scores)) if scores else np.nan
                rec["temporal_ari"] = np.nan
                if prev is not None and prev[0] == wi - 1:
                    common = sorted(set(prev[1]) & set(tids))
                    if len(common) >= MIN_CLUSTER_N:
                        pa = dict(zip(prev[1], prev[2]))
                        ca = dict(zip(tids, lab))
                        rec["temporal_ari"] = ari([pa[t] for t in common], [ca[t] for t in common])
                prev = (wi, tids, lab)
                recs.append(rec)
    return recs


def plot(summary: pd.DataFrame, path) -> None:
    plt = plot_ready()
    agg = summary[summary["method"] == "agglomerative"]
    spaces = sorted(agg["space"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, col, title in ((axes[0], "subsample_ari", "subsample ARI (80% of students)"),
                           (axes[1], "temporal_ari", "temporal ARI (consecutive windows)")):
        for s in spaces:
            d = agg[agg["space"] == s].sort_values("param")
            ax.plot(d["param"], d[col], marker="o", lw=1.5, label=s, alpha=0.9 if d["usable"].any() else 0.35)
        ax.set_xlabel("distance threshold tau")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.1, 1.05)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("adjusted Rand index")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Agglomerative clustering: stability against the threshold (faded = degenerate)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def stability_one(cfg: dict, stem: str, sets_wanted: list[str], channels, methods, boot: int, max_windows: int,
                  out, seed) -> list[dict]:
    """One session (worker process): the per-window records of every space and method."""
    from scripts.clustering.common import DEFAULT_CHANNELS

    rng = np.random.default_rng(seed)
    sd = load_session(cfg, stem)
    if sd.tracks.empty:
        return []
    sets = {k: v for k, v in descriptor_sets(list(sd.tracks.columns), out).items() if k in sets_wanted}
    chans = channels or [c for c in DEFAULT_CHANNELS if c in available_channels(sd)]
    stds = {k: TrackStandardiser(sd.tracks, v) for k, v in sets.items()}
    by_window = {int(w): g for w, g in sd.tracks.groupby("window")}
    step = 1 if not max_windows else max(1, len(sd.windows) // max_windows)
    picked = sorted({i for s in range(0, len(sd.windows), step) for i in (s, s + 1) if i < len(sd.windows)})
    spaces: dict[str, list] = {f"desc:{k}": [] for k in sets} | {f"sync:{c}": [] for c in chans}
    for wi in picked:                              # pairs of consecutive windows, for the temporal ARI
        w = sd.windows[wi]
        rows = by_window.get(wi)
        tracks = sd.visible(w)
        if rows is not None and len(rows) >= MIN_CLUSTER_N:
            tids = rows["tid"].astype(int).tolist()
            for k in sets:
                spaces[f"desc:{k}"].append((wi, tids, stds[k].transform(rows), None))
        if len(tracks) >= MIN_CLUSTER_N:
            tids = [tr.tid for tr in tracks]
            for c in chans:
                D = correlation_distance(pairwise_corr(series_matrix(sd, w, tracks, c)))
                spaces[f"sync:{c}"].append((wi, tids, None, D))
    return [{"stem": stem, **r} for r in evaluate(spaces, methods, boot, rng)]


def main() -> None:
    ap = base_parser(__doc__, workers=True)
    ap.add_argument("--sets", nargs="*", default=["all", "subset", "head"], help="descriptor sets (those present)")
    ap.add_argument("--channels", nargs="*", default=None, help="default: hp.yaw hp.head_height rf.total kg.speed")
    ap.add_argument("--taus-desc", nargs="*", type=float, default=[0.4, 0.6, 0.8, 1.0, 1.5])
    ap.add_argument("--taus-sync", nargs="*", type=float, default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--hdbscan", nargs="*", type=int, default=[2, 3], help="min_cluster_size values")
    ap.add_argument("--boot", type=int, default=10)
    ap.add_argument("--max-windows", type=int, default=0, help="per session, evenly spaced pairs of consecutive windows; 0 = all")
    args = ap.parse_args()
    cfg, out = load(args)

    methods = [("descriptor", "agglomerative", t) for t in args.taus_desc] +               [("synchrony", "agglomerative", t) for t in args.taus_sync] +               [(k, "hdbscan", m) for k in ("descriptor", "synchrony") for m in args.hdbscan]
    stems = cached_stems(cfg, args.sessions)
    jobs = [(cfg, stem, args.sets, args.channels, methods, args.boot, args.max_windows, out, [cfg["seed"], i])
            for i, stem in enumerate(stems)]
    recs = []
    for i, (job, r) in enumerate(map_sessions(stability_one, jobs, args.workers), 1):
        recs += r
        print(f"  [{i}/{len(stems)}] {job[1]}", flush=True)
    if not recs:
        raise SystemExit("no window with at least 4 visible students")
    df = pd.DataFrame(recs)
    summary = df.groupby(["space", "method", "param"]).agg(
        windows=("window", "size"), subsample_ari=("subsample_ari", "mean"), temporal_ari=("temporal_ari", "mean"),
        silhouette=("silhouette", "mean"), clusters_per_student=("clusters_per_student", "mean"),
        one_cluster=("one_cluster", "mean"), mostly_alone=("mostly_alone", "mean")).reset_index()
    summary["usable"] = (summary["one_cluster"] < 0.8) & (summary["mostly_alone"] < 0.8)
    summary = summary.sort_values(["usable", "subsample_ari"], ascending=[False, False])
    write_table(summary, out / "stability")
    plot(summary, out / "stability.png")
    print("\n" + summary.round(3).to_string(index=False))
    best = summary[summary["usable"]].head(5)
    if len(best):
        print("\nmost stable usable settings:")
        for _, r in best.iterrows():
            print(f"  {r['space']:22s} {r['method']:13s} {r['param']:<5g} subsample ARI {r['subsample_ari']:.3f}  "
                  f"temporal {r['temporal_ari']:.3f}  clusters/student {r['clusters_per_student']:.2f}")
    print(f"\nwrote {out}/stability.csv, stability.png")


if __name__ == "__main__":
    main()

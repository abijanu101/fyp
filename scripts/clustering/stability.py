"""How stable is each candidate clustering? Screen 6 of guide 15 §15.9.

  python -m scripts.clustering.stability [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--sets all subset head] [--channels hp.yaw rf.total] [--taus-desc 0.4 0.6 0.8 1.0 1.5 2.0 3.0]
         [--taus-sync 0.3 0.5 0.7 0.9 1.1 1.3] [--null-gap 10] [--boot 10] [--max-windows 0] [--sessions <stem> ...]
         [--workers 1] [--out outputs/experiments/10_clustering]

Resumable: each recording's result is saved under <out>/partial/stability/<settings hash>/ as it
finishes; a rerun with the same settings only computes the rest.

For every regular-grid window with at least 4 visible students, every space (descriptor sets,
standardised per track; synchrony channels, correlation distance) and every method (agglomerative
at each threshold tau, HDBSCAN with min_cluster_size 2 and 3):

  subsample ARI  cluster 80% of the students again (--boot times) and compare with the full
                 clustering on the students both contain (adjusted Rand index; 1 = identical).
                 Subsampling, not the bootstrap, because duplicated students form fake clusters.
  temporal ARI   the clustering of the students two consecutive windows share (they overlap by
                 50%, so half their frames are the same): a stable grouping should mostly persist
  null ARI       the same comparison against a random window of the same recording at least
                 --null-gap windows away (20 s at the default grid): what any clustering of these
                 students scores by chance, including the trivial agreement of coarse clusterings
                 (everyone in one cluster in both windows gives ARI 1.0)
  temporal excess  temporal ARI minus null ARI: how much more a grouping persists into the next
                 window than into an unrelated one. This, not the raw temporal ARI, is what tau is
                 chosen on: the raw value rises mechanically as clusters get coarser
  silhouette     where there are 2 to n-1 clusters
  degeneracy     the mean number of clusters per student, and the share of windows where everyone
                 is in one cluster, or where most students are alone (more clusters than n / 2, a
                 mean cluster size below 2): a setting that does either almost always gives
                 features with no variation, and a trivially high subsample ARI

With 7-20 students the silhouette is noisy; trust the two ARIs more (guide 15 R5).

Choosing tau: among usable agglomerative rows, the highest temporal excess (descriptor space on its
own; synchrony space averaged over the channels). If the best tau is the largest one tested, the
optimum may lie beyond the grid: extend --taus-desc / --taus-sync and rerun.

Writes to --out (tables as .csv and .json):
  stability       one row per (space, method, parameter): the measures above and `usable`
                  (not degenerate in more than 80% of windows)
  stability.png   subsample and temporal ARI against tau per space
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.clustering.common import (TrackStandardiser, available_channels, base_parser, cached_stems,
                                       descriptor_sets, load, load_session, map_sessions, partial_dir, plot_ready,
                                       series_matrix, write_table)
from scripts.run_pipeline import features_dir
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


def null_ari(done: list[tuple], gap: int, rng) -> list[float]:
    """For each (window, tids, labels) of one session, the ARI against a random window at least `gap`
    windows away, on the students both contain (NaN when none is far enough or too few are shared)."""
    out = []
    for wi, tids, lab in done:
        far = [d for d in done if abs(d[0] - wi) >= gap]
        if not far:
            out.append(np.nan)
            continue
        wj, tj, lj = far[int(rng.integers(len(far)))]
        common = sorted(set(tids) & set(tj))
        if len(common) < MIN_CLUSTER_N:
            out.append(np.nan)
            continue
        a, b = dict(zip(tids, lab)), dict(zip(tj, lj))
        out.append(ari([a[t] for t in common], [b[t] for t in common]))
    return out


def evaluate(spaces: dict, methods: list[tuple[str, str, float]], boot: int, rng, gap: int = 10) -> list[dict]:
    """spaces: {space: [(window index, tids, X or None, D or None), ...]} for one session."""
    recs = []
    for space, wins in spaces.items():
        kind = "descriptor" if space.startswith("desc:") else "synchrony"
        for mkind, method, param in methods:
            if mkind != kind:
                continue
            prev, done, first = None, [], len(recs)
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
                done.append((wi, tids, lab))
                recs.append(rec)
            for rec, v in zip(recs[first:], null_ari(done, gap, rng)):
                rec["null_ari"] = v
    return recs


def plot(summary: pd.DataFrame, path) -> None:
    plt = plot_ready()
    agg = summary[summary["method"] == "agglomerative"]
    spaces = sorted(agg["space"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, col, title in ((axes[0], "subsample_ari", "subsample ARI (80% of students)"),
                           (axes[1], "temporal_excess", "temporal ARI minus null (consecutive vs distant windows)")):
        for s in spaces:
            d = agg[agg["space"] == s].sort_values("param")
            ax.plot(d["param"], d[col], marker="o", lw=1.5, label=s, alpha=0.9 if d["usable"].any() else 0.35)
        ax.set_xlabel("distance threshold tau")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.1 if col == "subsample_ari" else None, 1.05 if col == "subsample_ari" else None)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("adjusted Rand index")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Agglomerative clustering: stability against the threshold (faded = degenerate)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def stability_one(cfg: dict, stem: str, sets_wanted: list[str], channels, methods, boot: int, max_windows: int,
                  out, seed, gap: int = 10) -> list[dict]:
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
    return [{"stem": stem, **r} for r in evaluate(spaces, methods, boot, rng, gap)]


def main() -> None:
    ap = base_parser(__doc__, workers=True)
    ap.add_argument("--sets", nargs="*", default=["all", "subset", "head"], help="descriptor sets (those present)")
    ap.add_argument("--channels", nargs="*", default=None, help="default: hp.yaw hp.head_height rf.total kg.speed")
    ap.add_argument("--taus-desc", nargs="*", type=float, default=[0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--taus-sync", nargs="*", type=float, default=[0.3, 0.5, 0.7, 0.9, 1.1, 1.3])
    ap.add_argument("--null-gap", type=int, default=10, help="windows between a window and its null partner")
    ap.add_argument("--hdbscan", nargs="*", type=int, default=[2, 3], help="min_cluster_size values")
    ap.add_argument("--boot", type=int, default=10)
    ap.add_argument("--max-windows", type=int, default=0, help="per session, evenly spaced pairs of consecutive windows; 0 = all")
    args = ap.parse_args()
    cfg, out = load(args)

    methods = [("descriptor", "agglomerative", t) for t in args.taus_desc] +               [("synchrony", "agglomerative", t) for t in args.taus_sync] +               [(k, "hdbscan", m) for k in ("descriptor", "synchrony") for m in args.hdbscan]
    stems = cached_stems(cfg, args.sessions)
    jobs = [(cfg, stem, args.sets, args.channels, methods, args.boot, args.max_windows, out, [cfg["seed"], i],
             args.null_gap) for i, stem in enumerate(stems)]
    subset = out / "descriptor_subset.json"
    resume = partial_dir(out, "stability", {"features": features_dir(cfg).name, "sets": args.sets,
                                            "channels": args.channels, "methods": methods, "boot": args.boot,
                                            "max_windows": args.max_windows, "gap": args.null_gap, "seed": cfg["seed"],
                                            "subset": subset.read_text(encoding="utf-8") if subset.exists() else None,
                                            "version": 2})
    recs = []
    for i, (job, r) in enumerate(map_sessions(stability_one, jobs, args.workers, stems, resume), 1):
        recs += r
        print(f"  [{i}/{len(stems)}] {job[1]}", flush=True)
    if not recs:
        raise SystemExit("no window with at least 4 visible students")
    df = pd.DataFrame(recs)
    summary = df.groupby(["space", "method", "param"]).agg(
        windows=("window", "size"), subsample_ari=("subsample_ari", "mean"), temporal_ari=("temporal_ari", "mean"),
        null_ari=("null_ari", "mean"), silhouette=("silhouette", "mean"),
        clusters_per_student=("clusters_per_student", "mean"),
        one_cluster=("one_cluster", "mean"), mostly_alone=("mostly_alone", "mean")).reset_index()
    summary["temporal_excess"] = summary["temporal_ari"] - summary["null_ari"]
    summary["usable"] = (summary["one_cluster"] < 0.8) & (summary["mostly_alone"] < 0.8)
    summary = summary.sort_values(["usable", "temporal_excess"], ascending=[False, False])
    write_table(summary, out / "stability")
    plot(summary, out / "stability.png")
    print("\n" + summary.round(3).to_string(index=False))
    agg = summary[summary["usable"] & (summary["method"] == "agglomerative")]
    desc = agg[agg["space"] == "desc:subset"] if (agg["space"] == "desc:subset").any() else agg[agg["space"].str.startswith("desc:")]
    sync = agg[agg["space"].str.startswith("sync:")].groupby("param")["temporal_excess"].mean().sort_values(ascending=False)
    print("\nchoice of tau (highest temporal excess = temporal ARI - null ARI, usable agglomerative settings):")
    if len(desc):
        d = desc.sort_values("temporal_excess", ascending=False).iloc[0]
        edge = " (largest tested: extend --taus-desc)" if d["param"] == max(args.taus_desc) else ""
        print(f"  tau-desc {d['param']:g}: excess {d['temporal_excess']:.3f} ({d['space']}){edge}")
    if len(sync):
        edge = " (largest tested: extend --taus-sync)" if sync.index[0] == max(args.taus_sync) else ""
        print(f"  tau-sync {sync.index[0]:g}: mean excess over channels {sync.iloc[0]:.3f}{edge}")
    print(f"\nwrote {out}/stability (.csv/.json), stability.png")


if __name__ == "__main__":
    main()

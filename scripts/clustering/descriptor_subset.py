"""Which window descriptors to cluster on, chosen without labels: screen 5 of guide 15 §15.9.

  python -m scripts.clustering.descriptor_subset [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--size 10] [--redundancy 0.9] [--min-within 0.3] [--sample 4000] [--sessions <stem> ...]
         [--out outputs/experiments/10_clustering]

Rows: the regular-grid per-student rows (<stem>.tracks.csv) of the original recordings, every
trk.* descriptor standardised per track over its session (R6), so what is left is how a student's
window differs from their own usual, not their seat.

  1 drop      descriptors that are mostly NaN (> 50%) or flat after standardisation, and, when
              signal_screen_descriptors.csv is in --out, those whose within-track share of the
              variance is below --min-within (screen 2: they describe the seat, not the moment)
  2 Laplacian score (He, Cai and Niyogi 2005) of each descriptor on a k-nearest-neighbour graph of
              the rows: low = it preserves the neighbourhood structure, which is what a clustering
              uses; no labels
  3 redundancy groups of descriptors with |Spearman rho| above --redundancy (average linkage on
              1 - |rho|): only the lowest-scoring member of each group is kept
  4 PCA       each descriptor's share of the components covering 90% of the variance (a check,
              not a criterion: experiment 08 found PCA itself does not help the decoders)

Writes to --out:
  descriptor_subset.csv   per descriptor: NaN share, within-track share (if screened), Laplacian
                          score and rank, redundancy group, PCA share, kept or why not
  descriptor_subset.json  {"subset": [...], "params": {...}}: read by stability, candidates and
                          ablation as the descriptor set `subset`
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.clustering.common import (TrackStandardiser, base_parser, cached_stems, descriptor_sets, load,
                                       load_session)
from src.cfmu.classrel import laplacian_score


def redundancy_groups(Z: pd.DataFrame, threshold: float) -> pd.Series:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if Z.shape[1] == 1:
        return pd.Series([0], index=Z.columns)
    rho = Z.rank().corr().abs().fillna(0).to_numpy()
    D = np.clip(1 - rho, 0, 1)
    np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), "average"), t=1 - threshold, criterion="distance")
    return pd.Series(lab - 1, index=Z.columns)


def pca_share(Z: np.ndarray, cover: float = 0.9) -> np.ndarray:
    Zc = Z - Z.mean(axis=0)
    _, s, vt = np.linalg.svd(Zc, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    k = int(np.searchsorted(np.cumsum(ev), cover) + 1)
    w = (vt[:k] ** 2 * ev[:k, None]).sum(axis=0)
    return w / w.sum()


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--set-name", default="all", help="descriptor set to choose from (common.descriptor_sets)")
    ap.add_argument("--size", type=int, default=10, help="most descriptors to keep")
    ap.add_argument("--redundancy", type=float, default=0.9)
    ap.add_argument("--min-within", type=float, default=0.3)
    ap.add_argument("--sample", type=int, default=4000, help="rows for the Laplacian score's neighbour graph")
    ap.add_argument("--neighbours", type=int, default=5)
    args = ap.parse_args()
    cfg, out = load(args)
    rng = np.random.default_rng(cfg["seed"])

    parts = []
    for stem in cached_stems(cfg, args.sessions):
        sd = load_session(cfg, stem)
        if sd.tracks.empty:
            continue
        cols = descriptor_sets(list(sd.tracks.columns))[args.set_name]
        Z = TrackStandardiser(sd.tracks, cols).transform(sd.tracks)
        parts.append(pd.DataFrame(Z, columns=cols))
    if not parts:
        raise SystemExit("no per-student rows (tracks.csv) in the cache")
    Z = pd.concat(parts, ignore_index=True)
    table = pd.DataFrame({"column": Z.columns, "nan_share": Z.isna().mean().to_numpy(),
                          "sd": Z.std(ddof=0).to_numpy()}).set_index("column")
    table["status"] = ""
    table.loc[table["nan_share"] > 0.5, "status"] = "mostly NaN"
    table.loc[(table["status"] == "") & ~(table["sd"] > 1e-9), "status"] = "flat"
    screen = out / "signal_screen_descriptors.csv"
    if screen.exists():
        w = pd.read_csv(screen).set_index("column")["within_track_share"]
        table["within_track_share"] = w.reindex(table.index)
        table.loc[(table["status"] == "") & (table["within_track_share"] < args.min_within), "status"] = "seat (within share low)"
    live = table.index[table["status"] == ""].tolist()
    if not live:
        raise SystemExit("every descriptor was dropped; lower --min-within or check the cache")
    X = Z[live].fillna(0.0).to_numpy()               # 0 = the track's own mean after standardisation
    idx = rng.choice(len(X), size=min(args.sample, len(X)), replace=False)
    table.loc[live, "laplacian"] = laplacian_score(X[idx], k=args.neighbours)
    table["laplacian_rank"] = table["laplacian"].rank()
    table.loc[live, "redundancy_group"] = redundancy_groups(Z[live].iloc[idx], args.redundancy)
    table.loc[live, "pca_share"] = pca_share(X[idx])
    best = table.loc[live].sort_values("laplacian").groupby("redundancy_group").head(1).index
    for c in live:
        if c not in best:
            table.loc[c, "status"] = "redundant"
    kept = table.loc[best].sort_values("laplacian").index[: args.size].tolist()
    for c in best:
        table.loc[c, "status"] = "kept" if c in kept else "beyond --size"
    table = table.sort_values(["laplacian"], na_position="last")
    table.reset_index().to_csv(out / "descriptor_subset.csv", index=False)
    params = {k: getattr(args, k) for k in ("set_name", "size", "redundancy", "min_within", "sample", "neighbours")}
    params["screened"] = screen.exists()
    (out / "descriptor_subset.json").write_text(json.dumps({"subset": kept, "params": params,
                    "scores": json.loads(table.reset_index().to_json(orient="records", double_precision=6))}, indent=1), encoding="utf-8")

    print(f"{len(Z)} student-window rows, {Z.shape[1]} descriptors, {len(live)} live, "
          f"{int(table['redundancy_group'].nunique())} redundancy groups")
    print(table[["status", "laplacian", "pca_share"] + (["within_track_share"] if screen.exists() else [])]
          .round(3).to_string())
    print(f"\nsubset ({len(kept)}): {', '.join(kept)}")
    print(f"wrote {out}/descriptor_subset.csv, descriptor_subset.json")


if __name__ == "__main__":
    main()

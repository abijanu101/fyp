"""Compute every candidate class-relative feature on the windows centred on rating entries: task 7
of guide 15 §15.9 (options 0, 1 and 2 of §15.4; option 4, the prototype codebook, is fitted per
fold by feature_screen, importance and ablation, because it is fitted on data).

  python -m scripts.clustering.candidates [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--channels hp.yaw hp.head_height rf.total kg.speed] [--set-name subset]
         [--tau-desc 0.8] [--tau-sync 0.6] [--hdb-min 3] [--max-lag-s 1.0] [--force]
         [--format json] [--workers 1] [--out outputs/experiments/10_clustering]

Builds the label rows exactly as train_models does (which caches <stem>.labelwin.csv), so the
candidates line up with the rows the models train on, augmented copies included. Per window centre
and per visible student (columns trk.cluster.*, per student; cluster.*, per window, repeated on its
students; the names guide 15 §15.7 fixes for the CFMU):

  sync.<channel>.{corr, lag, lag_corr, degree}        option 0, synchrony space (lag in seconds,
                                                      positive = after the class)
  sync.<channel>.{mean_corr, share_synced}  (window)
  sync.desc.dev                                       option 0, descriptor space: deviation from
                                                      the class (robust, per-track standardised)
  agg.desc.{share, singleton, in_largest, centroid_dist}   option 1 on the descriptors (Ward, tau)
  agg.desc.{n_clusters, largest_share, chaos, fragmentation}  (window)
  agg.<channel>.{share, singleton, in_largest, centroid_dist}  option 1 in synchrony space
  agg.<channel>.{n_clusters, largest_share}  (window)            (average linkage on 1 - r)
  hdb.desc.{share, singleton, strength, ...}, hdb.desc.noise_share (window)   option 2

Pick tau with stability.py first. Written to <out>/candidates/<feature hash>-<settings hash>/,
one file per session (originals and augmented copies), with the settings in params.json:
  --format json (default)  <stem>.json: {"session", "feature_hash", "params", "per_student_features",
                           "per_window_features", "windows": [{"t_centre", "window": [values],
                           "students": [{"tid", "values": [values]}]}]}: the names are listed once,
                           values follow their order, 6 significant digits, NaN is null
  --format csv             <stem>.csv: one flat row per (t_centre, tid)
  --format both            both
Existing files are reused unless --force, so an interrupted run resumes. --workers N computes N
sessions at once (CPU processes).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.clustering.common import (DEFAULT_CHANNELS, PER_STUDENT, PER_WINDOW, SD_FLOOR, TrackStandardiser,
                                       available_channels, base_parser, candidate_dir, candidates_to_json,
                                       descriptor_sets, full_tracks, label_table, load, load_session,
                                       map_sessions, series_matrix)
from src.aggregation import frames_in
from scripts.run_pipeline import features_dir
from src.cfmu.classrel import (MIN_COMMON, agglomerative_labels, cluster_features, correlation_distance,
                               deviation_from_class, hdbscan_labels, pairwise_corr, sync_features)
from src.core.types import Window


def window_candidates(sd, w: Window, tracks: list, X: np.ndarray, channels: list[str], p: dict) -> list[dict]:
    """Every candidate for the students of one window, in the order of `tracks`; X: their
    standardised descriptors (n, D), NaN rows where a student has none."""
    per: dict[str, np.ndarray] = {}
    win: dict[str, float] = {}

    def put(prefix: str, s: dict, wd: dict):
        per.update({f"{PER_STUDENT}{prefix}.{k}": v for k, v in s.items()})
        win.update({f"{PER_WINDOW}{prefix}.{k}": v for k, v in wd.items()})

    max_lag = max(1, int(round(p["max_lag_s"] * sd.fps)))
    # synchrony, lag and synchrony clustering over the context ending with the window (a 4 s window
    # has ~20 autocorrelated frames: too few for a per-student correlation or a seconds-scale lag)
    ctx = Window(w.t_end - max(p["context_s"], w.t_end - w.t_start), w.t_end)
    ctx_tracks = full_tracks(sd, [tr.tid for tr in tracks])
    n_frames = len(frames_in(sd.frame_t, ctx))
    min_common = max(MIN_COMMON, int(np.ceil(p["min_common_share"] * n_frames)))
    for ch in channels:
        S = series_matrix(sd, ctx, ctx_tracks, ch)
        s, wd = sync_features(S, max_lag=max_lag, min_common=min_common, threshold=p["sync_threshold"])
        s["lag"] = s["lag"] / sd.fps
        put(f"sync.{ch}", s, wd)
        D = correlation_distance(pairwise_corr(S, min_common))
        s, wd = cluster_features(agglomerative_labels(p["tau_sync"], D=D), D=D)
        put(f"agg.{ch}", s, {k: wd[k] for k in ("n_clusters", "largest_share")})
    per[f"{PER_STUDENT}sync.desc.dev"] = deviation_from_class(X)
    s, wd = cluster_features(agglomerative_labels(p["tau_desc"], X=X), X=X)
    put("agg.desc", s, wd)
    lab, strength = hdbscan_labels(X=X, min_cluster_size=p["hdb_min"])
    s, wd = cluster_features(lab, X=X, strength=strength)
    put("hdb.desc", s, {k: wd[k] for k in ("n_clusters", "largest_share", "noise_share")})
    return [{"tid": int(tr.tid), **{k: float(v[i]) for k, v in per.items()}, **win} for i, tr in enumerate(tracks)]


def session_candidates(cfg: dict, stem: str, channels: list[str] | None, p: dict, out) -> pd.DataFrame:
    sd = load_session(cfg, stem)
    lw = sd.labelwin(cfg)
    if lw is None or lw.empty:
        return pd.DataFrame()
    grid = sd.tracks if not sd.tracks.empty else lw
    cols = descriptor_sets(list(grid.columns), out)[p["set_name"]]
    std = TrackStandardiser(grid, cols)
    chans = channels or [c for c in DEFAULT_CHANNELS if c in available_channels(sd)]
    length = float(cfg["window"]["length_s"])
    recs = []
    for c, g in lw.groupby(lw["t_centre"].round(3)):
        centre = float(g["t_centre"].iloc[0])
        w = Window(centre - length / 2, centre + length / 2)
        tracks = sd.visible(w)
        if not tracks:
            continue
        Xg = pd.DataFrame(std.transform(g), index=g["tid"].astype(int).to_numpy())
        Xg = Xg[~Xg.index.duplicated()]
        X = Xg.reindex([tr.tid for tr in tracks]).to_numpy(float)
        recs += [{"stem": stem, "t_centre": centre, **r} for r in window_candidates(sd, w, tracks, X, chans, p)]
    return pd.DataFrame(recs)


def write_session(cfg: dict, stem: str, channels, p: dict, out, cdir, fmt: str) -> int:
    """Compute one session's candidates and write them (temporary file, then rename)."""
    df = session_candidates(cfg, stem, channels, p, out)
    if fmt in ("csv", "both"):
        tmp = cdir / f"{stem}.tmp.csv"
        df.to_csv(tmp, index=False)
        tmp.replace(cdir / f"{stem}.csv")
    if fmt in ("json", "both"):
        meta = {"feature_hash": features_dir(cfg).name, "params": p}
        tmp = cdir / f"{stem}.tmp.json"
        tmp.write_text(json.dumps(candidates_to_json(df, stem, meta), separators=(",", ":")), encoding="utf-8")
        tmp.replace(cdir / f"{stem}.json")
    return len(df)


def main() -> None:
    ap = base_parser(__doc__, workers=True)
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--set-name", default=None, help="descriptor set (default: subset if descriptor_subset.json exists, else all)")
    ap.add_argument("--tau-desc", type=float, default=0.8)
    ap.add_argument("--tau-sync", type=float, default=0.6)
    ap.add_argument("--hdb-min", type=int, default=3)
    ap.add_argument("--max-lag-s", type=float, default=3.0, help="lag range for synchrony, seconds either way")
    ap.add_argument("--context-s", type=float, default=24.0,
                    help="synchrony and lag are computed over this many seconds ending with the window")
    ap.add_argument("--min-common-share", type=float, default=0.5,
                    help="two students' series must share at least this share of the context's frames")
    ap.add_argument("--sync-threshold", type=float, default=0.3)
    ap.add_argument("--format", choices=["json", "csv", "both"], default="json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg, out = load(args)
    set_name = args.set_name or ("subset" if (out / "descriptor_subset.json").exists() else "all")
    p = {"channels": args.channels, "set_name": set_name, "tau_desc": args.tau_desc, "tau_sync": args.tau_sync,
         "hdb_min": args.hdb_min, "max_lag_s": args.max_lag_s, "sync_threshold": args.sync_threshold,
         "context_s": args.context_s, "min_common_share": args.min_common_share, "sd_floor": SD_FLOOR}
    if set_name == "subset":
        p["subset"] = json.loads((out / "descriptor_subset.json").read_text(encoding="utf-8"))["subset"]
    cdir = candidate_dir(cfg, out, p)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "params.json").write_text(json.dumps(p, indent=2), encoding="utf-8")

    rows = label_table(cfg)                   # builds every session's <stem>.labelwin.csv first
    stems = sorted(rows["stem"].unique())
    if args.sessions:
        stems = [s for s in stems if s.split(".")[0] in set(args.sessions)]
    exts = {"json": [".json"], "csv": [".csv"], "both": [".json", ".csv"]}[args.format]
    todo = [s for s in stems if args.force or not all((cdir / f"{s}{e}").exists() for e in exts)]
    print(f"{len(stems)} sessions ({len(stems) - len(todo)} already done), {args.workers} worker(s) -> {cdir}", flush=True)
    jobs = [(cfg, s, args.channels, p, out, cdir, args.format) for s in todo]
    n_rows = 0
    for i, (job, n) in enumerate(map_sessions(write_session, jobs, args.workers), 1):
        n_rows += n
        print(f"  [{i}/{len(todo)}] {job[1]}: {n} student rows", flush=True)
    print(f"\nwrote {cdir} ({len(todo)} sessions, {n_rows} new rows); settings in params.json")


if __name__ == "__main__":
    main()

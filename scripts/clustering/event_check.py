"""Do the candidate channels move together when the class should react together? Screen 4 of
guide 15 §15.9.

  python -m scripts.clustering.event_check [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--channels hp.yaw rf.total ...] [--events 36012.4 36090 ...] [--n-events 10]
         [--response-s 1.0] [--sessions <stem> ...] [--workers 1] [--out outputs/experiments/10_clustering]

Parallel (--workers) and resumable: each recording's result is saved under
<out>/partial/event_check/<settings hash>/ as it finishes.

DIPSER has no annotated class events (a recording is one activity), so events come from the data,
without circularity: the students of each session are split at random into two halves; events are
the largest jumps of half A's median signal (z-scored per student, peaks at least --min-sep-s
apart); the test is whether half B moves at those moments, against random moments. With --events
(seconds since midnight, the camera clock, e.g. moments found in label_viewer / output_viewer:
the teacher pointing, everyone starting to write), every student is tested at those moments.

A student "moves" at a moment when their own frame-to-frame change exceeds their own 90th
percentile within --response-s of it.

Camera check: a camera bump or lighting change moves every student at once and would pass this
test without any student reacting to another. An event is flagged as camera-like when the
whole-picture motion (scene feature `motion`) jumps above its own 90th percentile within
--response-s of it; the test is repeated on the events that are not camera-like (`*_nocam`).

Statistics: each recording counts once. Across recordings, the ratio minus 1 is tested with a
one-sided Wilcoxon signed-rank test, then Benjamini-Hochberg across the channels.

Writes to --out (tables as .csv and .json):
  event_check              per (session, channel): events, the share of half B moving at events and
                           at random moments, their ratio and an empirical p-value; the same without
                           camera-like events
  event_check_channels     per channel: the across-recording tests (p, q), with and without
                           camera-like events
  event_times.csv          every event (seconds since midnight and since the recording's start),
                           to look up in the viewers
  event_check_<channel>.png  per channel, for the first session: the students x time raster
                           (z-scored) with the events marked, and the event-locked mean of half B
                           against the random-moment band
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path

from scripts.clustering.common import (DEFAULT_CHANNELS, available_channels, base_parser, bh_qvalues,
                                       cached_stems, load, load_session, map_sessions, partial_dir,
                                       per_recording_test, plot_ready, series_matrix, write_table)
from scripts.run_pipeline import features_dir
from src.core.types import Window


def session_matrix(sd, name: str) -> np.ndarray:
    """(tracks, frames) the channel over the whole session, z-scored per student."""
    w = Window(sd.frame_t[0], sd.frame_t[-1] + 1e-6)
    S = series_matrix(sd, w, sd.tracklets, name)
    with np.errstate(invalid="ignore"):
        mu, sdv = np.nanmean(S, axis=1, keepdims=True), np.nanstd(S, axis=1, keepdims=True)
        return (S - mu) / np.where(sdv > 1e-12, sdv, np.nan)


def smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return x
    out = np.convolve(np.nan_to_num(x), np.ones(k) / k, mode="same")
    return np.where(np.isfinite(x), out, np.nan)


def pick_events(Z: np.ndarray, n_events: int, min_sep: int) -> np.ndarray:
    """Frames of the largest jumps of the median series, at least min_sep frames apart."""
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(Z).sum(axis=0) >= max(2, Z.shape[0] // 2)
        med = np.where(ok, np.nanmedian(np.where(np.isfinite(Z), Z, np.nan), axis=0), np.nan)
    jump = np.abs(np.diff(smooth(med)))
    jump = np.nan_to_num(jump, nan=-1)
    chosen = []
    for f in np.argsort(jump)[::-1]:
        if jump[f] <= 0 or len(chosen) >= n_events:
            break
        if all(abs(f - c) >= min_sep for c in chosen):
            chosen.append(int(f) + 1)
    return np.array(sorted(chosen), int)


def moving(Z: np.ndarray, q: float = 0.9) -> np.ndarray:
    """(tracks, frames) True where a student's own |change| exceeds their own q-quantile."""
    d = np.abs(np.diff(Z, axis=1, prepend=np.nan))
    with np.errstate(invalid="ignore"):
        thr = np.nanquantile(np.where(np.isfinite(d), d, np.nan), q, axis=1, keepdims=True)
    return d > thr


def response_share(M: np.ndarray, present: np.ndarray, frames: np.ndarray, half: int) -> np.ndarray:
    """Per moment: the share of students present around it who move within +-half frames."""
    out = []
    T = M.shape[1]
    for f in frames:
        a, b = max(0, f - half), min(T, f + half + 1)
        here = present[:, a:b].any(axis=1)
        if here.sum() < 2:
            out.append(np.nan)
            continue
        out.append(float(M[here, a:b].any(axis=1).mean()))
    return np.array(out)


def check(Z: np.ndarray, events: np.ndarray, half: int, n_null: int, rng, margin: int) -> dict:
    M = moving(Z)
    present = np.isfinite(Z)
    at = response_share(M, present, events, half)
    null_frames = rng.integers(margin, Z.shape[1] - margin, size=n_null) if Z.shape[1] > 2 * margin else np.zeros(0, int)
    null = response_share(M, present, null_frames, half)
    at_m, null_m = np.nanmean(at) if np.isfinite(at).any() else np.nan, np.nanmean(null) if np.isfinite(null).any() else np.nan
    # empirical p: how often a random set of as many moments responds at least as strongly
    k = int(np.isfinite(at).sum())
    nv = null[np.isfinite(null)]
    if k and len(nv) >= k:
        draws = np.array([rng.choice(nv, size=k, replace=False).mean() for _ in range(500)])
        p = float((draws >= at_m).mean())
    else:
        p = np.nan
    return {"n_events": int(len(events)), "share_at_events": at_m, "share_random": null_m,
            "ratio": at_m / null_m if null_m else np.nan, "p": p}


def locked(Z: np.ndarray, frames: np.ndarray, half: int) -> np.ndarray:
    """Event-locked mean |change| of the class median, frames -half..+half (NaN-padded)."""
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.where(np.isfinite(Z), Z, np.nan), axis=0)
    d = np.abs(np.diff(med, prepend=np.nan))
    segs = []
    for f in frames:
        seg = np.full(2 * half + 1, np.nan)
        a, b = f - half, f + half + 1
        lo, hi = max(a, 0), min(b, len(d))
        seg[lo - a: hi - a] = d[lo:hi]
        segs.append(seg)
    with np.errstate(invalid="ignore"):
        return np.nanmean(segs, axis=0) if segs else np.full(2 * half + 1, np.nan)


def plot(sd, ch: str, Z: np.ndarray, B: np.ndarray, events: np.ndarray, half: int, rng, path) -> None:
    plt = plot_ready()
    t0 = sd.frame_t[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [3, 1]})
    order = np.argsort([np.flatnonzero(np.isfinite(z))[0] if np.isfinite(z).any() else 0 for z in Z])
    a.imshow(np.clip(Z[order], -3, 3), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest",
             extent=(0, sd.frame_t[-1] - t0, len(Z), 0))
    for f in events:
        a.axvline(sd.frame_t[f] - t0, color="black", lw=0.8, alpha=0.6)
    a.set_xlabel("seconds since the recording's first frame")
    a.set_ylabel("student (track), by first appearance")
    a.set_title(f"{sd.stem}: {ch}, z-scored per student; lines = events", fontsize=9)
    lags = (np.arange(-half, half + 1)) / sd.fps
    b.plot(lags, locked(Z[B], events, half), color="#2a78d6", lw=2, label="half B at events")
    nulls = np.array([locked(Z[B], rng.integers(half, Z.shape[1] - half, size=max(len(events), 1)), half)
                      for _ in range(100)]) if Z.shape[1] > 2 * half else np.zeros((0, len(lags)))
    if len(nulls):
        b.fill_between(lags, np.nanquantile(nulls, 0.05, axis=0), np.nanquantile(nulls, 0.95, axis=0),
                       color="#cccccc", label="random moments (90%)")
    b.axvline(0, color="#999", lw=0.8, ls=":")
    b.set_xlabel("seconds from the event")
    b.set_ylabel("|change| of the class median")
    b.legend(fontsize=8)
    for ax in (a, b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def camera_like(sd, events: np.ndarray, half: int) -> np.ndarray:
    """True for events within +-half frames of a jump in whole-picture motion above its 90th percentile."""
    g = sd.scene_motion
    if not np.isfinite(g).any():
        return np.zeros(len(events), bool)
    jump = np.abs(np.diff(smooth(g), prepend=np.nan))
    thr = np.nanquantile(jump, 0.9)
    out = []
    for f in events:
        a, b = max(0, f - half), min(len(jump), f + half + 1)
        seg = jump[a:b]
        out.append(bool(np.isfinite(seg).any() and np.nanmax(seg) > thr))
    return np.array(out, bool)


def event_one(cfg: dict, stem: str, channels, a: dict, seed, plot_dir) -> tuple[list[dict], list[dict], list[str]]:
    """One recording (worker process): (per-channel records, event times, channels used). plot_dir:
    where to draw the channel plots, or None."""
    rng = np.random.default_rng(seed)
    sd = load_session(cfg, stem)
    chans = channels or [c for c in DEFAULT_CHANNELS if c in available_channels(sd)]
    half = margin = max(1, int(round(a["response_s"] * sd.fps)))
    recs, times = [], []
    for ch in chans:
        Z = session_matrix(sd, ch)
        if len(Z) < 4 or Z.shape[1] < 4 * margin:
            continue
        perm = rng.permutation(len(Z))
        A, B = np.sort(perm[: len(Z) // 2]), np.sort(perm[len(Z) // 2:])
        if a["events"]:
            events = np.clip(np.searchsorted(sd.frame_t, a["events"]), 0, len(sd.frame_t) - 1)
            B = np.arange(len(Z))
        else:
            events = pick_events(Z[A], a["n_events"], int(round(a["min_sep_s"] * sd.fps)))
        if not len(events):
            continue
        cam = camera_like(sd, events, half)
        res = check(Z[B], events, half, a["n_null"], rng, margin)
        nocam = events[~cam]
        res_nc = check(Z[B], nocam, half, a["n_null"], rng, margin) if len(nocam) >= 3 else {}
        recs.append({"stem": stem, "channel": ch, "n_students": len(Z),
                     "source": "given" if a["events"] else "half A", **res,
                     "n_events_camera_like": int(cam.sum()),
                     **{f"{k}_nocam": v for k, v in res_nc.items() if k != "n_events"},
                     "n_events_nocam": int(len(nocam))})
        times += [{"stem": stem, "channel": ch, "t": float(sd.frame_t[f]),
                   "t_from_start": float(sd.frame_t[f] - sd.frame_t[0])} for f in events]
        if plot_dir is not None:
            plot(sd, ch, Z, B, events, half, rng, Path(plot_dir) / f"event_check_{ch.replace('.', '_')}.png")
    return recs, times, chans


def main() -> None:
    ap = base_parser(__doc__, workers=True)
    ap.add_argument("--channels", nargs="*", default=None, help=f"default: {' '.join(DEFAULT_CHANNELS)} (those present)")
    ap.add_argument("--events", nargs="*", type=float, default=None, help="seconds since midnight; replaces the data events")
    ap.add_argument("--n-events", type=int, default=10)
    ap.add_argument("--min-sep-s", type=float, default=5.0)
    ap.add_argument("--response-s", type=float, default=1.0)
    ap.add_argument("--n-null", type=int, default=300)
    args = ap.parse_args()
    cfg, out = load(args)
    a = {k: getattr(args, k) for k in ("events", "n_events", "min_sep_s", "response_s", "n_null")}
    stems = cached_stems(cfg, args.sessions)
    jobs = [(cfg, stem, args.channels, a, [cfg["seed"], i], str(out) if i == 0 else None) for i, stem in enumerate(stems)]
    resume = partial_dir(out, "event_check", {"features": features_dir(cfg).name, "channels": args.channels, **a,
                                              "seed": cfg["seed"], "version": 2})
    recs, times = [], []
    for i, (job, (r, t, chans)) in enumerate(map_sessions(event_one, jobs, args.workers, stems, resume), 1):
        recs += r
        times += t
        print(f"  [{i}/{len(stems)}] {job[1]}: {', '.join(chans)}", flush=True)
    if not recs:
        raise SystemExit("no session with at least 4 students and events")
    df = pd.DataFrame(recs)
    write_table(df, out / "event_check")
    write_table(pd.DataFrame(times), out / "event_times")
    rows = []
    for ch, d in df.groupby("channel"):
        nc = d["ratio_nocam"] if "ratio_nocam" in d else pd.Series(dtype=float)
        rows.append({"channel": ch, "recordings": int(d["stem"].nunique()),
                     "share_at_events": d["share_at_events"].mean(), "share_random": d["share_random"].mean(),
                     "ratio": d["ratio"].mean(), "p_recordings": per_recording_test(d["ratio"] - 1),
                     "camera_like_share": d["n_events_camera_like"].sum() / max(d["n_events"].sum(), 1),
                     "ratio_nocam": nc.mean() if len(nc) else np.nan,
                     "p_recordings_nocam": per_recording_test(nc - 1) if len(nc) else np.nan})
    summary = pd.DataFrame(rows)
    summary["q_recordings"] = bh_qvalues(summary["p_recordings"])
    summary["q_recordings_nocam"] = bh_qvalues(summary["p_recordings_nocam"])
    write_table(summary, out / "event_check_channels")
    print("\nhalf B moving at half A's events against random moments, each recording counted once:")
    print(summary.round(3).to_string(index=False))
    print(f"\nwrote {out}/event_check, event_check_channels, event_times (.csv/.json), event_check_<channel>.png")


if __name__ == "__main__":
    main()

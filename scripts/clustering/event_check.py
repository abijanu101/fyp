"""Do the candidate channels move together when the class should react together? Screen 4 of
guide 15 §15.9.

  python -m scripts.clustering.event_check [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--channels hp.yaw rf.total ...] [--events 36012.4 36090 ...] [--n-events 10]
         [--response-s 1.0] [--sessions <stem> ...] [--out outputs/experiments/10_clustering]

DIPSER has no annotated class events (a recording is one activity), so events come from the data,
without circularity: the students of each session are split at random into two halves; events are
the largest jumps of half A's median signal (z-scored per student, peaks at least --min-sep-s
apart); the test is whether half B moves at those moments, against random moments. With --events
(seconds since midnight, the camera clock, e.g. moments found in label_viewer / output_viewer:
the teacher pointing, everyone starting to write), every student is tested at those moments.

A student "moves" at a moment when their own frame-to-frame change exceeds their own 90th
percentile within --response-s of it.

Writes to --out:
  event_check.csv          per (session, channel): events, the share of half B moving at events and
                           at random moments, their ratio and an empirical p-value
  event_times.csv          every event (seconds since midnight and since the recording's start),
                           to look up in the viewers
  event_check_<channel>.png  per channel, for the first session: the students x time raster
                           (z-scored) with the events marked, and the event-locked mean of half B
                           against the random-moment band
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.clustering.common import (DEFAULT_CHANNELS, available_channels, base_parser, cached_stems, load,
                                       load_session, plot_ready, series_matrix, write_table)
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


def main() -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--channels", nargs="*", default=None, help=f"default: {' '.join(DEFAULT_CHANNELS)} (those present)")
    ap.add_argument("--events", nargs="*", type=float, default=None, help="seconds since midnight; replaces the data events")
    ap.add_argument("--n-events", type=int, default=10)
    ap.add_argument("--min-sep-s", type=float, default=5.0)
    ap.add_argument("--response-s", type=float, default=1.0)
    ap.add_argument("--n-null", type=int, default=300)
    args = ap.parse_args()
    cfg, out = load(args)
    rng = np.random.default_rng(cfg["seed"])

    recs, times = [], []
    for si, stem in enumerate(cached_stems(cfg, args.sessions)):
        sd = load_session(cfg, stem)
        chans = args.channels or [c for c in DEFAULT_CHANNELS if c in available_channels(sd)]
        half, margin = max(1, int(round(args.response_s * sd.fps))), max(1, int(round(args.response_s * sd.fps)))
        for ch in chans:
            Z = session_matrix(sd, ch)
            if len(Z) < 4 or Z.shape[1] < 4 * margin:
                continue
            perm = rng.permutation(len(Z))
            A, B = np.sort(perm[: len(Z) // 2]), np.sort(perm[len(Z) // 2:])
            if args.events:
                events = np.clip(np.searchsorted(sd.frame_t, args.events), 0, len(sd.frame_t) - 1)
                B = np.arange(len(Z))
            else:
                events = pick_events(Z[A], args.n_events, int(round(args.min_sep_s * sd.fps)))
            if not len(events):
                continue
            recs.append({"stem": stem, "channel": ch, "n_students": len(Z),
                         "source": "given" if args.events else "half A", **check(Z[B], events, half, args.n_null, rng, margin)})
            times += [{"stem": stem, "channel": ch, "t": float(sd.frame_t[f]), "t_from_start": float(sd.frame_t[f] - sd.frame_t[0])}
                      for f in events]
            if si == 0:
                plot(sd, ch, Z, B, events, half, rng, out / f"event_check_{ch.replace('.', '_')}.png")
        print(f"  {stem}: {', '.join(chans)}", flush=True)
    if not recs:
        raise SystemExit("no session with at least 4 students and events")
    df = pd.DataFrame(recs)
    write_table(df, out / "event_check")
    write_table(pd.DataFrame(times), out / "event_times")
    summary = df.groupby("channel").agg(sessions=("stem", "nunique"), at_events=("share_at_events", "mean"),
                                        random=("share_random", "mean"), ratio=("ratio", "mean"),
                                        p_below_05=("p", lambda p: float((p < 0.05).mean())))
    print("\nhalf B moving within the response window (mean over sessions):")
    print(summary.round(3).to_string())
    print(f"\nwrote {out}/event_check.csv, event_times.csv, event_check_<channel>.png")


if __name__ == "__main__":
    main()

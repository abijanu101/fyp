"""Which signals are worth clustering on: the label-free screens 1-3 of guide 15 §15.9.

  python -m scripts.clustering.signal_screen [--config docs/experiments/09_vfoa.json] [--set k=v ...]
         [--channels hp.yaw rf.total ...] [--surrogates 20] [--max-windows 0] [--sessions <stem> ...]
         [--workers 1] [--out outputs/experiments/10_clustering]

Resumable: each recording's result is saved under <out>/partial/signal_screen/<settings hash>/ as it
finishes; a rerun with the same settings loads those and only computes the rest.

Reads only the caches of `run_pipeline extract` (frame-pass .npz, window and per-student tables), on
the original recordings. For every regular-grid window and every per-frame channel:

  1 noise floor        each visible student's within-window SD over the white-noise SD estimated
                       from first differences (classrel.noise_ratio: about 1 for pure jitter), and
                       the lag-1 autocorrelation (about 0 for jitter). A channel whose median ratio
                       is near 1 only measures the tracker's noise, so clustering on it groups
                       students by seat and distance.
  2 variance split     for every trk.* window descriptor and every channel's window mean: the share
                       of the variance between tracks (the seat) and between sessions (the camera),
                       and what is left within a track, where the labels move (only 33% of
                       labeler_01's variance is between students, experiment 06 §1.7).
  3 surrogate synchrony the class's mean pairwise correlation in each window against the same with
                       every student's series circularly shifted by its own random offset (keeps
                       each student's signal, breaks the timing between them). A channel whose real
                       synchrony is not above the shifted one carries no class synchrony.

Statistics: windows overlap and neighbouring windows are alike, so they are not independent. The
pass flag therefore uses one value per recording (its mean real-minus-shifted synchrony), tested
across recordings with a one-sided Wilcoxon signed-rank test, then Benjamini-Hochberg across the
channels (q < 0.05). The window-level sign test is kept as a description only.

Camera check: motion shared by every student at once (camera shake, lighting) keeps its timing in
the real data and so beats the shifted series too. Each window is tagged with its whole-picture
motion (the scene feature `motion`); synchrony is also reported in each recording's calmest third
of windows (`sync_excess_low_motion`, with its own per-recording test), with the Spearman
correlation between a window's excess synchrony and its global motion. Synchrony that holds in the
calm windows is not driven by the camera; synchrony that only shows when the picture moves may be.

Every table is written as .csv and .json (records). Writes to --out:
  signal_screen_channels.csv     per channel: noise ratio, lag-1 autocorrelation, real and
                                 surrogate synchrony, the per-recording test (p, q), the camera
                                 check, within-track share of the window mean, and pass flags
  signal_screen_recordings.csv   per (channel, recording): mean excess synchrony, overall and in the
                                 calm third of windows
  signal_screen_descriptors.csv  per trk.* descriptor: between-track, between-session and
                                 within-track variance shares (screen 2)
  signal_screen.png              the three screens side by side
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.clustering.common import (available_channels, base_parser, bh_qvalues, cached_stems, load,
                                       load_session, map_sessions, partial_dir, per_recording_test, plot_ready,
                                       series_matrix, write_table)
from scripts.run_pipeline import features_dir
from src.cfmu.classrel import circular_shift, lag1_autocorr, noise_ratio, pairwise_corr


def variance_shares(x: pd.Series, track: pd.Series, session: pd.Series) -> dict[str, float]:
    """Between-track and between-session shares of x's variance (1 - within / total)."""
    ok = x.notna()
    x, track, session = x[ok].astype(float), track[ok], session[ok]
    total = x.var(ddof=0)
    if len(x) < 10 or not np.isfinite(total) or total <= 1e-18:
        return {"between_track_share": np.nan, "between_session_share": np.nan, "within_track_share": np.nan}
    bt = 1 - (x - x.groupby(track).transform("mean")).var(ddof=0) / total
    bs = 1 - (x - x.groupby(session).transform("mean")).var(ddof=0) / total
    return {"between_track_share": float(bt), "between_session_share": float(bs) if session.nunique() > 1 else np.nan,
            "within_track_share": float(1 - bt)}


def screen_session(sd, channels: list[str], n_surr: int, max_windows: int, rng) -> tuple[list[dict], list[dict]]:
    """Per (window, channel) synchrony records and per (window, student, channel) noise records."""
    win_recs, stu_recs = [], []
    windows = sd.windows if not max_windows else sd.windows[:: max(1, len(sd.windows) // max_windows)]
    for wi, w in enumerate(windows):
        tracks = sd.visible(w)
        if len(tracks) < 3:
            continue
        for ch in channels:
            S = series_matrix(sd, w, tracks, ch)
            for i, tr in enumerate(tracks):
                stu_recs.append({"stem": sd.stem, "t": w.t_start, "tid": tr.tid, "channel": ch,
                                 "noise_ratio": noise_ratio(S[i]), "ac1": lag1_autocorr(S[i]),
                                 "window_mean": float(np.nanmean(S[i])) if np.isfinite(S[i]).any() else np.nan})
            C = pairwise_corr(S)
            real = np.nanmean(C) if np.isfinite(C).any() else np.nan
            if not np.isfinite(real):
                continue
            surr = []
            for _ in range(n_surr):
                Cs = pairwise_corr(circular_shift(S, rng))
                if np.isfinite(Cs).any():
                    surr.append(np.nanmean(Cs))
            surr = np.array(surr)
            sdv = surr.std() if len(surr) > 1 else np.nan
            win_recs.append({"stem": sd.stem, "t": w.t_start, "channel": ch, "n": len(tracks), "real": real,
                             "global_motion": sd.global_motion(w),
                             "surrogate": surr.mean() if len(surr) else np.nan,
                             "z": (real - surr.mean()) / sdv if sdv and np.isfinite(sdv) and sdv > 1e-12 else np.nan,
                             "above_95": float(real > np.quantile(surr, 0.95)) if len(surr) >= 10 else np.nan})
    return win_recs, stu_recs


def screen_one(cfg: dict, stem: str, channels, n_surr: int, max_windows: int, seed):
    """One session (worker process): (window records, student records, grid descriptors, channels)."""
    sd = load_session(cfg, stem)
    chans = channels or available_channels(sd)
    w, s = screen_session(sd, chans, n_surr, max_windows, np.random.default_rng(seed))
    return pd.DataFrame(w), pd.DataFrame(s), sd.tracks.assign(stem=stem), chans


def per_recording(win: pd.DataFrame) -> pd.DataFrame:
    """Per (channel, recording): mean excess synchrony over all windows and over the recording's
    calmest third (lowest global motion)."""
    w = win.assign(excess=win["real"] - win["surrogate"])
    w["calm"] = w.groupby(["channel", "stem"])["global_motion"].rank(pct=True) <= 1 / 3
    allw = w.groupby(["channel", "stem"]).agg(n_windows=("excess", "size"), excess=("excess", "mean"))
    calm = w[w["calm"]].groupby(["channel", "stem"]).agg(excess_low_motion=("excess", "mean"))
    return allw.join(calm).reset_index()


def summarise(win: pd.DataFrame, stu: pd.DataFrame, noise_pass: float) -> pd.DataFrame:
    from scipy.stats import binomtest, spearmanr

    out = []
    stu = stu.assign(track=stu["stem"] + "#" + stu["tid"].astype(str))
    per_rec = per_recording(win)
    for ch, s in stu.groupby("channel"):
        w = win[win["channel"] == ch]
        r = per_rec[per_rec["channel"] == ch]
        ex = (w["real"] - w["surrogate"])
        ok = ex.notna() & w["global_motion"].notna()
        rho_m = float(spearmanr(ex[ok], w.loc[ok, "global_motion"]).statistic) if ok.sum() > 10 else np.nan
        finite = s["noise_ratio"].replace(np.inf, np.nan)
        diff = (w["real"] - w["surrogate"]).dropna()
        p = binomtest(int((diff > 0).sum()), len(diff), 0.5, alternative="greater").pvalue if len(diff) else np.nan
        rec = {"channel": ch, "n_windows": int(len(w)), "n_student_windows": int(len(s)),
               "nan_share": float(s["noise_ratio"].isna().mean()),
               "noise_ratio_median": float(finite.median()), "share_above_noise": float((finite > noise_pass).mean()),
               "ac1_median": float(s["ac1"].median()),
               "sync_real": float(w["real"].mean()) if len(w) else np.nan,
               "sync_surrogate": float(w["surrogate"].mean()) if len(w) else np.nan,
               "sync_excess": float(diff.mean()) if len(diff) else np.nan,
               "sync_z_median": float(w["z"].median()) if len(w) else np.nan,
               "share_above_95": float(w["above_95"].mean()) if len(w) else np.nan,
               "sign_test_p_windows": float(p),
               "n_recordings": int(r["excess"].notna().sum()),
               "recordings_positive": float((r["excess"] > 0).mean()) if len(r) else np.nan,
               "p_recordings": per_recording_test(r["excess"]),
               "sync_excess_low_motion": float(r["excess_low_motion"].mean()) if len(r) else np.nan,
               "p_recordings_low_motion": per_recording_test(r["excess_low_motion"]),
               "rho_excess_vs_global_motion": rho_m,
               **variance_shares(s["window_mean"], s["track"], s["stem"])}
        out.append(rec)
    df = pd.DataFrame(out)
    df["q_recordings"] = bh_qvalues(df["p_recordings"])
    df["q_recordings_low_motion"] = bh_qvalues(df["p_recordings_low_motion"])
    df["pass_noise"] = df["noise_ratio_median"] > noise_pass
    df["pass_sync"] = (df["q_recordings"] < 0.05) & (df["sync_excess"] > 0)
    df["pass_sync_low_motion"] = (df["q_recordings_low_motion"] < 0.05) & (df["sync_excess_low_motion"] > 0)
    return df.sort_values(["pass_sync", "sync_excess"], ascending=[False, False])


def plot(ch: pd.DataFrame, desc: pd.DataFrame, stu: pd.DataFrame, noise_pass: float, path) -> None:
    plt = plot_ready()
    order = ch["channel"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(18, max(4.5, 0.32 * max(len(order), len(desc) / 2.5) + 2)))
    data = [stu.loc[stu["channel"] == c, "noise_ratio"].replace(np.inf, np.nan).dropna().clip(upper=10) for c in order]
    axes[0].boxplot(data, orientation="horizontal", showfliers=False)
    axes[0].set_yticks(range(1, len(order) + 1), order, fontsize=8)
    axes[0].axvline(1, color="#999", lw=1, ls=":")
    axes[0].axvline(noise_pass, color="#c0392b", lw=1)
    axes[0].set_xlabel("within-window SD / jitter SD (1 = pure noise)")
    axes[0].set_title("1  noise floor, per student-window", fontsize=10)
    y = np.arange(len(order))
    axes[1].barh(y + 0.2, ch["sync_real"], height=0.4, color="#2a78d6", label="real")
    axes[1].barh(y - 0.2, ch["sync_surrogate"], height=0.4, color="#bbbbbb", label="shifted (surrogate)")
    for yi, (ok, q, calm) in enumerate(zip(ch["pass_sync"], ch["q_recordings"], ch["pass_sync_low_motion"])):
        txt = (" pass" + (", calm too" if calm else ", NOT in calm windows")) if ok else f" q={q:.2f}"
        axes[1].text(axes[1].get_xlim()[1], yi, txt, va="center", fontsize=7)
    axes[1].set_yticks(y, order, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("class mean pairwise correlation")
    axes[1].set_title("3  synchrony against shifted series", fontsize=10)
    axes[1].legend(fontsize=8, loc="lower right")
    axes[0].invert_yaxis()
    d = desc.sort_values("within_track_share", ascending=False)
    yd = np.arange(len(d))
    axes[2].barh(yd, d["within_track_share"], color="#1baf7a", label="within track (where labels move)")
    axes[2].barh(yd, 1 - d["within_track_share"], left=d["within_track_share"], color="#e0e0e0", label="between tracks (seat)")
    axes[2].set_yticks(yd, d["column"], fontsize=6)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("share of variance")
    axes[2].set_title("2  variance split of the window descriptors", fontsize=10)
    axes[2].legend(fontsize=7, loc="lower right")
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = base_parser(__doc__, workers=True)
    ap.add_argument("--channels", nargs="*", default=None, help="default: every channel in the cache")
    ap.add_argument("--surrogates", type=int, default=20)
    ap.add_argument("--max-windows", type=int, default=0, help="per session, evenly spaced; 0 = all")
    ap.add_argument("--noise-pass", type=float, default=1.5, help="median noise ratio a channel must exceed")
    args = ap.parse_args()
    cfg, out = load(args)

    stems = cached_stems(cfg, args.sessions)
    jobs = [(cfg, stem, args.channels, args.surrogates, args.max_windows, [cfg["seed"], i]) for i, stem in enumerate(stems)]
    resume = partial_dir(out, "signal_screen", {"features": features_dir(cfg).name, "channels": args.channels,
                                                "surrogates": args.surrogates, "max_windows": args.max_windows,
                                                "seed": cfg["seed"], "version": 2})
    wins, stus, desc_rows, channels = [], [], [], set()
    for i, (job, (w, s, tracks, chans)) in enumerate(map_sessions(screen_one, jobs, args.workers, stems, resume), 1):
        wins.append(w)
        stus.append(s)
        desc_rows.append(tracks)
        channels |= set(chans)
        print(f"  [{i}/{len(stems)}] {job[1]}: {w['t'].nunique() if len(w) else 0} windows", flush=True)
    win, stu = pd.concat(wins, ignore_index=True), pd.concat(stus, ignore_index=True)
    if stu.empty:
        raise SystemExit("no window with at least 3 visible students")
    ch = summarise(win, stu, args.noise_pass)
    write_table(ch, out / "signal_screen_channels")
    write_table(per_recording(win), out / "signal_screen_recordings")

    grid = pd.concat(desc_rows, ignore_index=True)
    track = grid["stem"] + "#" + grid["tid"].astype(str)
    desc = pd.DataFrame([{"column": c, **variance_shares(grid[c], track, grid["stem"])}
                         for c in sorted(c for c in grid if c.startswith("trk."))])
    write_table(desc, out / "signal_screen_descriptors")
    plot(ch, desc.dropna(subset=["within_track_share"]), stu, args.noise_pass, out / "signal_screen.png")

    n_win = len(win.drop_duplicates(["stem", "t"])) if len(win) else 0
    print(f"\n{len(stems)} sessions, {n_win} windows with 3+ students, {len(channels)} channels")
    print(f"{'channel':22s} {'noise':>6s} {'real':>7s} {'shifted':>7s} {'q (rec)':>8s} {'calm excess':>11s} "
          f"{'q calm':>7s} {'rho motion':>10s} {'within':>7s}  pass")
    for _, r in ch.iterrows():
        flags = ("noise " if r["pass_noise"] else "") + ("sync " if r["pass_sync"] else "") + \
                ("calm" if r["pass_sync_low_motion"] else "")
        print(f"{r['channel']:22s} {r['noise_ratio_median']:6.2f} {r['sync_real']:7.3f} {r['sync_surrogate']:7.3f} "
              f"{r['q_recordings']:8.3f} {r['sync_excess_low_motion']:11.3f} {r['q_recordings_low_motion']:7.3f} "
              f"{r['rho_excess_vs_global_motion']:10.3f} {r['within_track_share']:7.2f}  {flags}")
    print(f"\nwrote {out}/signal_screen_channels, signal_screen_recordings, signal_screen_descriptors (.csv/.json), "
          f"signal_screen.png")


if __name__ == "__main__":
    main()

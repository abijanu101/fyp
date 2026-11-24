"""Held-out retention curves, every window predicted by models that never saw its group.

  python -m scripts.reporting.plot_results [--config configs/deliverable3.json] [--set key=value ...]
      [--sessions experiment_03/context_cam_01 ...] [--out DIR] [--class-curve] [--final]

--sessions keeps only the sessions whose key contains one of the given substrings (the models are
still fit on every training group); --out writes there instead of <output_dir>/plots/<hash>/;
--final also draws the final curve (per-labeler mode; off by default, since the median of curves at
different labelers' levels matches no one's entries).

For each held-out group (leave-one-group-out, as in train_models), the gate and the models are fit
on the other groups and predict the held-out group's windows. One PNG per session, plus one
overview image, under
  <output_dir>/plots/<config hash[:12]>/

engagement.labelers = "per_labeler" (the default), on the 1-5 rating scale. Each panel follows one
student: the one with the most self-reports in the session (ties: the larger variance of those
self-reports) among those with a linked track, with each model's curve for that student alone (mean
over the student's tracks per window, gaps where it is not tracked) and only that student's entries.
--class-curve plots the class curves (mean over the window's students) and every entry instead.
  labeler curves       self thick orange, drawn on top; each expert labeler thin, in its own light colour
  final curve          with --class-curve --final, black dotted: their median (output.prediction =
                       per_labeler) or the majority curve
  rating entries       held-out ratings as entered, at their time: self orange o, experts x in their
                       labeler's colour
  dashed line          labels.tau_att (engaged at or above)
  title                the true activity, the gate's hit rate, and the MAE of each labeler's model
                       against that labeler's own entries (experts, self)

engagement.labelers = "consensus" / "ensemble" (the window-grid paths), on 0-1:
  engagement score     the mixture's window score (late: estimated share of engaged students)
  labelled share       share of labelled students whose consensus attention >= labels.tau_att
  cut-off              the score at which a window counts as engaged (decision_threshold)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_pipeline import parse_sets
from scripts.train_models import engagement_rows, heldout_splits, load_table, mixture_factory, with_augmented
from src.core.config import config_hash, load_config
from src.core.pipeline import SCENE_PREFIX, VEC_PREFIX
from src.experts import decision_threshold, fit_engagement, predict_windows
from src.spu import GATES


def held_out_predictions(cfg: dict, df) -> tuple[np.ndarray, np.ndarray]:
    """(engagement score, gate probabilities) per window, each from models that never saw its group."""
    classes, mcfg = cfg["classes"], cfg["mixture"]
    vec_cols = sorted(c for c in df if c.startswith(VEC_PREFIX))
    gate_cols = sorted(c for c in df if c.startswith(SCENE_PREFIX)) + (vec_cols if cfg["gate_uses_vec"] else [])
    Xg, exp = df[gate_cols].to_numpy(float), df["exp"].to_numpy()
    rows = engagement_rows(cfg, df, vec_cols)
    p, g = np.full(len(df), np.nan), np.zeros((len(df), len(classes)))
    for _, tr, te in heldout_splits(df):                     # augmented copies train, never scored
        gate = GATES.build(cfg["gate"]["name"], classes=classes, seed=cfg["seed"],
                           **cfg["gate"].get("params", {})).fit(Xg[tr], exp[tr])
        g[te] = gate.predict_proba(Xg[te])
        mix, fill = fit_engagement(cfg["engagement"], mixture_factory(cfg, rows, None, mcfg["expert_scope"],
                                                                       mcfg["mode"]), rows, tr)
        p[te] = predict_windows(mix, rows, te, g[te], fill)
    return p, g


def plot_session(ax, sdf, p, g, classes, tau, colors):
    t = (sdf["t_start"].to_numpy() + sdf["t_end"].to_numpy()) / 2
    t = t - sdf["t_start"].iloc[0]
    half = (sdf["t_end"].iloc[0] - sdf["t_start"].iloc[0]) / 4          # half the window step at 50% overlap
    for ti, k in zip(t, g.argmax(1)):
        ax.axvspan(ti - half, ti + half, color=colors[k], alpha=0.18, lw=0)
    ax.plot(t, sdf["share"].to_numpy(), color="0.35", lw=1.2, drawstyle="steps-mid", label="labelled share")
    ax.plot(t, p, color="C3", lw=1.8, label="engagement score")
    ax.axhline(tau, color="0.2", lw=0.8, ls="--", label=f"cut-off {tau:g}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(t[0] - half, t[-1] + half)
    exp, key = sdf["exp"].iloc[0], sdf["session"].iloc[0]
    hit = float(np.mean(np.array(classes)[g.argmax(1)] == exp))
    ok = np.isfinite(sdf["share"].to_numpy())
    r = np.corrcoef(p[ok], sdf["share"].to_numpy()[ok])[0, 1] if ok.sum() > 2 and np.std(p[ok]) > 0 else np.nan
    ax.set_title(f"{key}   true activity {exp}, gate right {hit:.0%}, r = {r:.2f}", fontsize=9)
    ax.set_xlabel("seconds into session", fontsize=8)
    ax.tick_params(labelsize=7)


def own_mae(entries, curves, t, t0) -> float:
    """MAE of each entry against its own labeler's model curve, as the tables score it (windows
    where the curve is NaN, e.g. the student is not tracked, are interpolated over)."""
    err = []
    for name, e in entries.groupby("labeler"):
        c = curves.get(name)
        ok = np.isfinite(c) if c is not None else np.zeros(0, bool)
        if ok.sum() >= 2:
            err.append(np.abs(np.interp(e["t"] - t0, t[ok], c[ok]) - e["rating"]))
    return float(np.concatenate(err).mean()) if err else np.nan


def student_curves(cfg: dict, res: dict, session: str, idx: np.ndarray, entries) -> tuple[str | None, dict]:
    """The student of a session with the most self-reports (ties: the larger variance of those
    self-reports; then the lower subject id) among those with a linked track, and each labeler
    model's curve for that student alone: per window, the mean prediction over the student's
    tracks, NaN where the student is not tracked."""
    from scripts.run_pipeline import tracks_dir
    from src.core.store import load_track
    from src.core.types import SessionKey
    from src.datasets import DipserIndex
    from src.tracking import associate_sids

    key = SessionKey(*session.split("#")[0].split("/"))
    st = load_track(tracks_dir(cfg) / f"{res['stems'][idx[0]]}.npz")
    assoc = cfg["association"]
    tid_to_sid = associate_sids(st.tracklets, DipserIndex.from_config(cfg).annotations(key),
                                assoc["min_iou"], assoc["min_match_share"], assoc.get("method", "per_frame"))
    own = entries[(entries["labeler"] == "self_labeling") & entries["sid"].isin(set(tid_to_sid.values()))]
    if own.empty:
        return None, {}
    stats = own.groupby("sid")["rating"].agg(["size", "var"]).fillna(0.0)
    sid = sorted(stats.index, key=lambda s: (-stats.at[s, "size"], -stats.at[s, "var"], s))[0]
    tids = [tid for tid, s in tid_to_sid.items() if s == sid]
    pos = np.full(int(max(res["grid_window"].max(), idx.max())) + 1, -1)
    pos[idx] = np.arange(len(idx))
    sel = (pos[res["grid_window"]] >= 0) & np.isin(res["grid_tid"], tids)
    w = pos[res["grid_window"][sel]]
    count = np.bincount(w, minlength=len(idx))
    curves = {}
    for name, v in res["grid_curves"].items():
        total = np.bincount(w, weights=np.nan_to_num(v[sel]), minlength=len(idx))
        curves[name] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return sid, curves


EXPERT_COLOURS = ("tab:blue", "tab:green", "tab:purple", "tab:cyan", "tab:olive", "tab:gray")


def expert_colour(name: str, lighten: float = 0.45):
    """A light, desaturated colour per expert labeler (labeler_01 -> blue, _02 -> green, ...), kept
    away from the self-report's orange."""
    from matplotlib.colors import to_rgb
    digits = "".join(ch for ch in name if ch.isdigit())
    base = np.array(to_rgb(EXPERT_COLOURS[(int(digits) - 1 if digits else 0) % len(EXPERT_COLOURS)]))
    return tuple(base + (1 - base) * lighten)


def plot_session_per_labeler(ax, sdf, final, curves, g, entries, classes, tau_att, final_name, who="class"):
    t0 = sdf["t_start"].iloc[0]
    t = (sdf["t_start"].to_numpy() + sdf["t_end"].to_numpy()) / 2 - t0
    half = (sdf["t_end"].iloc[0] - sdf["t_start"].iloc[0]) / 4
    ex = entries[entries["labeler"] != "self_labeling"]
    se = entries[entries["labeler"] == "self_labeling"]
    jitter = lambda n: np.random.default_rng(0).uniform(-0.08, 0.08, n)
    # the self-report is what matters: each expert in its own light colour behind, self drawn last and thick
    for name in sorted(n for n in curves if n != "self_labeling"):
        col = expert_colour(name)
        e = ex[ex["labeler"] == name]
        ax.scatter(e["t"] - t0, e["rating"] + jitter(len(e)), s=9, marker="x", color=col, lw=0.8, zorder=1)
        ax.plot(t, curves[name], lw=1.1, color=col, label=name, zorder=2)
    if final_name:
        ax.plot(t, final, color="black", lw=1.6, ls=":", label=final_name, zorder=3)
    ax.scatter(se["t"] - t0, se["rating"] + jitter(len(se)), s=16, marker="o", color="#d4570b", alpha=0.75,
               edgecolors="none", label="self entries", zorder=4)
    if "self_labeling" in curves:
        ax.plot(t, curves["self_labeling"], lw=2.6, color="#d4570b", label="self model", zorder=5)
    ax.axhline(tau_att, color="0.3", lw=0.7, ls="--", label=f"tau_att {tau_att:g}", zorder=0)
    ax.set_ylim(0.8, 5.2)
    ax.set_xlim(t[0] - half, t[-1] + half)
    exp, key = sdf["exp"].iloc[0], sdf["session"].iloc[0]
    hit = float(np.mean(np.array(classes)[g.argmax(1)] == exp))
    ax.set_title(f"{key}   {who}   true activity {exp}, gate right {hit:.0%}, MAE vs own entries: experts "
                 f"{own_mae(ex, curves, t, t0):.2f}, self {own_mae(se, curves, t, t0):.2f}", fontsize=9)
    ax.set_xlabel("seconds into session", fontsize=8)
    ax.set_ylabel("rating", fontsize=8)
    ax.tick_params(labelsize=7)


def main_per_labeler(cfg: dict, df, only: list[str] | None = None, out_dir: str | None = None,
                     draw_final: bool = False, class_curve: bool = False) -> None:
    from scripts.train_models import heldout_curves

    classes, tau_att = cfg["classes"], float(cfg["labels"]["tau_att"])
    res = heldout_curves(cfg, df)
    res["stems"] = df["stem"].to_numpy()
    rows = res["rows"]
    entries = rows[~rows["aug"]].drop_duplicates("entry_id")
    final_name = ("median" if cfg["output"]["prediction"] == "per_labeler" else "majority") if draw_final else None
    out = Path(out_dir) if out_dir else Path(cfg["output_dir"]) / "plots" / config_hash(cfg)[:12]
    out.mkdir(parents=True, exist_ok=True)
    sessions = [(s, i) for s, i in df[~df["aug"]].groupby("session", sort=True).indices.items()
                if not only or any(o in s for o in only)]
    overview, axes = plt.subplots(len(sessions), 1, figsize=(11, 2.8 * len(sessions)), squeeze=False)
    for (session, idx), ax in zip(sessions, axes[:, 0]):
        ents = entries[entries["session"] == session]
        curves, who = {n: c[idx] for n, c in res["curves"].items()}, "class"
        if not class_curve:
            sid, curves = student_curves(cfg, res, session, idx, ents)
            ents, who = ents[ents["sid"] == sid], f"student {sid}"
        args = (df.iloc[idx], res["final"][idx], curves, res["g"][idx], ents, classes, tau_att,
                final_name if class_curve else None, who)
        plot_session_per_labeler(ax, *args)
        fig, one = plt.subplots(figsize=(11, 3.4))
        plot_session_per_labeler(one, *args)
        fig.legend(*one.get_legend_handles_labels(), loc="lower center", ncol=8, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(out / f"{session.replace('/', '__')}.png", dpi=120)
        plt.close(fig)
    overview.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=8, fontsize=8,
                    frameon=False)
    overview.tight_layout(rect=(0, 0.03, 1, 1))
    overview.savefig(out / "overview.png", dpi=110)
    print(f"{len(sessions)} sessions, per_labeler (final: {final_name}), {len(entries)} held-out entries -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--sessions", nargs="*", help="per-labeler mode: plot only sessions whose key contains one of these")
    ap.add_argument("--out", help="per-labeler mode: output folder (default <output_dir>/plots/<hash>/)")
    ap.add_argument("--final", action="store_true", help="per-labeler mode: also draw the final (median) curve")
    ap.add_argument("--class-curve", action="store_true",
                    help="per-labeler mode: plot the class curves (mean over students) and every entry instead "
                         "of the student with the most self-reports")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    if cfg["engagement"].get("labelers") in ("per_labeler", "pooled"):
        return main_per_labeler(cfg, with_augmented(cfg, load_table(cfg)), args.sessions, args.out,
                                args.final, args.class_curve)
    classes = cfg["classes"]
    tau = decision_threshold(cfg["engagement"]["aggregate"], cfg["mixture"]["tau"], cfg["labels"]["tau_share"])
    colors = [plt.get_cmap("tab10")(i) for i in range(len(classes))]

    df = with_augmented(cfg, load_table(cfg))
    p, g = held_out_predictions(cfg, df)
    out = Path(cfg["output_dir"]) / "plots" / config_hash(cfg)[:12]
    out.mkdir(parents=True, exist_ok=True)

    sessions = list(df[~df["aug"]].groupby("session", sort=True).indices.items())   # originals only
    legend = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in colors]
    overview, axes = plt.subplots(len(sessions), 1, figsize=(11, 2.4 * len(sessions)), squeeze=False)
    for (session, idx), ax in zip(sessions, axes[:, 0]):
        plot_session(ax, df.iloc[idx], p[idx], g[idx], classes, tau, colors)
        fig, one = plt.subplots(figsize=(11, 3))
        plot_session(one, df.iloc[idx], p[idx], g[idx], classes, tau, colors)
        one.legend(loc="upper right", fontsize=7)
        fig.legend(legend, [f"gate: {c}" for c in classes], loc="lower center", ncol=len(classes), fontsize=7)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(out / f"{session.replace('/', '__')}.png", dpi=120)
        plt.close(fig)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    overview.legend(legend, [f"gate: {c}" for c in classes], loc="lower center", ncol=len(classes), fontsize=8)
    overview.tight_layout(rect=(0, 0.03, 1, 1))
    overview.savefig(out / "overview.png", dpi=110)
    print(f"{len(sessions)} sessions, aggregate={cfg['engagement']['aggregate']}, cut-off {tau:g} -> {out}")


if __name__ == "__main__":
    main()

"""Held-out retention curves, every window predicted by models that never saw its group.

  python -m scripts.plot_results [--config configs/deliverable3.json] [--set key=value ...]

For each held-out group (leave-one-group-out, as in train_models), the gate and the models are fit
on the other groups and predict the held-out group's windows. One PNG per session, plus one
overview image, under
  <output_dir>/plots/<config hash[:12]>/

engagement.labelers = "per_labeler" (the default), on the 1-5 rating scale:
  labeler curves       each labeler model's class curve: experts blue, self orange
  final curve          black: their median (output.prediction = per_labeler) or the majority curve
  rating entries       every held-out rating as entered, at its time: experts blue x, self orange o
  dashed line          labels.tau_att (engaged at or above)
  shaded background    the gate's predicted activity; the title gives the true one and the final
                       curve's MAE against the expert entries

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


def plot_session_per_labeler(ax, sdf, final, curves, g, entries, classes, tau_att, colors, final_name):
    t0 = sdf["t_start"].iloc[0]
    t = (sdf["t_start"].to_numpy() + sdf["t_end"].to_numpy()) / 2 - t0
    half = (sdf["t_end"].iloc[0] - sdf["t_start"].iloc[0]) / 4
    for ti, k in zip(t, g.argmax(1)):
        ax.axvspan(ti - half, ti + half, color=colors[k], alpha=0.18, lw=0)
    first_expert = True
    for name, c in curves.items():
        own = name == "self_labeling"
        ax.plot(t, c, lw=1.0, alpha=0.75, color="tab:orange" if own else "tab:blue",
                label="self model" if own else ("expert models" if first_expert else None))
        first_expert = first_expert and own
    ax.plot(t, final, color="black", lw=2.4, label=final_name)
    ex = entries[entries["labeler"] != "self_labeling"]
    se = entries[entries["labeler"] == "self_labeling"]
    jitter = lambda n: np.random.default_rng(0).uniform(-0.08, 0.08, n)
    ax.scatter(ex["t"] - t0, ex["rating"] + jitter(len(ex)), s=7, marker="x", color="tab:blue", alpha=0.45,
               lw=0.7, label="expert entries")
    ax.scatter(se["t"] - t0, se["rating"] + jitter(len(se)), s=9, marker="o", color="tab:orange", alpha=0.6,
               label="self entries")
    ax.axhline(tau_att, color="0.2", lw=0.8, ls="--", label=f"tau_att {tau_att:g}")
    ax.set_ylim(0.8, 5.2)
    ax.set_xlim(t[0] - half, t[-1] + half)
    exp, key = sdf["exp"].iloc[0], sdf["session"].iloc[0]
    hit = float(np.mean(np.array(classes)[g.argmax(1)] == exp))
    mae = np.nan
    if len(ex):
        mae = float(np.mean(np.abs(np.interp(ex["t"] - t0, t, final) - ex["rating"])))
    ax.set_title(f"{key}   true activity {exp}, gate right {hit:.0%}, final-curve MAE vs expert entries "
                 f"{mae:.2f}", fontsize=9)
    ax.set_xlabel("seconds into session", fontsize=8)
    ax.set_ylabel("rating", fontsize=8)
    ax.tick_params(labelsize=7)


def main_per_labeler(cfg: dict, df) -> None:
    from scripts.train_models import heldout_curves

    classes, tau_att = cfg["classes"], float(cfg["labels"]["tau_att"])
    colors = [plt.get_cmap("tab10")(i) for i in range(len(classes))]
    res = heldout_curves(cfg, df)
    rows = res["rows"]
    entries = rows[~rows["aug"]].drop_duplicates("entry_id")
    final_name = "median" if cfg["output"]["prediction"] == "per_labeler" else "majority"
    out = Path(cfg["output_dir"]) / "plots" / config_hash(cfg)[:12]
    out.mkdir(parents=True, exist_ok=True)
    sessions = list(df[~df["aug"]].groupby("session", sort=True).indices.items())
    legend = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in colors]
    overview, axes = plt.subplots(len(sessions), 1, figsize=(11, 2.8 * len(sessions)), squeeze=False)
    for (session, idx), ax in zip(sessions, axes[:, 0]):
        args = (df.iloc[idx], res["final"][idx], {n: c[idx] for n, c in res["curves"].items()}, res["g"][idx],
                entries[entries["session"] == session], classes, tau_att, colors, final_name)
        plot_session_per_labeler(ax, *args)
        fig, one = plt.subplots(figsize=(11, 3.4))
        plot_session_per_labeler(one, *args)
        one.legend(loc="upper right", fontsize=7, ncol=3)
        fig.legend(legend, [f"gate: {c}" for c in classes], loc="lower center", ncol=len(classes), fontsize=7)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(out / f"{session.replace('/', '__')}.png", dpi=120)
        plt.close(fig)
    axes[0, 0].legend(loc="upper right", fontsize=7, ncol=3)
    overview.legend(legend, [f"gate: {c}" for c in classes], loc="lower center", ncol=len(classes), fontsize=8)
    overview.tight_layout(rect=(0, 0.03, 1, 1))
    overview.savefig(out / "overview.png", dpi=110)
    print(f"{len(sessions)} sessions, per_labeler ({final_name}), {len(entries)} held-out entries -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    if cfg["engagement"].get("labelers") == "per_labeler":
        return main_per_labeler(cfg, with_augmented(cfg, load_table(cfg)))
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

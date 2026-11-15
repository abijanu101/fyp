"""Retention graph assembly and writers (design §9).

retention_graph.json carries only aggregate, time-indexed records: no bbox, TID, SID or
per-student value can reach it, because build_graph only accepts window-level arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.types import Window


def build_graph(windows: list[Window], context_probs: np.ndarray, classes: list[str],
                engagement_prob: np.ndarray, n_tracks: np.ndarray, tau: float,
                layout_family: str = "grid") -> list[dict]:
    t0 = windows[0].t_start if windows else 0.0
    records = []
    for i, w in enumerate(windows):
        k = int(np.argmax(context_probs[i]))
        records.append({
            "t_start": round(w.t_start - t0, 3),
            "t_end": round(w.t_end - t0, 3),
            "predicted_context": {"label": classes[k], "confidence": round(float(context_probs[i, k]), 4)},
            "layout_family": layout_family,
            "engaged": "Y" if engagement_prob[i] >= tau else "N",
            "engagement_prob": round(float(engagement_prob[i]), 4),
            "n_visible_tracks": int(n_tracks[i]),
        })
    return records


def build_rating_graph(windows: list[Window], context_probs: np.ndarray, classes: list[str],
                       rating: np.ndarray, labeler_ratings: dict[str, np.ndarray], n_tracks: np.ndarray,
                       tau_att: float, mode: str, layout_family: str = "grid") -> list[dict]:
    """Per-labeler output (docs/misc/targets.md §7.1 items 8-9): the final class rating per window
    (median of the labeler curves, or the majority curve), every labeler model's class curve, and
    engaged = final rating >= tau_att. Window-level arrays only, like build_graph."""
    t0 = windows[0].t_start if windows else 0.0
    records = []
    for i, w in enumerate(windows):
        k = int(np.argmax(context_probs[i]))
        records.append({
            "t_start": round(w.t_start - t0, 3),
            "t_end": round(w.t_end - t0, 3),
            "predicted_context": {"label": classes[k], "confidence": round(float(context_probs[i, k]), 4)},
            "layout_family": layout_family,
            "prediction": mode,
            "rating": round(float(rating[i]), 4),
            "engaged": "Y" if rating[i] >= tau_att else "N",
            "labeler_ratings": {n: round(float(r[i]), 4) for n, r in labeler_ratings.items()},
            "n_visible_tracks": int(n_tracks[i]),
        })
    return records


def write_graph_json(records: list[dict], out_dir: Path) -> Path:
    path = Path(out_dir) / "retention_graph.json"
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path


def write_graph_png(records: list[dict], out_dir: Path, smooth_windows: int = 5) -> Path:
    """Smoothed engagement_prob curve with context-segment shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([(r["t_start"] + r["t_end"]) / 2 for r in records])
    rated = "rating" in records[0]
    p = np.array([r["rating" if rated else "engagement_prob"] for r in records])
    k = max(1, min(smooth_windows, len(p)))

    def smoothed(v: np.ndarray) -> np.ndarray:
        return np.convolve(np.pad(v, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")

    smooth = smoothed(p)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    labels = [r["predicted_context"]["label"] for r in records]
    palette = plt.get_cmap("tab10")
    seen = set()
    start = 0
    for i in range(1, len(records) + 1):
        if i == len(records) or labels[i] != labels[start]:
            lab = labels[start]
            ax.axvspan(records[start]["t_start"], records[i - 1]["t_end"], alpha=0.12,
                       color=palette(int(lab) % 10), label=None if lab in seen else f"context {lab}", lw=0)
            seen.add(lab)
            start = i
    if rated:                                   # per-labeler curves: experts one colour, self another
        names = list(records[0]["labeler_ratings"])
        for n in names:
            own = n == "self_labeling"
            ax.plot(t, smoothed(np.array([r["labeler_ratings"][n] for r in records])), lw=1.0, alpha=0.7,
                    color="tab:orange" if own else "tab:blue",
                    label=("self" if own else ("experts" if n == next(x for x in names if x != "self_labeling")
                                               else None)))
        final = "median" if records[0]["prediction"] == "per_labeler" else "majority"
        ax.plot(t, smooth, color="black", lw=2.6, label=final)
        ax.set_ylim(1, 5)
        ax.set_ylabel("attention rating (1-5)")
    else:
        ax.plot(t, smooth, color="black", lw=1.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("engagement probability")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    path = Path(out_dir) / "retention_graph.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_manifest(manifest: dict, out_dir: Path) -> Path:
    path = Path(out_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path

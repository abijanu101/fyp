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
    p = np.array([r["engagement_prob"] for r in records])
    k = max(1, min(smooth_windows, len(p)))
    smooth = np.convolve(np.pad(p, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")

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
    ax.plot(t, smooth, color="black", lw=1.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("engagement probability")
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

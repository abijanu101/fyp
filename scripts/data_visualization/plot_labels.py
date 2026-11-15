"""Plot DIPSER attention labels over time, one colour per labeler.

    python scripts/data_visualization/plot_labels.py group_01 experiment_01 03
    python scripts/data_visualization/plot_labels.py group_02 experiment_04 subject_02 --exclude-self --out s02.png
    python scripts/data_visualization/plot_labels.py group_01 experiment_01 all        # every subject, one panel each
    python scripts/data_visualization/plot_labels.py --scan [--scan-dir outputs/label_plots]

--scan ignores group/experiment/subject and walks every group/experiment under DIPSEER/, saving
    <scan-dir>/<group>/<experiment>/subject_XX.png   one plot per subject
    <scan-dir>/<group>/<experiment>/_aggregate.png   the session's classroom-level graph

Per-subject plot: attention is a step function, each labeler's value holding until their next
attention entry. Dots mark the moments a labeler changed the value; lines are nudged vertically
by a small per-labeler offset so identical values stay visible; black = median across labelers
(what the pipeline's window labels are built from).

Aggregate plot: per labeler, the mean over the students that labeler rated; black = mean over
students of each student's median (the classroom attention curve the pipeline is trying to
predict), with the 25-75% band across students.

Time runs from the first to the last frame of the session's first context camera, so a final
rating holds until the session ends. Standalone (matplotlib + numpy only).
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
import zipfile
from pathlib import Path, PurePosixPath

import matplotlib
import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "DIPSER"
DEFAULT_SCAN_DIR = Path(__file__).resolve().parents[2] / "outputs" / "label_plots"
CLOCK = re.compile(r"(\d{1,2})[_:](\d{2})[_:](\d{2})[_:](\d+)")
SELF = "self_labeling"
COLOURS = {"labeler_01": "#1f77b4", "labeler_02": "#ff7f0e", "labeler_03": "#2ca02c",
           "labeler_04": "#d62728", "labeler_05": "#9467bd", SELF: "#8c564b"}
JITTER = 0.05
GRID_POINTS = 2000


# ----------------------------------------------------------------------------- data

def parse_clock(text: str) -> float:
    h, m, s, frac = CLOCK.fullmatch(text).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(frac) / 10 ** len(frac)


def load_subject(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{labeler: (change times, attention values)}; values outside 1-5 are dropped."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for m in sorted(zf.namelist()):
            p = PurePosixPath(m)
            if p.parent.name == "labels" and p.suffix == ".json":
                pts = sorted((parse_clock(e["datetime"]), int(e["attention"]))
                             for e in json.loads(zf.read(m))
                             if str(e.get("attention", "")).strip() and 1 <= int(e["attention"]) <= 5)
                out[p.stem] = (np.array([a for a, _ in pts]), np.array([b for _, b in pts], float))
    return dict(sorted(out.items(), key=lambda kv: (kv[0] == SELF, kv[0])))


def load_session(exp_dir: Path) -> dict[str, dict]:
    return {p.stem: load_subject(p) for p in sorted((exp_dir / "subjects").glob("subject_*.zip"))}


def session_span(exp_dir: Path, labels: dict) -> tuple[float, float] | None:
    """First and last frame time of the first context camera; falls back to label times."""
    for cam in sorted(exp_dir.glob("context_cam_*.zip")):
        with zipfile.ZipFile(cam) as zf:
            t = [parse_clock(PurePosixPath(m).stem) for m in zf.namelist()
                 if PurePosixPath(m).parent.name == "images" and m.lower().endswith((".png", ".jpg", ".jpeg"))]
        if t:
            return min(t), max(t)
    times = [tr[0] for labs in labels.values() for tr in labs.values() if len(tr[0])]
    if not times:
        return None
    times = np.concatenate(times)
    return float(times.min()), float(times.max())


def value_at(track, t: np.ndarray) -> np.ndarray:
    times, vals = track
    if not len(vals):
        return np.full(len(t), np.nan)
    i = np.searchsorted(times, t, side="right") - 1
    return np.where(i >= 0, vals[np.clip(i, 0, len(vals) - 1)], np.nan)


def nan_reduce(fn, stack: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns -> NaN
        return fn(stack, axis=0)


def student_median(labs: dict, grid: np.ndarray, exclude_self: bool) -> np.ndarray:
    use = [tr for n, tr in labs.items() if not (exclude_self and n == SELF)]
    if not use:
        return np.full(len(grid), np.nan)
    return nan_reduce(np.nanmedian, np.stack([value_at(tr, grid) for tr in use]))


# ----------------------------------------------------------------------------- plotting

def style_axis(ax, t0: float, t1: float) -> None:
    ax.set_ylim(0.6, 5.4)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlim(0, t1 - t0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("attention")


def time_axis(ax, t0: float, t1: float) -> None:
    ticks = np.arange(0, t1 - t0 + 1, 30)
    ax.set_xticks(ticks, [f"{int(s // 60)}:{int(s % 60):02d}" for s in ticks])
    ax.set_xlabel("time since first camera frame (m:ss)")


def plot_subject(ax, name: str, labs: dict, t0: float, t1: float, exclude_self: bool) -> None:
    for k, (labeler, (times, vals)) in enumerate(labs.items()):
        if len(times) == 0:
            continue
        dy = (k - (len(labs) - 1) / 2) * JITTER
        style = dict(color=COLOURS.get(labeler, "grey"), lw=1.6, alpha=0.9, ls="--" if labeler == SELF else "-")
        ax.step(np.append(times, t1) - t0, np.append(vals, vals[-1]) + dy, where="post",
                label=f"{labeler} ({len(times)} changes)", **style)
        ax.plot(times - t0, vals + dy, "o", ms=2.5, color=style["color"], alpha=0.8)
    grid = np.linspace(t0, t1, GRID_POINTS)
    ax.plot(grid - t0, student_median(labs, grid, exclude_self), color="black", lw=2.2, alpha=0.55,
            label="median" + (" (no self)" if exclude_self else ""))
    style_axis(ax, t0, t1)
    ax.set_title(name, loc="left", fontsize=10)


def subjects_figure(title: str, labels: dict, chosen: list[str], t0: float, t1: float, exclude_self: bool):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(chosen), 1, figsize=(13, 3.2 if len(chosen) == 1 else 1.9 * len(chosen)),
                             sharex=True, squeeze=False)
    for ax, s in zip(axes[:, 0], chosen):
        plot_subject(ax, s, labels[s], t0, t1, exclude_self)
    handles = {}
    for ax in axes[:, 0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l if len(chosen) == 1 else l.split(" (")[0], h)
    axes[0, 0].legend(handles.values(), handles.keys(), loc="upper right", fontsize=8,
                      ncol=min(len(handles), 4), framealpha=0.9)
    time_axis(axes[-1, 0], t0, t1)
    fig.suptitle(f"{title}: attention labels", fontsize=12)
    fig.tight_layout()
    return fig


def aggregate_figure(title: str, labels: dict, t0: float, t1: float, exclude_self: bool):
    import matplotlib.pyplot as plt
    grid = np.linspace(t0, t1, GRID_POINTS)
    fig, ax = plt.subplots(figsize=(13, 4))
    labelers = sorted({n for labs in labels.values() for n in labs}, key=lambda n: (n == SELF, n))
    for labeler in labelers:
        rated = [value_at(labs[labeler], grid) for labs in labels.values() if labeler in labs and len(labs[labeler][0])]
        if rated:
            ax.plot(grid - t0, nan_reduce(np.nanmean, np.stack(rated)), color=COLOURS.get(labeler, "grey"),
                    lw=1.5, alpha=0.9, ls="--" if labeler == SELF else "-",
                    label=f"{labeler} mean over {len(rated)} students")
    medians = np.stack([student_median(labs, grid, exclude_self) for labs in labels.values()])
    lo = nan_reduce(lambda a, axis: np.nanpercentile(a, 25, axis=axis), medians)
    hi = nan_reduce(lambda a, axis: np.nanpercentile(a, 75, axis=axis), medians)
    ax.fill_between(grid - t0, lo, hi, color="black", alpha=0.10, lw=0, label="25-75% of students (median)")
    ax.plot(grid - t0, nan_reduce(np.nanmean, medians), color="black", lw=2.4,
            label="class mean of student medians" + (" (no self)" if exclude_self else ""))
    style_axis(ax, t0, t1)
    time_axis(ax, t0, t1)
    ax.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.9)
    ax.set_title(f"{title}: classroom attention ({len(labels)} students)", fontsize=12)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- entry points

def scan(root: Path, out_dir: Path, exclude_self: bool) -> None:
    import matplotlib.pyplot as plt
    raw = root / "DIPSEER"
    sessions = sorted(e for g in raw.glob("group_*") for e in g.glob("experiment_*") if (e / "subjects").is_dir())
    for exp_dir in sessions:
        group, experiment = exp_dir.parent.name, exp_dir.name
        labels = load_session(exp_dir)
        span = session_span(exp_dir, labels) if labels else None
        if span is None:
            print(f"{group}/{experiment}: no labels, skipped")
            continue
        t0, t1 = span
        dest = out_dir / group / experiment
        dest.mkdir(parents=True, exist_ok=True)
        title = f"{group} / {experiment}"
        for s in labels:
            fig = subjects_figure(f"{title} / {s}", labels, [s], t0, t1, exclude_self)
            fig.savefig(dest / f"{s}.png", dpi=110)
            plt.close(fig)
        fig = aggregate_figure(title, labels, t0, t1, exclude_self)
        fig.savefig(dest / "_aggregate.png", dpi=110)
        plt.close(fig)
        print(f"{group}/{experiment}: {len(labels)} subjects -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", nargs="?")
    ap.add_argument("experiment", nargs="?")
    ap.add_argument("subject", nargs="?", help="'03', 'subject_03', or 'all'")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--exclude-self", action="store_true", help="medians over external labelers only")
    ap.add_argument("--out", help="save to this image instead of opening a window")
    ap.add_argument("--scan", action="store_true", help="ignore group/experiment/subject; save every plot to disk")
    ap.add_argument("--scan-dir", default=str(DEFAULT_SCAN_DIR))
    args = ap.parse_args()
    root = Path(args.root)

    if args.scan or args.out:
        matplotlib.use("Agg")
    if args.scan:
        scan(root, Path(args.scan_dir), args.exclude_self)
        return
    if not (args.group and args.experiment and args.subject):
        ap.error("group, experiment and subject are required unless --scan is given")

    exp_dir = root / "DIPSEER" / args.group / args.experiment
    labels = load_session(exp_dir)
    if args.subject == "all":
        chosen = list(labels)
    else:
        sid = args.subject if args.subject.startswith("subject_") else f"subject_{int(args.subject):02d}"
        if sid not in labels:
            raise SystemExit(f"{sid} not found; available: {', '.join(labels)}")
        chosen = [sid]
    span = session_span(exp_dir, labels)
    if span is None:
        raise SystemExit("no labels in this session")
    fig = subjects_figure(f"{args.group} / {args.experiment}", labels, chosen, *span, args.exclude_self)
    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"saved {args.out}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()

"""Runtime of the DIPSER recordings, from the frame timestamps (docs/experiments/01_dipser.md section 1.1).

  python -m scripts.analysis.recording_runtime          # the config's scope -> outputs/dataset/recordings.json
  python -m scripts.analysis.recording_runtime --all    # every group, experiment 01-09, camera 01-06
                                                        #   -> outputs/dataset/recordings_all.json

Scope without --all: the config's groups, experiments and cameras (52 session-cameras). Zero-byte PNGs are
dropped at indexing, as in the pipeline. fps is measured from the median frame interval; a gap is an
interval longer than 3 x the median (src/ingestion/manifest.py, timing_audit). Prints a summary, totals
per experiment, group and camera, then experiment x camera matrices of minutes (per group and all groups,
with totals) as markdown."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.core.config import load_config
from src.datasets import DipserIndex
from src.ingestion import ZipFrameSource, timing_audit

ALL = {"dataset.experiments": [f"experiment_{i:02d}" for i in range(1, 10)],
       "dataset.cameras": [f"context_cam_{i:02d}" for i in range(1, 7)]}


def summary(rows: list[dict]) -> None:
    d = np.array([r["duration_s"] for r in rows]); f = np.array([r["frames"] for r in rows])
    fps = np.array([r["fps"] for r in rows])
    g = np.array([r["gaps"] for r in rows]); gs = np.array([r["gap_s"] for r in rows])
    print(len(rows), "recordings")
    print(f"duration s: total {d.sum():.0f} ({d.sum() / 3600:.2f} h) mean {d.mean():.1f} median {np.median(d):.1f} "
          f"min {d.min():.1f} max {d.max():.1f}")
    print(f"frames: total {f.sum()} mean {f.mean():.0f} min {f.min()} max {f.max()}")
    print(f"fps: median {np.median(fps):.3f} min {fps.min():.3f} max {fps.max():.3f}")
    print(f"frames per second of recording: mean {np.mean(f / d):.3f} min {np.min(f / d):.3f} max {np.max(f / d):.3f}")
    print(f"gaps per recording: median {np.median(g):.0f} min {g.min()} max {g.max()}; gap time {gs.sum():.0f} s "
          f"({100 * gs.sum() / d.sum():.1f}%), longest {max(r['max_dt_s'] for r in rows):.1f} s")
    for field in ("experiment", "group", "camera"):
        for v in sorted({r[field] for r in rows}):
            s = [r for r in rows if r[field] == v]
            dd = np.array([r["duration_s"] for r in s]); ff = np.array([r["fps"] for r in s])
            print(f"  {v}: {len(s)} recordings, {dd.sum() / 60:.1f} min, duration {dd.min():.0f}-{dd.max():.0f} s, "
                  f"fps {ff.min():.2f}-{ff.max():.2f}")
    order = sorted(rows, key=lambda r: r["duration_s"])
    for r in order[:3] + order[-2:]:
        print("  ", r)


def matrices(rows: list[dict]) -> str:
    """Markdown experiment x camera matrices of minutes of footage, with row and column totals: one per
    group, then all groups together. "–" = no recording."""
    exps = sorted({r["experiment"] for r in rows}); cams = sorted({r["camera"] for r in rows})
    out = []
    for name, sel in [*((g, [r for r in rows if r["group"] == g]) for g in sorted({r["group"] for r in rows})),
                      ("all groups", rows)]:
        m = {(e, c): 0.0 for e in exps for c in cams}
        count = {(e, c): 0 for e in exps for c in cams}
        for r in sel:
            m[(r["experiment"], r["camera"])] += r["duration_s"] / 60
            count[(r["experiment"], r["camera"])] += 1
        has = {k for k, v in count.items() if v}
        cell = lambda e, c: f"{m[(e, c)]:.1f}" if (e, c) in has else "–"
        n = lambda keep: sum(v for k, v in count.items() if keep(k))
        out.append(f"**{name.replace('_', ' ').capitalize()}** (minutes; recordings in brackets)\n")
        out.append("| Experiment | " + " | ".join(f"cam {c[-2:]}" for c in cams) + " | **Total** |")
        out.append("|---|" + "---|" * (len(cams) + 1))
        for e in exps:
            tot = sum(m[(e, c)] for c in cams)
            out.append(f"| {e[-2:]} | " + " | ".join(cell(e, c) for c in cams)
                       + f" | **{tot:.1f}** ({n(lambda k: k[0] == e)}) |")
        col = [f"**{sum(m[(e, c)] for e in exps):.1f}** ({n(lambda k: k[1] == c)})" for c in cams]
        out.append("| **Total** | " + " | ".join(col) + f" | **{sum(m.values()):.1f}** ({len(sel)}) |\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="every experiment (01-09) and camera (01-06)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")            # the matrices use "–"; Windows consoles default to cp1252
    idx =DipserIndex.from_config(load_config("configs/deliverable3.json", ALL if args.all else None))
    rows = []
    for k in idx.sessions():
        a = timing_audit(ZipFrameSource(idx.frames_zip(k)).timestamps)
        rows.append({"group": k.group, "experiment": k.experiment, "camera": k.camera,
                     "frames": a["n_frames"], "duration_s": a["duration_s"], "fps": a["fps"],
                     "gaps": len(a["gaps"]), "gap_s": sum(x["length_s"] for x in a["gaps"]),
                     "max_dt_s": a.get("max_dt_s", 0.0)})
    out = Path("outputs/dataset") / ("recordings_all.json" if args.all else "recordings.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    summary(rows)
    print()
    print(matrices(rows))


if __name__ == "__main__":
    main()

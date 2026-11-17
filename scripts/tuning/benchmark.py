"""Time the extraction pipeline per stage, for several tracker variants and sessions (design §11 compute).

  python -m scripts.tuning.benchmark [--name NAME] [--variants V ...] [--sessions g/e/c ...] [--max-frames N]
                              [--set key=value ...]

Each (variant, session) runs a fresh extraction (Pipeline.extract: both passes, nothing cached)
and appends one row to outputs/benchmarks/<name>.csv as soon as it finishes: per-stage seconds,
frames, wall time, and the machine's average CPU load during the run (other jobs inflate the
CPU-bound stages; the load column makes such rows visible). Re-running the same command skips the
(variant, session) pairs already logged, so it resumes after an interruption. At the end it writes
outputs/benchmarks/<name>.md: per variant, mean ms/frame per stage, throughput, and the time
projected for a 45-minute session at the measured frame rate.

Variants and default sessions live in the config's `benchmark` section; each variant is a set of
dotted config overrides (e.g. {"tracker.options.use_embeddings": true}). Model loading happens
once per variant before its first session and is reported separately (`load_s`).
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import pandas as pd
import psutil

from scripts.run_pipeline import parse_sets
from src.core.config import load_config, set_dotted
from src.core.pipeline import Pipeline
from src.core.types import SessionKey
from src.datasets import DipserIndex

TARGET_SESSION_MIN = 45          # design §11: a 45-minute single-camera session in <= 3 minutes


def variant_config(cfg: dict, overrides: dict) -> dict:
    out = copy.deepcopy(cfg)
    for k, v in overrides.items():
        set_dotted(out, k, v)
    return out


def summarize(df: pd.DataFrame, path: Path) -> None:
    stage_cols = [c for c in df if c.startswith("s.")]
    lines = [f"# Benchmark: {path.stem}", "",
             f"Fresh extraction per (variant, session); {df['session'].nunique()} sessions. "
             "ms/frame per stage (mean over sessions). CPU-bound stages are inflated when `cpu_load` is high.", ""]
    rows = []
    for variant, g in df.groupby("variant", sort=False):
        frames = g["frames"].sum()
        per_stage = {c[2:]: 1000 * g[c].fillna(0).sum() / frames for c in stage_cols if g[c].notna().any()}
        ms_frame = 1000 * g["frame_pass_s"].sum() / frames
        fps = g["frames"].sum() / g["wall_s"].sum()
        capture_fps = (g["frames"] / g["duration_s"]).mean()
        projected_min = TARGET_SESSION_MIN * 60 * capture_fps * ms_frame / 1000 / 60
        rows.append({"variant": variant, "sessions": len(g), "ms/frame": round(ms_frame, 1),
                     "processing fps": round(fps, 2), "45-min session (min)": round(projected_min, 1),
                     "cpu load %": round(g["cpu_load"].mean()), "load_s": round(g["load_s"].max(), 1),
                     **{k: round(v, 1) for k, v in per_stage.items()}})
    table = pd.DataFrame(rows)
    lines.append(table.to_markdown(index=False))
    lines += ["", f"Target (design §11): a {TARGET_SESSION_MIN}-minute session in 3 minutes or less."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + table.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--name", default="trackers")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--max-frames", type=int, help="cap frames per session (smoke tests)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    bench = cfg["benchmark"]
    if args.max_frames:
        cfg["ingestion"]["max_frames"] = args.max_frames
    variants = {k: bench["variants"][k] for k in (args.variants or bench["variants"])}
    keys = [SessionKey(*s.split("/")) for s in (args.sessions or bench["sessions"])]
    index = DipserIndex.from_config(cfg)

    out_dir = Path(cfg["output_dir"]) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.name}.csv"
    done = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        done = set(zip(prev["variant"], prev["session"]))
        print(f"resuming: {len(done)} runs already in {csv_path.name}")

    total = len(variants) * len(keys)
    n = 0
    for vname, overrides in variants.items():
        todo = [k for k in keys if (vname, str(k)) not in done]
        n += len(keys) - len(todo)
        if not todo:
            continue
        vcfg = variant_config(cfg, overrides)
        t0 = time.perf_counter()
        pipe = Pipeline(vcfg)
        pipe.detector                                       # load weights now, timed as load_s
        for e in pipe.extractors:
            if hasattr(e, "detectors"):
                e.detectors
        load_s = time.perf_counter() - t0
        for key in todo:
            n += 1
            print(f"[{n}/{total}] {vname} | {key} ...", flush=True)
            psutil.cpu_percent(interval=None)               # prime: next call averages over the run
            t0 = time.perf_counter()
            feats = pipe.extract(index.frames_zip(key), key)
            wall = time.perf_counter() - t0
            cpu = psutil.cpu_percent(interval=None)
            stages = feats.manifest["stage_seconds"]
            frame_pass = sum(v for k, v in stages.items() if k not in ("tracklets", "aggregation"))
            row = {"variant": vname, "session": str(key), "frames": feats.manifest["timing"]["n_frames"],
                   "duration_s": feats.manifest["timing"]["duration_s"], "wall_s": round(wall, 2),
                   "frame_pass_s": round(frame_pass, 2), "load_s": round(load_s, 2), "cpu_load": cpu,
                   "n_tracklets": feats.manifest["n_tracklets"],
                   **{f"s.{k}": round(v, 3) for k, v in stages.items()}}
            log = pd.concat([pd.read_csv(csv_path), pd.DataFrame([row])], ignore_index=True)                 if csv_path.exists() else pd.DataFrame([row])
            log.to_csv(csv_path, index=False)               # rewritten whole: stage columns may differ
            print(f"    {row['frames']} frames, wall {wall:.0f}s ({1000 * frame_pass / row['frames']:.0f} ms/frame), "
                  f"cpu load {cpu:.0f}% | " + " ".join(f"{k}={v:.1f}" for k, v in stages.items()), flush=True)

    df = pd.read_csv(csv_path)
    df = df[df["variant"].isin(variants)]
    summarize(df, out_dir / f"{args.name}.md")
    print(f"\nlog: {csv_path}\nsummary: {out_dir / (args.name + '.md')}")


if __name__ == "__main__":
    main()

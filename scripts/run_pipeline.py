"""Run the pipeline.

  python -m scripts.run_pipeline extract [--session group_01/experiment_01/context_cam_01] [--reuse]
                                         [--set key=value ...]
      Both passes for every in-scope session (or one). Writes under <output_dir>:
        tracks/<tracking hash>/<session>.npz     frame pass (detection, tracking, pixel signals, scene)
        features/<feature hash>/<session>.csv    window rows (Vec + scene features), + .manifest.json
        features/<feature hash>/<session>.tracks.csv   per-student rows (window, tid, trk.* features)
      By default nothing is read back: every session runs the full frame pass, so the stage timings
      in each manifest are real runtime. --reuse (development) skips sessions whose features exist
      and loads cached tracks, so changing only feature-pass settings (window, tracklets,
      aggregation, window-level BFMU, CFMU) takes seconds; such manifests say frame_pass_cached.

  python -m scripts.run_pipeline infer --zip path/to/context_cam_XX.zip --models models.joblib --out dir
      Full inference on one camera ZIP -> retention_graph.json, retention_graph.png, manifest.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.core.config import config_hash, load_config
from src.core.pipeline import Pipeline, feature_hash, to_frame, to_track_frame, tracking_hash
from src.core.store import load_track, save_track
from src.core.types import SessionKey
from src.datasets import DipserIndex
from src.ingestion import aug_tag, sample_augmentation
from src.output import write_graph_json, write_graph_png, write_manifest


def parse_sets(pairs: list[str]) -> dict:
    """`key=value` overrides; the value is JSON, or a plain string when it is not valid JSON (so
    `mixture.expert_scope=shared` works: Windows PowerShell 5.1 strips the quotes of '"shared"')."""
    def value(v: str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return {k: value(v) for k, v in (p.split("=", 1) for p in pairs)}


def tracks_dir(cfg: dict) -> Path:
    return Path(cfg["output_dir"]) / "cache" / "tracks" / tracking_hash(cfg)[:12]


def features_dir(cfg: dict) -> Path:
    return Path(cfg["output_dir"]) / "cache" / "features" / feature_hash(cfg)[:12]


def session_stem(key: SessionKey) -> str:
    return f"{key.group}__{key.experiment}__{key.camera}"


def session_track(pipe: Pipeline, index: DipserIndex, key: SessionKey, reuse: bool,
                  stem: str | None = None, augment: dict | None = None):
    """Run the session's frame pass and cache it as <stem>.npz; with reuse, load an existing cache
    entry instead. augment: an augmented copy's parameters, applied as frames are read (never written).
    Returns (track, was_cached)."""
    path = tracks_dir(pipe.cfg) / f"{stem or session_stem(key)}.npz"
    if reuse and path.exists():
        return load_track(path), True
    st = pipe.track(index.frames_zip(key), key, augment)
    save_track(st, path)
    return st, False


def cmd_extract(cfg: dict, session: str | None, reuse: bool) -> None:
    """All sessions first, then (if augmentation.copies > 0) every session's augmented copies, each
    under `<stem>.<aug tag>` next to the originals. Copies are for training only."""
    index = DipserIndex.from_config(cfg)
    keys = [SessionKey(*session.split("/"))] if session else index.sessions()
    out = features_dir(cfg)
    for d in (out, tracks_dir(cfg)):
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    pipe = Pipeline(cfg)
    jobs = extract_jobs(cfg, keys)
    for i, (key, copy) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] " + extract_one(cfg, pipe, index, key, copy, reuse))


def extract_jobs(cfg: dict, keys: list) -> list[tuple]:
    """(session, copy) extraction jobs: every original (copy None), then augmented copies 1..N."""
    aug = cfg.get("augmentation", {})
    return [(key, None) for key in keys] + [(key, c) for c in range(1, aug.get("copies", 0) + 1) for key in keys]


def extract_one(cfg: dict, pipe: Pipeline, index: DipserIndex, key: SessionKey, copy: int | None,
                reuse: bool) -> str:
    """Frame pass + feature pass of one session (or one augmented copy of it), written next to the
    others; returns a one-line report. Shared by `extract` and scripts/parallel_runs.py."""
    out, aug = features_dir(cfg), cfg.get("augmentation", {})
    stem = session_stem(key) + ("" if copy is None else f".{aug_tag(aug, copy)}")
    label = f"{key}" + ("" if copy is None else f" (augmented copy {copy})")
    if reuse and (out / f"{stem}.csv").exists() and (out / f"{stem}.tracks.csv").exists():
        return f"{label}: features cached"
    params = None if copy is None else sample_augmentation(aug, str(key), copy)   # augmented on the fly
    st, cached = session_track(pipe, index, key, reuse, stem, params)
    feats = pipe.features(st)
    feats.manifest["frame_pass_cached"] = cached
    if params is not None:
        feats.manifest["augmentation"] = {**params, "tag": aug_tag(aug, copy), "frame_wh": list(st.frame_wh)}
    (out / f"{stem}.manifest.json").write_text(json.dumps(feats.manifest, indent=2, default=str), encoding="utf-8")
    to_track_frame(feats).to_csv(out / f"{stem}.tracks.csv", index=False)   # per-student rows (late aggregation)
    to_frame(feats).to_csv(out / f"{stem}.csv", index=False)   # written last: marks the session done
    t, secs = st.timing, feats.manifest["stage_seconds"]
    frame_pass = "cached" if cached else f"{sum(st.stage_seconds.values()):.0f}s"
    return (f"{label}: {t['n_frames']} frames @ {t['fps']:.2f} fps, frame pass {frame_pass}, "
            f"{len(feats.windows)} windows, {feats.manifest['n_tracklets']} tracklets "
            f"({feats.manifest['n_raw_tracks']} raw) | " + " ".join(f"{k}={v:.1f}" for k, v in secs.items()))


def cmd_infer(cfg: dict, zip_path: str, models_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = joblib.load(models_path)
    if models["config_hash"] != config_hash(cfg):
        print("WARNING: models were trained under a different config hash")
    pipe = Pipeline(cfg)
    feats = pipe.extract(zip_path)
    records = pipe.predict(feats, models)
    write_graph_json(records, out)
    if records:
        write_graph_png(records, out, cfg["output"]["smooth_windows"])
    write_manifest({**feats.manifest, "config": cfg, "models": models_path,
                    "label_offset_s": cfg["labels"]["offset_s"]}, out)
    print(f"wrote {len(records)} windows to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["extract", "infer"])
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, JSON values: window.length_s=10")
    ap.add_argument("--session")
    ap.add_argument("--reuse", action="store_true",
                    help="development: reuse cached tracks/features (timings then are not runtime)")
    ap.add_argument("--zip")
    ap.add_argument("--models")
    ap.add_argument("--out", default="outputs/run")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    if args.command == "extract":
        cmd_extract(cfg, args.session, args.reuse)
    else:
        cmd_infer(cfg, args.zip, args.models, args.out)


if __name__ == "__main__":
    main()

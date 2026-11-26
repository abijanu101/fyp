"""Frame pass + feature pass for OUC-CGE clips, with the same Pipeline the DIPSER runs use.

  python -m scripts.ouc_cge.extract [--config configs/ouc_cge.json] [--split test] [--limit 30]
                                    [--shard 1/4] [--reuse] [--max-hours 10] [--set key=value ...]

Per clip (stem `ouc_cge__<level>__<viewN>`), under <output_dir>/cache/:
  tracks/<tracking hash>/<stem>.npz              frame pass (as run_pipeline extract)
  features/<feature hash>/ouc_cge/<stem>.csv     ONE row: the whole clip as one window (Vec + scene)
  features/<feature hash>/ouc_cge/<stem>.tracks.csv, <stem>.manifest.json
The clip is the unit because OUC-CGE has one label per clip (config `ouc_cge.window` = "clip").

--limit N takes N clips spread evenly over the three levels (smoke tests); --shard i/n runs every
n-th clip starting at i (1-based), so n terminals or machines can split the work; --reuse skips
clips already done (resume after a stop); --max-hours stops cleanly after that long. A clip that
fails is logged to errors.txt in the features folder and skipped, so one bad file never stops a run.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from scripts.run_pipeline import features_dir, parse_sets, tracks_dir
from src.core.config import load_config
from src.core.pipeline import Pipeline, to_frame, to_track_frame
from src.core.store import load_track, save_track
from src.core.types import SessionFeatures, Window
from src.datasets.ouc_cge import LEVEL_NAMES, OucCgeIndex


def clip_stem(clip) -> str:
    return f"ouc_cge__{clip.level}__{clip.stem}"


def out_dir(cfg: dict) -> Path:
    return features_dir(cfg) / "ouc_cge"


def spread(clips: list, n: int) -> list:
    """n clips taken round-robin over the levels (low, mid, high, low, ...), in file order."""
    by = {lv: [c for c in clips if c.level == lv] for lv in LEVEL_NAMES}
    picked, i = [], 0
    while len(picked) < n and any(i < len(v) for v in by.values()):
        for lv in LEVEL_NAMES:
            if i < len(by[lv]) and len(picked) < n:
                picked.append(by[lv][i])
        i += 1
    return picked


def clip_features(pipe: Pipeline, st) -> SessionFeatures:
    """The whole clip as one window (window = "clip"), or the config's regular grid ("grid")."""
    if pipe.cfg.get("ouc_cge", {}).get("window", "clip") == "grid":
        return pipe.features(st)
    t = st.timing
    fps = t.get("fps") or 1.0
    windows = [Window(t.get("t_first", 0.0), t.get("t_last", 0.0) + 1.0 / fps)] if t.get("n_frames", 0) else []
    vec, track_rows, scene = pipe.window_features(st, windows)
    feats = pipe.features(st)                 # for its manifest (tracklet counts, hashes, timings)
    return SessionFeatures(st.key, windows, vec, scene, feats.manifest, track_rows)


def extract_clip(pipe: Pipeline, index: OucCgeIndex, clip, reuse: bool) -> str:
    stem, out = clip_stem(clip), out_dir(pipe.cfg)
    if reuse and (out / f"{stem}.csv").exists():
        return f"{clip.clip_id}: cached"
    tpath = tracks_dir(pipe.cfg) / f"{stem}.npz"
    cached = reuse and tpath.exists()
    st = load_track(tpath) if cached else pipe.track(index.video(clip), clip.key)
    if not cached:
        save_track(st, tpath)
    feats = clip_features(pipe, st)
    feats.manifest.update({"clip_id": clip.clip_id, "label": clip.label, "split": clip.split,
                           "video": str(index.video(clip)), "frame_pass_cached": cached})
    (out / f"{stem}.manifest.json").write_text(json.dumps(feats.manifest, indent=2, default=str), encoding="utf-8")
    to_track_frame(feats).to_csv(out / f"{stem}.tracks.csv", index=False)
    df = to_frame(feats)
    df.insert(0, "clip_id", clip.clip_id)
    df.to_csv(out / f"{stem}.csv", index=False)          # written last: marks the clip done
    secs = sum(st.stage_seconds.values())
    return (f"{clip.clip_id}: {st.timing['n_frames']} frames @ {st.timing['fps'] or 0:.2f} fps, "
            f"frame pass {'cached' if cached else f'{secs:.1f}s'}, {feats.manifest['n_tracklets']} tracklets")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ouc_cge.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--split", choices=["train", "val", "test"], action="append",
                    help="repeatable; default: every split in dataset.splits")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", default="1/1", help="i/n: run every n-th clip starting at i")
    ap.add_argument("--reuse", action="store_true", help="skip finished clips (resume)")
    ap.add_argument("--max-hours", type=float)
    a = ap.parse_args()
    cfg = load_config(a.config, parse_sets(a.set))
    index = OucCgeIndex.from_config(cfg)
    clips = [c for c in index.clips() if not a.split or c.split in a.split]
    if index.missing():
        print(f"WARNING: {len(index.missing())} split clips have no video under {index.root}")
    if a.limit:
        clips = spread(clips, a.limit)
    i, n = map(int, a.shard.split("/"))
    clips = clips[i - 1::n]

    out = out_dir(cfg)
    for d in (out, tracks_dir(cfg)):
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"{len(clips)} clips -> {out}")
    pipe, t0, failed = Pipeline(cfg), time.time(), 0
    for k, clip in enumerate(clips, 1):
        if a.max_hours and time.time() - t0 > a.max_hours * 3600:
            print(f"stopping after {a.max_hours} h at clip {k}/{len(clips)}; rerun with --reuse to continue")
            break
        try:
            print(f"[{k}/{len(clips)}] " + extract_clip(pipe, index, clip, a.reuse), flush=True)
        except Exception as e:        # noqa: BLE001  one bad clip must not stop a long run
            failed += 1
            with open(out / "errors.txt", "a", encoding="utf-8") as f:
                f.write(f"{clip.clip_id}\t{index.video(clip)}\t{e!r}\n{traceback.format_exc()}\n")
            print(f"[{k}/{len(clips)}] {clip.clip_id}: FAILED ({e!r}); see errors.txt", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {failed} failed")


if __name__ == "__main__":
    main()

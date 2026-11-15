"""Materialise augmented copies of DIPSER on disk (config `augmentation`), as complete datasets.

  python -m scripts.augment_dataset [--copies 1 2] [--set dataset.groups=[...] ...]

Copy k of the configured augmentation is written to <augmentation.root>/<aug tag>/ with the same
layout as data/DIPSER, so every tool can run on it by pointing `dataset.root` there:
  DIPSEER/<group>/<experiment>/context_cam_ZZ.zip   augmented frames (JPEG, same timestamp names)
  DIPSEER/<group>/<experiment>/subjects/*           the label archives, hard-linked (unchanged)
  abi_labels/<group>/<experiment>.json              manual boxes, mirrored where the copy is flipped
  abi_labels/experiments.tsv, cameras.yml           copied
  augmentation.json                                 settings and every session's exact parameters
One parameter set per (session, copy) applies to every frame (src/ingestion/augment.py).
Resumable per camera ZIP; `run_pipeline extract` calls `ensure_session` for any copy it needs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from scripts.run_pipeline import parse_sets
from src.core.config import load_config
from src.core.types import SessionKey
from src.datasets import DipserIndex
from src.ingestion import aug_tag, augmented_root, hflip_boxes, sample_params, write_augmented_zip

DEFAULT_WORKERS = max(1, min(6, (os.cpu_count() or 2) // 2))   # leave cores for other running jobs


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)                     # same drive: no extra space
    except OSError:
        shutil.copy2(src, dst)


def _manifest(root: Path, aug: dict, copy: int) -> dict:
    path = root / "augmentation.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"tag": aug_tag(aug, copy), "copy": copy, "settings": {k: v for k, v in aug.items() if k != "copies"},
            "sessions": {}}


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _zip_path(aug: dict, key: SessionKey, copy: int) -> Path:
    return augmented_root(aug, copy) / "DIPSEER" / key.group / key.experiment / f"{key.camera}.zip"


def write_frames(cfg: dict, key: SessionKey, copy: int) -> dict:
    """Worker-safe: write the augmented camera ZIP only (no shared files). Returns its parameters
    with the frame width."""
    aug = cfg["augmentation"]
    params = sample_params(aug, str(key), copy)
    width = write_augmented_zip(DipserIndex.from_config(cfg).frames_zip(key), _zip_path(aug, key, copy),
                                params, aug.get("jpeg_quality", 95))
    return {**params, "frame_width": width}


def ensure_many(cfg: dict, jobs: list[tuple[SessionKey, int]], workers: int = 1, log=print) -> None:
    """ensure_session for many (session, copy) jobs, writing missing frame ZIPs in `workers` parallel
    processes (CPU-bound: decode, augment, JPEG-encode). Shared files are written here only."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    aug = cfg["augmentation"]
    todo = [(k, c) for k, c in jobs if str(k) not in _manifest(augmented_root(aug, c), aug, c)["sessions"]
            or not _zip_path(aug, k, c).exists()]
    for k, c in jobs:
        if (k, c) not in todo:
            ensure_session(cfg, k, c)                         # labels/boxes only (cheap, idempotent)
    if not todo:
        return
    log(f"writing {len(todo)} augmented session copies with {workers} worker(s)")
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(write_frames, cfg, k, c): (k, c) for k, c in todo}
        for n, f in enumerate(as_completed(futures), 1):
            k, c = futures[f]
            dst, p = ensure_session(cfg, k, c, entry=f.result())
            log(f"  [{n}/{len(todo)}] copy {c} {k}: flip={p['hflip']} b={p['brightness']:+.0f} "
                f"noise={p['noise_kind']} {p['noise']:.2f} masks={len(p['mask_rects'])} "
                f"({dst.stat().st_size / 2 ** 20:.0f} MB)")


def ensure_session(cfg: dict, key: SessionKey, copy: int, entry: dict | None = None) -> tuple[Path, dict]:
    """Make sure copy `copy` of session `key` exists on disk; return (its camera ZIP, its parameters
    with the frame width). Also records its parameters, links its labels and writes its manual
    boxes (mirrored if flipped). `entry`: the result of write_frames when a worker already wrote it."""
    aug = cfg["augmentation"]
    index = DipserIndex.from_config(cfg)
    root = augmented_root(aug, copy)
    rel = Path("DIPSEER") / key.group / key.experiment
    dst = _zip_path(aug, key, copy)
    manifest = _manifest(root, aug, copy)
    if entry is None:
        entry = manifest["sessions"].get(str(key))
        if entry is None or not dst.exists():
            entry = write_frames(cfg, key, copy)
    if manifest["sessions"].get(str(key)) != entry:
        manifest["sessions"][str(key)] = entry
        _write_json(root / "augmentation.json", manifest)

    # labels: link the label archives, write this camera's (possibly mirrored) manual boxes
    for f in (index.raw / key.group / key.experiment / "subjects").glob("*"):
        _link(f, root / rel / "subjects" / f.name)
    for f in index.annotations_dir.glob("*.*"):
        if not (root / "abi_labels" / f.name).exists():
            (root / "abi_labels").mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, root / "abi_labels" / f.name)
    ann_path = root / "abi_labels" / key.group / f"{key.experiment}.json"
    ann = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    boxes = {}
    for sid, b in index.annotations(key).items():
        if entry["hflip"]:
            b = hflip_boxes(b[None], entry["frame_width"])[0]
        boxes[sid] = {"x": float(b[0]), "y": float(b[1]), "w": float(b[2] - b[0]), "h": float(b[3] - b[1])}
    if ann.get(key.camera) != boxes:
        ann[key.camera] = boxes
        _write_json(ann_path, ann)
    return dst, entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--copies", nargs="*", type=int, help="which copies (default 1..augmentation.copies)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel writer processes")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    copies = args.copies or list(range(1, cfg["augmentation"]["copies"] + 1))
    if not copies:
        raise SystemExit("augmentation.copies is 0; pass --copies 1 2 or --set augmentation.copies=2")
    keys = DipserIndex.from_config(cfg).sessions()
    for c in copies:
        print(f"== copy {c} -> {augmented_root(cfg['augmentation'], c)}", flush=True)
    ensure_many(cfg, [(k, c) for c in copies for k in keys], args.workers, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()

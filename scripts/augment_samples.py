"""Save augmentation samples for visual review (config `augmentation`).

  python -m scripts.augment_samples [--sessions g/e/c ...] [--copies 4] [--at 30] [--set ...]

For each session, one frame (--at seconds in) rendered through AugmentedZipFrameSource, the same
code the training copies use, with the manual student boxes drawn (mirrored when flipped):
  original.png        the frame as recorded
  copy_<k>.png        the parameters augmented copy k of this session gets (sample_params)
  extreme_bright.png  every setting at the bright end of its range at once, maximum noise, the
                      maximum number of the largest masks, flipped
  extreme_dark.png    the same at the dark end
plus sheet.png (all of them side by side) and params.json (the exact parameters of each image).
Output: <output_dir>/augmentation/<augmentation settings hash>/<session>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.run_pipeline import parse_sets
from src.core.config import config_hash, load_config
from src.core.types import SessionKey
from src.datasets import DipserIndex
from src.ingestion import AugmentedZipFrameSource, ZipFrameSource, aug_tag, hflip_boxes, sample_params

DEFAULT_SESSIONS = ["group_01/experiment_03/context_cam_02", "group_02/experiment_05/context_cam_01",
                    "group_03/experiment_06/context_cam_03"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def extreme(aug: dict, bright: bool) -> dict:
    """All ranges at one end simultaneously: the most a single copy could ever be changed."""
    s = 1 if bright else -1
    m, n = aug.get("mask", {}), aug.get("noise", {})
    count, size = m.get("count", [1, 1])[1], m.get("size", [0.05, 0.1])[1]
    rects = [((i % 3) * 0.33 + 0.02, (i // 3) * 0.4 + 0.1, size, size) for i in range(count)]
    return {"copy": "extreme_" + ("bright" if bright else "dark"), "hflip": aug.get("hflip", 0) > 0,
            "brightness": s * aug.get("brightness", 0), "contrast": 1 + s * aug.get("contrast", 0),
            "saturation": 1 + s * aug.get("saturation", 0), "hue_deg": s * aug.get("hue_deg", 0),
            "gamma": (1 + aug.get("gamma", 0)) ** s, "noise_kind": n.get("kinds", ["gaussian"])[-1],
            "noise": n.get("max", 0), "mask_kind": m.get("kind", "none") if count else "none",
            "mask_rects": rects, "noise_seed": 0}


def draw(img: np.ndarray, boxes: np.ndarray, title: str, params: dict | None) -> np.ndarray:
    img = img.copy()
    for b in boxes.astype(int):
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 220, 0), 2)
    lines = [title]
    if params:
        lines.append(f"flip {params['hflip']}  b {params['brightness']:+.1f}  c {params['contrast']:.2f}  "
                     f"s {params['saturation']:.2f}  hue {params['hue_deg']:+.1f}  gamma {params['gamma']:.2f}")
        lines.append(f"{params['noise_kind']} noise {params['noise']:.3f}  "
                     f"{params['mask_kind']} x{len(params['mask_rects'])}")
    for i, line in enumerate(lines):
        y = 34 + i * 30
        cv2.putText(img, line, (12, y), FONT, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (12, y), FONT, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--sessions", nargs="*", default=DEFAULT_SESSIONS)
    ap.add_argument("--copies", type=int, default=4, help="how many sampled copies to show per session")
    ap.add_argument("--at", type=float, default=30.0, help="seconds into the session")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    aug = cfg["augmentation"]
    index = DipserIndex.from_config(cfg)
    root = Path(cfg["output_dir"]) / "augmentation" / aug_tag(aug, 0).split("-", 1)[1]   # settings hash

    for s in args.sessions:
        key = SessionKey(*s.split("/"))
        zip_path = index.frames_zip(key)
        base = ZipFrameSource(zip_path)
        i = int(np.searchsorted(base.timestamps, base.timestamps[0] + args.at))
        ann = index.annotations(key)
        boxes = np.stack(list(ann.values())) if ann else np.zeros((0, 4))
        width = base.read(i).image.shape[1]
        variants = [("original", None)] + [(f"copy_{c}", sample_params(aug, str(key), c))
                                           for c in range(1, args.copies + 1)]
        variants += [("extreme_bright", extreme(aug, True)), ("extreme_dark", extreme(aug, False))]
        out = root / s.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        tiles, record = [], {}
        for name, p in variants:
            frame = base.read(i) if p is None else AugmentedZipFrameSource(zip_path, p)[i]   # torch Dataset access
            b = hflip_boxes(boxes, width) if p and p["hflip"] and len(boxes) else boxes
            img = draw(frame.image, b, f"{s}  {name}", p)
            cv2.imwrite(str(out / f"{name}.png"), img)
            tiles.append(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA))
            record[name] = p
        cols = 3
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
        cv2.imwrite(str(out / "sheet.png"), sheet)
        (out / "params.json").write_text(json.dumps({"augmentation": aug, "frame_index": i, "images": record},
                                                    indent=2), encoding="utf-8")
        print(f"{s}: {len(variants)} images -> {out}")


if __name__ == "__main__":
    main()

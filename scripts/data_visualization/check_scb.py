"""SCB behaviour-detector sanity check on DIPSER (design §11; gates trusting BFMU (c)).

  python -m scripts.data_visualization.check_scb [--frames 12] [--conf 0.25] [--seed 0] [--set key=value ...]

Samples frames from the in-scope sessions (seeded, spread over groups/experiments/cameras), runs
every SCB model configured in the `scb` BFMU entry, and writes one annotated image per frame to
<output_dir>/scb_check/ plus counts.csv (detections per class per frame). The check is a human
precision read: open the images and judge whether boxes land on the right students with the
right behaviour. Person boxes from the pipeline's detector are drawn thin grey for reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.run_pipeline import parse_sets
from src.bfmu.scb import normalize_class
from src.bfmu.yolov7 import Yolov7Detector
from src.core.config import load_config
from src.core.pipeline import Pipeline
from src.datasets import DipserIndex
from src.ingestion import ZipFrameSource

COLOURS = [(0, 200, 255), (0, 220, 0), (255, 120, 0), (200, 0, 200), (0, 0, 255), (255, 255, 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deliverable3.json")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--conf", type=float, help="override every model's confidence threshold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_sets(args.set))
    spec = next(e for e in cfg["bfmu"] if e["name"] == "scb")
    models = []
    for m in spec["models"]:
        det = Yolov7Detector(m["weights"], conf=args.conf or m.get("conf", 0.25))
        found = [normalize_class(n) for n in det.names]
        print(f"{m['weights']}: classes {found}")
        models.append((det, found))
    person = Pipeline(cfg).detector

    index = DipserIndex.from_config(cfg)
    rng = np.random.default_rng(args.seed)
    keys = index.sessions()
    picks = [keys[i] for i in rng.choice(len(keys), size=min(args.frames, len(keys)), replace=False)]
    out = Path(cfg["output_dir"]) / "scb_check"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in picks:
        src = ZipFrameSource(index.frames_zip(key))
        i = int(rng.integers(len(src.timestamps)))
        frame = src.read(i)
        if frame is None:
            continue
        img = frame.image.copy()
        for b in person.detect(frame).xyxy:
            cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (160, 160, 160), 1)
        counts = {"session": str(key), "frame": i}
        k = 0
        for det, names in models:
            xyxy, conf, cls = det.detect(frame.image)
            for ci, name in enumerate(names):
                m = cls == ci
                counts[name] = int(m.sum())
                col = COLOURS[k % len(COLOURS)]
                k += 1
                for b, s in zip(xyxy[m], conf[m]):
                    cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), col, 2)
                    cv2.putText(img, f"{name} {s:.2f}", (int(b[0]), max(int(b[1]) - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        rows.append(counts)
        name = f"{key.group}__{key.experiment}__{key.camera}__{i:05d}.jpg"
        cv2.imwrite(str(out / name), img)
        print(counts)
    pd.DataFrame(rows).to_csv(out / "counts.csv", index=False)
    print(f"wrote {len(rows)} images to {out}")


if __name__ == "__main__":
    main()

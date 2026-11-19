"""Fetch every model the pipeline needs, so a fresh clone can run. Safe to re-run: what is already
there is skipped.

  python -m scripts.setup            # download what is missing
  python -m scripts.setup --check    # only report what is there and what is missing

Run from the repository root, in the project environment (conda env "fyp-local", README "Setup").

What it sets up:
  models/yolo11n.pt          person detector (detector.weights; experiments up to 08)
  models/yolo11s-pose.pt     pose model: detector and head-pose keypoints of experiment 09
  models/yolo11n-pose.pt     the head-pose unit's default weights, and the pilot's nano variant
  models/osnet_x0_25_msmt17.pt   the tracker's ReID model (BoT-SORT with use_embeddings), fetched by
                             BoxMOT itself, the same way the pipeline loads it
  data/SCB-ST-Dataset4/SCB5-Handrise-Read-write-2024-9-17/exp/weights/best.pt
                             the SCB behaviour model (bfmu "scb"): only this checkpoint, not the whole
                             SCB dataset (scripts/downloading/scb.py fetches all of it)
  lib/yolov7/                WongKinYiu/yolov7 at a207844: the model code the SCB checkpoint needs

Not covered: the DIPSER data (needs the dataset authors' FTP credentials; scripts/downloading/dipser.py)
and trained models (rebuilt with scripts.train_models).

At the end it lists every *.pt path named in configs/*.json and docs/experiments/*.json and flags any
that is still missing, so a new model in a config cannot be forgotten here.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ULTRALYTICS = "https://github.com/ultralytics/assets/releases/download/v8.3.0/{}"
YOLO_WEIGHTS = ["yolo11n.pt", "yolo11s-pose.pt", "yolo11n-pose.pt"]

SCB_REPO = "wintonYF/SCB-Dataset"             # Hugging Face dataset
SCB_FILE = "SCB5-Handrise-Read-write-2024-9-17/exp/weights/best.pt"
SCB_DIR = ROOT / "data" / "SCB-ST-Dataset4"

YOLOV7_URL, YOLOV7_COMMIT = "https://github.com/WongKinYiu/yolov7", "a207844"
YOLOV7_DIR = ROOT / "lib" / "yolov7"

REID = ROOT / "models" / "osnet_x0_25_msmt17.pt"


def have(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def download(url: str, dest: Path) -> None:
    """Download to a temporary name, then rename: an interrupted download never looks complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def yolo(name: str, check: bool) -> bool:
    dest = ROOT / "models" / name
    if have(dest):
        return True
    if check:
        return False
    print(f"  downloading {name}")
    download(ULTRALYTICS.format(name), dest)
    return have(dest)


def reid(check: bool) -> bool:
    """Let BoxMOT fetch its ReID model exactly as the pipeline's tracker does on first use (it lands
    in models/ when run from the repository root)."""
    if have(REID) or check:
        return have(REID)
    print("  BoxMOT is fetching its ReID model")
    import numpy as np
    from src.tracking import TRACKERS
    tracker = TRACKERS.build("botsort", use_embeddings=True)
    tracker.tracker.update(np.array([[10, 10, 100, 200, 0.9, 0]], float), np.zeros((360, 640, 3), np.uint8))
    return have(REID)


def scb(check: bool) -> bool:
    dest = SCB_DIR / SCB_FILE
    if have(dest) or check:
        return have(dest)
    print(f"  downloading {SCB_FILE} from {SCB_REPO}")
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=SCB_REPO, repo_type="dataset", filename=SCB_FILE, local_dir=str(SCB_DIR))
    return have(dest)


def yolov7(check: bool) -> bool:
    ok = (YOLOV7_DIR / "models" / "yolo.py").is_file()
    if ok or check:
        return ok
    print(f"  cloning {YOLOV7_URL} at {YOLOV7_COMMIT}")
    YOLOV7_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", YOLOV7_URL, str(YOLOV7_DIR)], check=True)
    subprocess.run(["git", "-C", str(YOLOV7_DIR), "checkout", "-q", YOLOV7_COMMIT], check=True)
    return (YOLOV7_DIR / "models" / "yolo.py").is_file()


def configured_weights() -> set[str]:
    """Every *.pt path named in the configs and the frozen experiment configs."""
    paths = set()
    for f in [*ROOT.glob("configs/*.json"), *ROOT.glob("docs/experiments/*.json")]:
        paths |= set(re.findall(r'"([^"]+\.pt)"', f.read_text(encoding="utf-8")))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report what is there and what is missing")
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        sys.exit(f"run from the repository root ({ROOT}): BoxMOT puts its ReID model under ./models")

    steps = [*((f"models/{n}", lambda n=n: yolo(n, args.check)) for n in YOLO_WEIGHTS),
             ("models/osnet_x0_25_msmt17.pt (ReID)", lambda: reid(args.check)),
             (f"SCB checkpoint ({SCB_FILE})", lambda: scb(args.check)),
             (f"lib/yolov7 @ {YOLOV7_COMMIT}", lambda: yolov7(args.check))]
    failed = []
    for label, step in steps:
        try:
            ok = step()
        except Exception as e:                      # keep going: report every failure at the end
            print(f"  error: {e}")
            ok = False
        print(f"[{'ok' if ok else 'missing'}] {label}")
        if not ok:
            failed.append(label)

    missing = sorted(p for p in configured_weights() if "..." not in p and not have(ROOT / p))
    for p in missing:
        print(f"[missing] {p}  (named in a config; not something this script fetches)")
    if failed or missing:
        sys.exit(1)
    print("\nAll models are in place. Next: the DIPSER data (README, Setup > Data).")


if __name__ == "__main__":
    main()

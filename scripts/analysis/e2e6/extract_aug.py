"""Augmented copies of the six-session run (2026-11-16): frame pass + features per copy, next to the
cached originals (the run's own config; new keys from the current one).
  python scripts/analysis/e2e6/extract_aug.py [copies]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import scripts.run_pipeline as rp
from src.core.config import load_config

RESULTS = Path("outputs/experiments/04_six_session/results/24387875c765b05d0a49d1c402d747c9ad7a263d43a3df273966aebd7cbc22d6.json")
SESSIONS = ["group_01/experiment_01/context_cam_01", "group_02/experiment_02/context_cam_01",
            "group_03/experiment_03/context_cam_01", "group_01/experiment_04/context_cam_01",
            "group_02/experiment_05/context_cam_01", "group_03/experiment_06/context_cam_01"]


def merge(new, old):
    if not (isinstance(new, dict) and isinstance(old, dict)):
        return old
    return {k: merge(new[k], old[k]) if k in old else new[k] for k in new} | {k: v for k, v in old.items() if k not in new}


cfg = merge(load_config("configs/deliverable3.json"), json.loads(RESULTS.read_text())["config"])
cfg["output_dir"] = "tmp/e2e6"
cfg["augmentation"] = {**load_config("configs/deliverable3.json")["augmentation"], "copies": int(sys.argv[1]) if len(sys.argv) > 1 else 1}
rp.features_dir = lambda cfg: Path("outputs/cache/e2e6/features/9a18772f0052")
rp.tracks_dir = lambda cfg: Path("outputs/cache/e2e6/tracks/300d595df270")
for s in SESSIONS:
    rp.cmd_extract(cfg, s, reuse=True)
print("augmented extraction done", flush=True)

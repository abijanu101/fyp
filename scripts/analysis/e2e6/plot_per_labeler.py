"""plot_results for the per-labeler six-session run (2026-11-16), from the run's caches.
  python scripts/analysis/e2e6/plot_per_labeler.py [key=value ...]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import scripts.train_models as tm
import scripts.plot_results as pr
from scripts.run_pipeline import parse_sets
from src.core.config import load_config

RESULTS = Path("outputs/experiments/04_six_session/results/24387875c765b05d0a49d1c402d747c9ad7a263d43a3df273966aebd7cbc22d6.json")


def merge(new, old):
    if not (isinstance(new, dict) and isinstance(old, dict)):
        return old
    return {k: merge(new[k], old[k]) if k in old else new[k] for k in new} | {k: v for k, v in old.items() if k not in new}


cfg = merge(load_config("configs/deliverable3.json"), json.loads(RESULTS.read_text())["config"])
for k, v in parse_sets(['output_dir="tmp/e2e6_pl"', 'engagement.labelers="per_labeler"', "labels.tau_att=3",
                        *sys.argv[1:]]).items():
    d = cfg
    *head, last = k.split(".")
    for h in head:
        d = d[h]
    d[last] = v
tm.features_dir = lambda cfg: Path("outputs/cache/e2e6/features/9a18772f0052")
tm.tracks_dir = lambda cfg: Path("outputs/cache/e2e6/tracks/300d595df270")
pr.main_per_labeler(cfg, tm.with_augmented(cfg, tm.load_table(cfg)))

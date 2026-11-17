"""tune_models (expert, per_labeler) smoke test on the six-session caches (2026-11-16)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import scripts.train_models as tm
import scripts.tuning.tune_models as tu

cfg = json.loads(Path(sorted(Path("outputs/experiments/05_per_labeler/results").glob("*.json"))[-1]).read_text())["config"]
Path("outputs/experiments/05_per_labeler/tuning/tune_cfg.json").write_text(json.dumps(cfg))
tm.features_dir = lambda cfg: Path("outputs/cache/e2e6/features/9a18772f0052")
tm.tracks_dir = lambda cfg: Path("outputs/cache/e2e6/tracks/300d595df270")
sys.argv = ["tune_models", "--config", "outputs/experiments/05_per_labeler/tuning/tune_cfg.json", "--component", "expert", "--trials", "3"]
tu.main()

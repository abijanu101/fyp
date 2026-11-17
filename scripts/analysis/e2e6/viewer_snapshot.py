"""Output-viewer snapshot with the per-labeler bundle on the six-session caches (2026-11-16)."""
import argparse, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import cv2
import scripts.output_viewer as ov

cfg = json.loads(Path(sorted(glob.glob("outputs/experiments/05_per_labeler/results/*.json"))[-1]).read_text())["config"]
cfg["output"]["prediction"] = sys.argv[1] if len(sys.argv) > 1 else "per_labeler"
ov.features_dir = lambda cfg: Path("outputs/cache/e2e6/features/9a18772f0052")
ov.tracks_dir = lambda cfg: Path("outputs/cache/e2e6/tracks/300d595df270")
args = argparse.Namespace(group="group_02", experiment="experiment_05", camera="context_cam_01", start=120.0, scale=0.6,
                          models=sorted(glob.glob("outputs/cache/e2e6_pl/models/*.joblib"))[-1])
v = ov.OutputViewer(args, cfg)
cv2.imwrite(f"outputs/experiments/05_per_labeler/viewer_{cfg['output']['prediction']}.png", v.render())
print("ok", v.score.min().round(3), v.score.max().round(3), "tau", v.tau)

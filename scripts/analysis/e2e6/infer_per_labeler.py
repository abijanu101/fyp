"""Inference on the unseen lecture with the per-labeler bundle, from the cached frame pass (2026-11-16).
  python scripts/analysis/e2e6/infer_per_labeler.py [per_labeler|majority]"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import joblib
import numpy as np

from src.core.pipeline import Pipeline
from src.core.store import load_track
from src.output import write_graph_json, write_graph_png

sys.argv += [] if len(sys.argv) > 1 else ["per_labeler"]
bundle = joblib.load(sorted(glob.glob("outputs/cache/e2e6_pl/models/*.joblib"))[-1])
cfg = json.loads(Path(sorted(glob.glob("outputs/experiments/05_per_labeler/results/*.json"))[-1]).read_text())["config"]
cfg["output"]["prediction"] = sys.argv[1]
pipe = Pipeline(cfg)
st = load_track(Path("outputs/cache/e2e6/tracks/300d595df270/group_01__experiment_01__context_cam_01.npz"))
records = pipe.predict(pipe.features(st), bundle)
out = Path(f"outputs/experiments/05_per_labeler/smoke_g1e1_{sys.argv[1]}"); out.mkdir(parents=True, exist_ok=True)
write_graph_json(records, out); write_graph_png(records, out, cfg["output"]["smooth_windows"])
r = np.array([x["rating"] for x in records])
print(sys.argv[1], len(records), "windows; final rating", r.min().round(2), "-", r.max().round(2), "mean", r.mean().round(2),
      "| engaged", sum(x["engaged"] == "Y" for x in records), "| labelers", list(records[0]["labeler_ratings"]))

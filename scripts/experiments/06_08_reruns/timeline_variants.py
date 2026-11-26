"""The variants whose held-out timelines go into 06's and 08's appendices; each must reproduce its
archived results hash. python scripts/experiments/06_08_reruns/timeline_variants.py  -> prints the check and writes timelines.sh"""
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.run_pipeline import parse_sets  # noqa: E402
from src.core.config import config_hash, load_config  # noqa: E402

RF = 'expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20}'
RF4 = 'expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}'
C06, C08 = "docs/experiments/06_full_pipeline_and_labels.json", "docs/experiments/08_pca.json"
NCTX = "engagement.late_context=false"
VARIANTS = [
    ("06_full_pipeline", "logistic_regression", C06, [], "c5341cb6385e"),
    ("06_full_pipeline", "random_forest", C06, ["expert.name=random_forest", RF4], "5df1987449dd"),
    ("06_full_pipeline", "knn", C06, ["expert.name=knn", 'expert.params={"n_neighbors": 25, "n_jobs": 4}'], "75a55ca34835"),
    ("08_pca", "logistic_regression", C08, [NCTX], "c724586518c0"),
    ("08_pca", "random_forest", C08, ["expert.name=random_forest", RF4, NCTX, "expert.pca=27"], "c36f89504032"),
    ("08_pca", "knn_k500", C08, ["expert.name=knn", 'expert.params={"n_neighbors": 500, "n_jobs": 4}', NCTX,
                                 "expert.pca=0.9"], "659b59ec05eb"),
    ("08_pca", "knn_k5", C08, ["expert.name=knn", 'expert.params={"n_neighbors": 5, "n_jobs": 4}', NCTX,
                               "expert.pca=20"], "8f5cfb454394"),
]

lines = ["#!/usr/bin/env bash", "# Held-out timelines, experiment 03 camera 01 of each group. Resumable.",
         "cd /e/Coding/Projects/fyp", "P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe"]
ok = True
for exp, name, cfg_path, sets, want in VARIANTS:
    h = config_hash(load_config(cfg_path, parse_sets(sets)))[:12]
    print(f"{exp:18} {name:20} {h} {'ok' if h == want else 'MISMATCH, want ' + want}")
    ok &= h == want
    out = f"outputs/experiments/{exp}/timelines/{name}"
    cmd = " ".join(["$P -m scripts.reporting.plot_results --config", cfg_path,
                    *(["--set", *map(shlex.quote, sets)] if sets else []),
                    "--sessions experiment_03/context_cam_01 --out", out])
    lines.append(f'[ -f {out}/overview.png ] || {{ echo "=== {exp} {name} $(date +%H:%M)"; {cmd}; }}')
lines.append('echo "=== done $(date +%H:%M)"')
Path("scripts/experiments/06_08_reruns/timelines.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
sys.exit(0 if ok else 1)

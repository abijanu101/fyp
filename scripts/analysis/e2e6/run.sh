#!/usr/bin/env bash
# Full-length end-to-end run: 6 sessions (one per activity) -> train (late, default) -> infer on an unseen session.
# Resumable: extract --reuse skips finished sessions.
set -euo pipefail
cd /e/Coding/Projects/fyp
PY=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
OUT=tmp/e2e6
SETS=(output_dir="\"$OUT\"" labels.tau_att=3 labels.tau_share=0.6)

for s in group_01/experiment_01 group_02/experiment_02 group_03/experiment_03 \
         group_01/experiment_04 group_02/experiment_05 group_03/experiment_06; do
  echo "=== extract $s  $(date +%T)"
  $PY -m scripts.run_pipeline extract --reuse --session "$s/context_cam_01" --set "${SETS[@]}"
done

echo "=== train  $(date +%T)"
$PY -m scripts.train_models --set "${SETS[@]}"

MODEL=$(ls -t $OUT/models/*.joblib | head -1)
echo "=== infer group_01/experiment_03 (unseen) with $MODEL  $(date +%T)"
$PY -m scripts.run_pipeline infer --set "${SETS[@]}" \
    --zip data/DIPSER/DIPSEER/group_01/experiment_03/context_cam_01.zip --models "$MODEL" --out $OUT/infer_g1e3
echo "=== done  $(date +%T)"

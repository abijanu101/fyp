#!/usr/bin/env bash
# Experiment 07: k-NN, k only, on experiment 06's input (all 95 columns, no PCA, context on).
# Resumable: an arm whose log already holds its config line is skipped.
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
OUT=outputs/experiments/07_knn/logs
CFG=docs/experiments/07_knn.json
run() {
  name=$1; shift
  if grep -q "^config" $OUT/$name.log 2>/dev/null; then echo "$name done already: $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')"; return; fi
  echo "=== $name ($(date +%H:%M))"
  $P -m scripts.train_models --config $CFG --set "$@" > $OUT/$name.log 2>&1 \
    || { echo "FAILED $name"; tail -5 $OUT/$name.log; return; }
  echo "$name $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')"
}
for k in 1 5 25 50 100 250 500 1000; do
  run knn_k$k expert.name=knn "expert.params={\"n_neighbors\": $k, \"n_jobs\": 4}"
done
echo "=== done ($(date +%H:%M))"

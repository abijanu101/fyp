#!/usr/bin/env bash
# Experiment 07, first arms: PCA inside the expert, with and without the class context (logistic regression).
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
OUT=outputs/experiments/07_knn/logs
CFG=docs/experiments/07_knn.json
run() {
  name=$1; shift
  echo "=== $name ($(date +%H:%M))"
  if [ $# -gt 0 ]; then set -- --set "$@"; fi
  $P -m scripts.train_models --config $CFG "$@" > $OUT/$name.log 2>&1 \
    || { echo "FAILED $name"; tail -5 $OUT/$name.log; return; }
  echo "$name $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')"
}
run pca20       expert.pca=20
run pca90       expert.pca=0.9
run pca95       expert.pca=0.95
run nctx        engagement.late_context=false
run nctx_pca5   engagement.late_context=false expert.pca=5
run nctx_pca10  engagement.late_context=false expert.pca=10
run nctx_pca90  engagement.late_context=false expert.pca=0.9
echo "=== done ($(date +%H:%M))"

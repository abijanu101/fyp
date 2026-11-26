#!/usr/bin/env bash
# Experiment 06 Part A: the single decoder with each classical expert model (logreg is run_all.sh's "single").
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
OUT=outputs/experiments/06_full_pipeline/logs
CFG=docs/experiments/06_full_pipeline_and_labels.json
run() {
  name=$1; shift
  echo "=== $name: $* ($(date +%H:%M))"
  $P -m scripts.train_models --config $CFG --set mixture.expert_scope=shared "$@" > $OUT/$name.log 2>&1 \
    || { echo "FAILED $name"; tail -5 $OUT/$name.log; return; }
  echo "$name $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')" | tee -a $OUT/hashes_models.txt
}
: > $OUT/hashes_models.txt
run random_forest expert.name=random_forest 'expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}'
run knn           expert.name=knn 'expert.params={"n_neighbors": 25, "n_jobs": 4}'
run svm           expert.name=svm 'expert.params={"max_train_rows": 4000}'
echo "=== done ($(date +%H:%M))"

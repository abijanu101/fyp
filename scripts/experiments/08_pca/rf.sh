#!/usr/bin/env bash
# Experiment 08, random forest: the same PCA and context arms as logistic regression (§1.2.1),
# with experiment 06's forest (200 trees, depth 6, leaf >= 20). Resumable: finished arms are skipped.
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
OUT=outputs/experiments/08_pca/logs
CFG=docs/experiments/08_pca.json
RF='expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}'
run() {
  name=$1; shift
  if grep -q "^config" $OUT/$name.log 2>/dev/null; then echo "$name done already: $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')"; return; fi
  echo "=== $name ($(date +%H:%M))"
  $P -m scripts.train_models --config $CFG --set expert.name=random_forest "$RF" "$@" > $OUT/$name.log 2>&1 \
    || { echo "FAILED $name"; tail -5 $OUT/$name.log; return; }
  echo "$name $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')"
}
run rf_base
run rf_pca5        expert.pca=5
run rf_pca10       expert.pca=10
run rf_pca20       expert.pca=20
run rf_pca90       expert.pca=0.9
run rf_pca95       expert.pca=0.95
run rf_nctx        engagement.late_context=false
run rf_nctx_pca5   engagement.late_context=false expert.pca=5
run rf_nctx_pca10  engagement.late_context=false expert.pca=10
run rf_nctx_pca90  engagement.late_context=false expert.pca=0.9
echo "=== done ($(date +%H:%M))"

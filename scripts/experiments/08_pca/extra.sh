#!/usr/bin/env bash
# Experiment 08: the arms that complete the column set for logistic regression and the random forest:
# PCA keeping every component (a rotation), and the no-context PCA 20 / 95% / all-27 arms.
# Resumable: finished arms are skipped.
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
OUT=outputs/experiments/08_pca/logs
CFG=docs/experiments/08_pca.json
RF='expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}'
run() {
  name=$1; shift
  if grep -q "^config" $OUT/$name.log 2>/dev/null; then echo "$name done already: $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')"; return; fi
  echo "=== $name ($(date +%H:%M))"
  $P -m scripts.train_models --config $CFG --set "$@" > $OUT/$name.log 2>&1 \
    || { echo "FAILED $name"; tail -5 $OUT/$name.log; return; }
  echo "$name $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')"
}
run lr_pca_all        expert.pca=95
run lr_nctx_pca_all   engagement.late_context=false expert.pca=27
run lr_nctx_pca20     engagement.late_context=false expert.pca=20
run lr_nctx_pca95     engagement.late_context=false expert.pca=0.95
run rf_pca_all        expert.name=random_forest "$RF" expert.pca=95
run rf_nctx_pca_all   expert.name=random_forest "$RF" engagement.late_context=false expert.pca=27
run rf_nctx_pca20     expert.name=random_forest "$RF" engagement.late_context=false expert.pca=20
run rf_nctx_pca95     expert.name=random_forest "$RF" engagement.late_context=false expert.pca=0.95
echo "=== done ($(date +%H:%M))"

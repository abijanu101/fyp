#!/usr/bin/env bash
# Experiment 08, k-NN with k = 5 (experiment 07's smallest k, where distances matter most): the 14 PCA and context arms. Resumable.
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
OUT=outputs/experiments/08_pca/logs
CFG=docs/experiments/08_pca.json
KNN='expert.params={"n_neighbors": 5, "n_jobs": 4}'
run() {
  name=$1; shift
  if grep -q "^config" $OUT/$name.log 2>/dev/null; then echo "$name done already: $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')"; return; fi
  echo "=== $name ($(date +%H:%M))"
  $P -m scripts.train_models --config $CFG --set expert.name=knn "$KNN" "$@" > $OUT/$name.log 2>&1 \
    || { echo "FAILED $name"; tail -5 $OUT/$name.log; return; }
  echo "$name $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')"
}
run knn5_base
run knn5_pca5        expert.pca=5
run knn5_pca10       expert.pca=10
run knn5_pca20       expert.pca=20
run knn5_pca90       expert.pca=0.9
run knn5_pca95       expert.pca=0.95
run knn5_nctx        engagement.late_context=false
run knn5_nctx_pca5   engagement.late_context=false expert.pca=5
run knn5_nctx_pca10  engagement.late_context=false expert.pca=10
run knn5_nctx_pca90  engagement.late_context=false expert.pca=0.9
run knn5_pca_all       expert.pca=95
run knn5_nctx_pca_all  engagement.late_context=false expert.pca=27
run knn5_nctx_pca20    engagement.late_context=false expert.pca=20
run knn5_nctx_pca95    engagement.late_context=false expert.pca=0.95
echo "=== done ($(date +%H:%M))"

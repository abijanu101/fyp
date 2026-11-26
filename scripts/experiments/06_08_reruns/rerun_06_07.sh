#!/usr/bin/env bash
# Re-run every experiment 06/07 arm so its results carry training-entry baselines and the
# prediction-spread metrics (18 Nov). Same configs, so the same hashes: results are overwritten.
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
OUT=outputs/experiments/06_full_pipeline/logs
CFG=docs/experiments/06_full_pipeline_and_labels.json
run() {
  name=$1; want=$2; shift 2
  echo "=== $name ($(date +%H:%M))"
  if [ $# -gt 0 ]; then set -- --set "$@"; fi
  $P -m scripts.train_models --config $CFG "$@" > $OUT/$name.log 2>&1 \
    || { echo "FAILED $name"; tail -5 $OUT/$name.log; return; }
  got=$(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')
  [ "$got" = "$want" ] && echo "$name $got ok" || echo "$name $got MISMATCH (expected $want)"
}
run single   c5341cb6385e
run c0       f6dcb84aaa6b augmentation.copies=0
run pooled   fdb140e9477e engagement.labelers=pooled
run moe      38c985c0f859 mixture.expert_scope=per_experiment
run ramp3    9f5c78e85899 labels.interp.mode=ramp labels.interp.ramp_s=3
run ramp7    5c38aa124d4e labels.interp.mode=ramp labels.interp.ramp_s=7
run gauss1   83551175c8ce labels.interp.mode=gauss labels.interp.sigma_s=1
run gauss2   59fe8967ec34 labels.interp.mode=gauss labels.interp.sigma_s=2
run vanilla  07cc8c676819 labels.train_target=train_vanilla
run step8s   0159e99cadc2 window.length_s=8
run rf       5df1987449dd expert.name=random_forest 'expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}'
run knn      75a55ca34835 expert.name=knn 'expert.params={"n_neighbors": 25, "n_jobs": 4}'
echo "=== done ($(date +%H:%M))"

#!/usr/bin/env bash
# The remaining experiment-06 arms, one after another (SVM, interpolation, 8 s windows).
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
OUT=outputs/experiments/06_full_pipeline/logs
CFG=docs/experiments/06_full_pipeline_and_labels.json
run() {
  name=$1; shift
  echo "=== $name: $* ($(date +%H:%M))"
  $P -m scripts.train_models --config $CFG --set "$@" > $OUT/$name.log 2>&1 \
    || { echo "FAILED $name"; tail -5 $OUT/$name.log; return; }
  echo "$name $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')"
}
run svm      expert.name=svm 'expert.params={"max_train_rows": 4000}'
run ramp3    labels.interp.mode=ramp labels.interp.ramp_s=3
run ramp7    labels.interp.mode=ramp labels.interp.ramp_s=7
run gauss1   labels.interp.mode=gauss labels.interp.sigma_s=1
run gauss2   labels.interp.mode=gauss labels.interp.sigma_s=2
run vanilla  labels.train_target=train_vanilla
run step8s   window.length_s=8
echo "=== done ($(date +%H:%M))"

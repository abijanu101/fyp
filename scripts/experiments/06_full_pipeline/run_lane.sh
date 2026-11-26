#!/usr/bin/env bash
# One lane of the remaining experiment-06 retrains: bash run_lane.sh A|B
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
OUT=outputs/experiments/06_full_pipeline/logs
CFG=docs/experiments/06_full_pipeline_and_labels.json
run() {
  name=$1; shift
  echo "=== $name: $* ($(date +%H:%M))"
  $P -m scripts.train_models --config $CFG --set mixture.expert_scope=shared "$@" > $OUT/$name.log 2>&1 \
    || { echo "FAILED $name"; tail -5 $OUT/$name.log; return; }
  echo "$name $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')" | tee -a $OUT/hashes.txt
}
if [ "$1" = A ]; then
  run ramp3   labels.interp.mode=ramp labels.interp.ramp_s=3
  run ramp7   labels.interp.mode=ramp labels.interp.ramp_s=7
  run gauss1  labels.interp.mode=gauss labels.interp.sigma_s=1
else
  run gauss2  labels.interp.mode=gauss labels.interp.sigma_s=2
  run vanilla labels.train_target=train_vanilla
  echo "=== extract 8 s windows ($(date +%H:%M))"
  $P -m scripts.run_pipeline extract --reuse --config $CFG --set window.length_s=8 > $OUT/extract_8s.log 2>&1 \
    || { echo "FAILED extract_8s"; tail -5 $OUT/extract_8s.log; }
  run step8s  window.length_s=8
fi
echo "=== lane $1 done ($(date +%H:%M))"

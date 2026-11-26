#!/usr/bin/env bash
# Experiment 06 part A additions: retrains on the copies=1 cache (train scores, pooled arm,
# interpolation in training, 8 s windows). One log per run; hashes collected in hashes.txt.
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
OUT=outputs/experiments/06_full_pipeline/logs
BASE="augmentation.copies=1 labels.tau_att=3"
: > $OUT/hashes.txt
run() {
  name=$1; shift
  echo "=== $name: $* ($(date +%H:%M))"
  $P -m scripts.train_models --set "$@" > $OUT/$name.log 2>&1 || { echo "FAILED $name"; tail -5 $OUT/$name.log; return; }
  echo "$name $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')" | tee -a $OUT/hashes.txt
}
run moe            $BASE
run single         $BASE mixture.expert_scope=shared
run single_c0      augmentation.copies=0 labels.tau_att=3 mixture.expert_scope=shared
run pooled         $BASE mixture.expert_scope=shared engagement.labelers=pooled
run ramp3          $BASE mixture.expert_scope=shared labels.interp.mode=ramp labels.interp.ramp_s=3
run ramp7          $BASE mixture.expert_scope=shared labels.interp.mode=ramp labels.interp.ramp_s=7
run gauss1         $BASE mixture.expert_scope=shared labels.interp.mode=gauss labels.interp.sigma_s=1
run gauss2         $BASE mixture.expert_scope=shared labels.interp.mode=gauss labels.interp.sigma_s=2
run vanilla        $BASE mixture.expert_scope=shared labels.train_target=train_vanilla
echo "=== extract 8 s windows ($(date +%H:%M))"
$P -m scripts.run_pipeline extract --reuse --set window.length_s=8 augmentation.copies=1 > $OUT/extract_8s.log 2>&1 || { echo "FAILED extract_8s"; tail -5 $OUT/extract_8s.log; }
run step8s         $BASE mixture.expert_scope=shared window.length_s=8
echo "=== done ($(date +%H:%M))"

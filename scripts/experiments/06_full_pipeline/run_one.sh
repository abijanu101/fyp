#!/usr/bin/env bash
# One experiment-06 retrain: bash run_one.sh <name> <--set args...>; or "wait <name>" to record a run
# already in progress; "step8s" extracts the 8 s features first.
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
OUT=outputs/experiments/06_full_pipeline/logs
CFG=docs/experiments/06_full_pipeline_and_labels.json
record() { echo "$1 $(grep -m1 '^config' $OUT/$1.log | awk '{print $2}')" >> $OUT/hashes.txt;
           grep -E "configured/(classified|train)/own_labeler|Traceback|Error" $OUT/$1.log; }
if [ "$1" = wait ]; then
  until grep -q "^config\|Traceback" $OUT/$2.log; do sleep 20; done; record $2; exit
fi
name=$1; shift
if [ "$name" = step8s ]; then
  $P -m scripts.run_pipeline extract --reuse --config $CFG --set window.length_s=8 > $OUT/extract_8s.log 2>&1 \
    || { echo "FAILED extract_8s"; tail -5 $OUT/extract_8s.log; exit 1; }
fi
$P -m scripts.train_models --config $CFG --set mixture.expert_scope=shared "$@" > $OUT/$name.log 2>&1 \
  || { echo "FAILED $name"; tail -5 $OUT/$name.log; exit 1; }
record $name

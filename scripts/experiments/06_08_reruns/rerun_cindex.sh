#!/usr/bin/env bash
# Retrain every arm cited in experiments 06-08 whose results predate the C-index (18 Nov 2026), so their
# tables get the new column and their per-entry predictions are saved (outputs/predictions/).
# Same configs, so the same hashes: results files are overwritten in place. Resumable: an arm whose
# results file already has the C-index is skipped. Log: tmp/rerun_cindex.log
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
LOG=tmp/rerun_cindex
mkdir -p $LOG
has_cindex() {   # $1 = hash prefix
  $P -c "import json,glob,sys; f=glob.glob('outputs/results/$1*.json'); sys.exit(0 if f and 'c_index' in json.load(open(f[0]))['evaluation']['experts/configured/classified/own_labeler']['summary'] else 1)" 2>/dev/null
}
run() {          # name, expected hash, config, --set args...
  name=$1; want=$2; cfg=$3; shift 3
  if has_cindex $want; then echo "$name $want done already"; return; fi
  echo "=== $name ($(date +%H:%M))"
  if [ $# -gt 0 ]; then set -- --set "$@"; fi
  $P -m scripts.train_models --config $cfg "$@" > $LOG/$name.log 2>&1 || { echo "FAILED $name"; tail -5 $LOG/$name.log; return; }
  got=$(grep -m1 '^config' $LOG/$name.log | awk '{print $2}')
  [ "$got" = "$want" ] && echo "$name $got ok" || echo "$name $got MISMATCH (expected $want)"
}
C6=docs/experiments/06_full_pipeline_and_labels.json
C7=docs/experiments/07_knn.json
C8=docs/experiments/08_pca.json
RF='expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}'
KNN='expert.params={"n_neighbors": 500, "n_jobs": 4}'
NC=engagement.late_context=false

# 06 (logistic regression unless stated)
run e06_single  c5341cb6385e $C6
run e06_c0      f6dcb84aaa6b $C6 augmentation.copies=0
run e06_pooled  fdb140e9477e $C6 engagement.labelers=pooled
run e06_ramp3   9f5c78e85899 $C6 labels.interp.mode=ramp labels.interp.ramp_s=3
run e06_ramp7   5c38aa124d4e $C6 labels.interp.mode=ramp labels.interp.ramp_s=7
run e06_gauss1  83551175c8ce $C6 labels.interp.mode=gauss labels.interp.sigma_s=1
run e06_gauss2  59fe8967ec34 $C6 labels.interp.mode=gauss labels.interp.sigma_s=2
run e06_vanilla 07cc8c676819 $C6 labels.train_target=train_vanilla
run e06_step8s  0159e99cadc2 $C6 window.length_s=8
run e06_rf      5df1987449dd $C6 expert.name=random_forest 'expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}'
run e06_knn     75a55ca34835 $C6 expert.name=knn 'expert.params={"n_neighbors": 25, "n_jobs": 4}'

# 07 (k-NN, k only)
for kh in 5:e79c4c117090 25:75a782eb8951 50:efa51da0cd61 100:2fd336ace408 250:dbc58c4983ba 500:2fcb719b3a54 1000:386791e78fd1; do
  k=${kh%%:*}; h=${kh##*:}
  run e07_k$k $h $C7 expert.name=knn "expert.params={\"n_neighbors\": $k, \"n_jobs\": 4}"
done

# 08: logistic regression and the forest, every PCA / context arm
arms=( "base|" "pca_all|expert.pca=95" "pca5|expert.pca=5" "pca10|expert.pca=10" "pca20|expert.pca=20"
       "pca90|expert.pca=0.9" "pca95|expert.pca=0.95" "nctx|$NC" "nctx_pca_all|$NC expert.pca=27"
       "nctx_pca5|$NC expert.pca=5" "nctx_pca10|$NC expert.pca=10" "nctx_pca20|$NC expert.pca=20"
       "nctx_pca90|$NC expert.pca=0.9" "nctx_pca95|$NC expert.pca=0.95" )
declare -A LR=( [base]=b3fbc8430d44 [pca_all]=a2096ec10bef [pca5]=5580383edd8d [pca10]=9990d1832233
  [pca20]=69cc580dcd9a [pca90]=617bf1d8b13b [pca95]=9cd26eabef2a [nctx]=c724586518c0 [nctx_pca_all]=5d86ad5c10cf
  [nctx_pca5]=049b908f551c [nctx_pca10]=60fc84b50016 [nctx_pca20]=b4476ab7dc49 [nctx_pca90]=84227cefa9b3
  [nctx_pca95]=41511c48f91c )
for a in "${arms[@]}"; do
  n=${a%%|*}; s=${a#*|}
  run e08_lr_$n ${LR[$n]} $C8 $s
done
for a in "${arms[@]}"; do
  n=${a%%|*}; s=${a#*|}
  h=$(grep -h "^rf_$n " outputs/experiments/08_pca/logs/rf.log outputs/experiments/08_pca/logs/extra.log 2>/dev/null | awk '{print $2}' | head -1)
  [ -z "$h" ] && { echo "e08_rf_$n: no hash recorded yet, skipped"; continue; }
  run e08_rf_$n $h $C8 expert.name=random_forest "$RF" $s
done
for a in "${arms[@]}"; do
  n=${a%%|*}; s=${a#*|}
  h=$(grep -h "^knn_$n " outputs/experiments/08_pca/logs/knn.log 2>/dev/null | awk '{print $2}' | head -1)
  [ -z "$h" ] && { echo "e08_knn_$n: no hash recorded yet, skipped"; continue; }
  run e08_knn_$n $h $C8 expert.name=knn "$KNN" $s
done
echo "=== done ($(date +%H:%M))"

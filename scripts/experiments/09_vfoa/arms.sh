#!/usr/bin/env bash
# Experiment 09 §5-§5.1: the three comparisons, with experiment 08's four models (logistic regression,
# random forest, k-NN at k = 500 and k = 5). Resumable: a finished arm (its log has the "config <hash>"
# line) is skipped, so stop and rerun freely. From the repository root:
#
#   bash scripts/experiments/09_vfoa/arms.sh c1        # comparison 1, VFoA: 4 inputs x context off/on x 4 models (32 runs)
#   bash scripts/experiments/09_vfoa/arms.sh c2        # comparison 2, scale shrinking, on inputs 1 and 2 (16 runs)
#   bash scripts/experiments/09_vfoa/arms.sh c3        # comparison 3, filtered training data, p = 0.26 and 0.5, inputs 1 and 2 (32 runs)
#   bash scripts/experiments/09_vfoa/arms.sh all       # all three, in that order
#
# k-NN regression (09 section 5.2), arm names knnr5_ (k = 5), knnrk<k>_ (any other k; knnr_ = k = 500),
# then the k-NN classifier at k = 50 in the same arms, arm names knnck50_:
#   bash scripts/experiments/09_vfoa/arms.sh knnreg            # regression at every k in KS, then the classifier at every k in
#                                            # KCLS, each in every arm (inputs 1-4, context off/on, all three
#                                            # comparisons: 32 arms per k): 2 + 1 k = 96 runs, one k at a time
#   bash scripts/experiments/09_vfoa/arms.sh knnreg 5 50     # regression at only these k (then the classifier as above)
#
# Logs: outputs/experiments/09_vfoa/logs/<arm>.log; arm -> hash lines appended to logs/arms.log (the original run's index: outputs/experiments/09_vfoa/arms.log)
set -u
P=                                         # Git Bash mounts C: at /c, WSL bash (PowerShell's `bash`) at /mnt/c
for p in /c/Users/abi/miniconda3/envs/fyp-local/python.exe /mnt/c/Users/abi/miniconda3/envs/fyp-local/python.exe; do
  [ -x "$p" ] && { P=$p; break; }
done
[ -n "$P" ] || { echo "fyp-local python.exe not found under /c or /mnt/c"; exit 1; }
CFG=docs/experiments/09_vfoa.json          # 1-5 scale, class context off, 4 augmented copies
OUT=outputs/experiments/09_vfoa/logs
PHASE=${1:-all}
shift $(( $# > 0 ? 1 : 0 ))
KS="5 50"      # regression; 09 section 5.2 (k = 15 and 500 dropped for compute time)
KCLS="50"      # the classifier at the same k, to compare the two at k = 50
[ $# -gt 0 ] && KS="$*"

RF='expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}'
K500='expert.params={"n_neighbors": 500, "n_jobs": 4}'
K5='expert.params={"n_neighbors": 5, "n_jobs": 4}'
KR=expert.name=knn_regression
NO_HP='engagement.columns={"exclude": ["trk.hp.", "vec.hp.", "rel."]}'
HP_ONLY='engagement.columns={"include": ["trk.hp.", "vec.hp.", "rel."]}'
CTX='engagement.use_class_context=true'
REL='engagement.class_relative=true'
LMH='labels.collapse=[[1,2],[3],[4,5]]'
KEEP26='labels.train_keep={"2": 0.26, "3": 0.26}'
KEEP50='labels.train_keep={"2": 0.5, "3": 0.5}'

run() {
  name=$1; shift
  if grep -q "^config" $OUT/$name.log 2>/dev/null; then echo "$name done already: $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')"; return; fi
  echo "=== $name ($(date +%H:%M))"
  $P -m scripts.train_models --config $CFG --set "$@" > $OUT/$name.log 2>&1 \
    || { echo "FAILED $name"; tail -5 $OUT/$name.log; return; }
  echo "$name $(grep -m1 '^config' $OUT/$name.log | awk '{print $2}')"
}

model_args() {            # the model's --set arguments, into the array M
  case $1 in
    lr)    M=() ;;
    rf)    M=(expert.name=random_forest "$RF") ;;
    knn)   M=(expert.name=knn "$K500") ;;
    knn5)  M=(expert.name=knn "$K5") ;;
    knnr)  M=($KR "$K500") ;;
    knnr5) M=($KR "$K5") ;;
    knnrk*) M=($KR "expert.params={\"n_neighbors\": ${1#knnrk}, \"n_jobs\": 4}") ;;
    knnck*) M=(expert.name=knn "expert.params={\"n_neighbors\": ${1#knnck}, \"n_jobs\": 4}") ;;
  esac
}

# inputs 1 and 2, both contexts, one model, plus any extra settings ($3...), named <model>_<tag>_...
inputs_1_2() {
  m=$1; tag=$2; shift 2
  model_args $m
  run ${m}_${tag}_1_base      "${M[@]}" "$NO_HP" "$@"
  run ${m}_${tag}_1_base_ctx  "${M[@]}" "$NO_HP" "$CTX" "$@"
  run ${m}_${tag}_2_hp        "${M[@]}" "$@"
  run ${m}_${tag}_2_hp_ctx    "${M[@]}" "$CTX" "$@"
}

# all four inputs, both contexts (inputs 1 and 2, then 3 and 4)
inputs_all() {
  m=$1; tag=$2; shift 2
  inputs_1_2 $m $tag "$@"
  model_args $m
  run ${m}_${tag}_3_hp_only      "${M[@]}" "$HP_ONLY" "$@"
  run ${m}_${tag}_3_hp_only_ctx  "${M[@]}" "$HP_ONLY" "$CTX" "$@"
  run ${m}_${tag}_4_relative     "${M[@]}" "$REL" "$@"
  run ${m}_${tag}_4_relative_ctx "${M[@]}" "$REL" "$CTX" "$@"
}
knnr_name() { case $1 in 500) echo knnr ;; 5) echo knnr5 ;; *) echo knnrk$1 ;; esac; }

c1() { for m in "$@"; do inputs_all $m c1; done; }
c2() { for m in "$@"; do inputs_1_2 $m c2 "$LMH"; done; }
c3() { for m in "$@"; do inputs_1_2 $m c3p26 "$KEEP26"; inputs_1_2 $m c3p50 "$KEEP50"; done; }

{
  if [ $PHASE = c1 ] || [ $PHASE = all ]; then c1 lr rf knn knn5; fi
  if [ $PHASE = c2 ] || [ $PHASE = all ]; then
    if [ ! -f outputs/labels/agreement_lmh.json ]; then   # the human ceiling on the three levels (~10 min)
      echo "=== human ceiling, LOW / MID / HIGH ($(date +%H:%M))"
      $P -m scripts.labels.label_agreement ceiling --config $CFG --set "$LMH" --model-entries \
        --out outputs/labels/agreement_lmh.json > $OUT/ceiling_lmh.log 2>&1 || echo "FAILED ceiling ($OUT/ceiling_lmh.log)"
    fi
    c2 lr rf knn knn5
  fi
  if [ $PHASE = c3 ] || [ $PHASE = all ]; then c3 lr rf knn knn5; fi
  if [ $PHASE = knnreg ]; then
    for k in $KS; do m=$(knnr_name $k)
      inputs_all $m c1; inputs_all $m c2 "$LMH"; inputs_all $m c3p26 "$KEEP26"; inputs_all $m c3p50 "$KEEP50"
    done
    for k in $KCLS; do m=knnck$k
      inputs_all $m c1; inputs_all $m c2 "$LMH"; inputs_all $m c3p26 "$KEEP26"; inputs_all $m c3p50 "$KEEP50"
    done
  fi
} 2>&1 | tee -a $OUT/arms.log

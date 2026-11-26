#!/usr/bin/env bash
# Experiment 10, steps 4-7 of scripts/clustering/README.md, on experiment 09's caches, with the choices
# between steps read from the outputs (scripts/experiments/10_clustering/pick.py) instead of by hand:
#   channels   = signal_screen_channels.csv rows with pass_sync true
#   tau-desc   = desc:subset, agglomerative, usable: highest temporal ARI
#   tau-sync   = sync:*, agglomerative, usable: highest mean temporal ARI over the channels
#   best arm   = what the first ablation run chose for sync+best, per split: each split's frozen runs
#                use its own choice (BEST=<arm> bash scripts/experiments/10_clustering/run.sh forces one arm for both)
# Resumable: each finished step leaves outputs/experiments/10_clustering/logs/done/<step>, and is skipped on a rerun (delete the
# marker to redo one). From the repository root:  bash scripts/experiments/10_clustering/run.sh
# Logs: outputs/experiments/10_clustering/logs/<step>.log (also shown live). Outputs: outputs/experiments/10_clustering/
set -u
P=                                         # Git Bash mounts C: at /c, WSL bash (PowerShell's `bash`) at /mnt/c
for p in /c/Users/abi/miniconda3/envs/fyp-local/python.exe /mnt/c/Users/abi/miniconda3/envs/fyp-local/python.exe; do
  [ -x "$p" ] && { P=$p; break; }
done
[ -n "$P" ] || { echo "fyp-local python.exe not found under /c or /mnt/c"; exit 1; }
W=${W:-$(( $(nproc) - 2 ))}               # workers: CPU cores minus 2
OUT=outputs/experiments/10_clustering/logs
mkdir -p $OUT/done

pick() { $P scripts/experiments/10_clustering/pick.py "$@" | tr -d '\r'; }

step() {                                   # step <name> <args to python -m ...>
  name=$1; shift
  if [ -f $OUT/done/$name ]; then echo "$name done already"; return 0; fi
  echo "=== $name ($(date '+%d %b %H:%M')): python -m $*"
  $P -m "$@" < /dev/null 2>&1 | tee $OUT/$name.log   # no stdin: WSL's interop would let python eat the loop's input
  if [ ${PIPESTATUS[0]} -ne 0 ]; then echo "FAILED $name (see $OUT/$name.log); fix and rerun"; exit 1; fi
  date '+%d %b %H:%M' > $OUT/done/$name
}

# 4. what to cluster on (no labels)
step signal_screen scripts.clustering.signal_screen --workers $W
CH=$(pick channels) || exit 1
echo "channels (pass_sync): $CH"
step event_check scripts.clustering.event_check --channels $CH
step descriptor_subset scripts.clustering.descriptor_subset
step stability scripts.clustering.stability --workers $W --channels $CH
TAUS=$(pick taus) || exit 1
set -- $TAUS; TD=$1; TS=$2
echo "tau-desc $TD, tau-sync $TS"

# 5. candidate features (resumes per session itself)
step candidates scripts.clustering.candidates --workers $W --channels $CH --tau-desc $TD --tau-sync $TS

# 6. do they predict attention?
step feature_screen scripts.clustering.feature_screen
step importance scripts.clustering.importance --gate
step ablation scripts.clustering.ablation
if [ -n "${BEST:-}" ]; then PICKS="activity $BEST
group $BEST"; else PICKS=$(pick best) || exit 1; fi
[ -n "$PICKS" ] || { echo "no best arm picked"; exit 1; }
# each split frozen on the arm its own first run chose (they can differ), one ablation run per split
while read -r split arm; do
  echo "best arm, $split split: $arm"
  step ablation_best_$split scripts.clustering.ablation --split $split --best $arm --arms null normalised sync sync+best
  step ablation_nowin_$split scripts.clustering.ablation --split $split --best $arm --arms null normalised sync sync+best --no-window
done <<< "$PICKS"

# 7. one file with every result
step report scripts.clustering.report
echo "all done: outputs/experiments/10_clustering/clustering_report.json"

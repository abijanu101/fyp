#!/usr/bin/env bash
# Regenerate every results table in experiment 06 Part 2 (§2.3.1, 2.3.2, 2.3.4, 2.4) with results_table.
cd /e/Coding/Projects/fyp
P=/c/Users/abi/miniconda3/envs/fyp-local/python.exe
T=tmp/exp06_tables; mkdir -p $T   # the tables, to paste into 06 (frozen): the scripts that pasted them were single-use
M=("Logistic regression=c5341cb6385e" "Random forest=5df1987449dd" "k-NN=75a55ca34835" "Pooled labelers=fdb140e9477e" "No augmented copy=f6dcb84aaa6b")
$P -m scripts.reporting.results_table "${M[@]}" > $T/models_experts.md
$P -m scripts.reporting.results_table "${M[@]}" --track self > $T/models_self.md
$P -m scripts.reporting.results_table "step (default)=c5341cb6385e" "ramp 3 s=9f5c78e85899" "ramp 7 s=5c38aa124d4e" "Gaussian 1 s=83551175c8ce" "Gaussian 2 s=59fe8967ec34" "no interpolation=07cc8c676819" "step, 8 s windows=0159e99cadc2" > $T/targets.md
$P -m scripts.reporting.results_table "Logistic regression=c5341cb6385e" --by-labeler > $T/by_labeler.md

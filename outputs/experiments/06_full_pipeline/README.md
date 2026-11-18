# Experiment 06: runs (frozen 18 November 2026)

Results files are in `outputs/results/<hash>.json`; the training logs of the final runs (the 18 November
re-run, with training-set, prediction-spread and per-labeler scores) are in `logs/`. `label_agreement.log`
is Part 1's analysis; its tables are in `outputs/labels/agreement.json`. Each arm is
`python -m scripts.train_models --config docs/experiments/06_full_pipeline_and_labels.json` plus the
setting shown (docs/experiments/06_full_pipeline_and_labels.md §2.2).

| Arm | Setting | Results | Log |
|---|---|---|---|
| per-labeler, logistic regression (the config) | – | `c5341cb6385e` | `single.log` |
| no augmented copy | `augmentation.copies=0` | `f6dcb84aaa6b` | `c0.log` |
| pooled labelers | `engagement.labelers=pooled` | `fdb140e9477e` | `pooled.log` |
| random forest | `expert.name=random_forest` + params (§2.2) | `5df1987449dd` | `rf.log` |
| k-NN | `expert.name=knn` + params (§2.2) | `75a55ca34835` | `knn.log` |
| ramp 3 s / 7 s | `labels.interp.mode=ramp labels.interp.ramp_s=3` / `7` | `9f5c78e85899` / `5c38aa124d4e` | `ramp3.log` / `ramp7.log` |
| Gaussian 1 s / 2 s | `labels.interp.mode=gauss labels.interp.sigma_s=1` / `2` | `83551175c8ce` / `59fe8967ec34` | `gauss1.log` / `gauss2.log` |
| no interpolation | `labels.train_target=train_vanilla` | `07cc8c676819` | `vanilla.log` |
| step, 8 s windows | `window.length_s=8` (8 s features: `extract_8s.log`) | `0159e99cadc2` | `step8s.log` |
| SVM (dropped, not reported) | `expert.name=svm 'expert.params={"max_train_rows": 4000}'` | `13785d8fe677` | – |
| mixture of experts (moved to experiment 09) | `mixture.expert_scope=per_experiment` | `38c985c0f859` | `moe.log` |

`rerun_index.log` lists every re-run arm with its hash, each checked against the hash above.

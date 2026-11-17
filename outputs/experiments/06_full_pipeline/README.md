# Experiment 06: runs

Results files are in `outputs/results/<hash>.json`; logs of the training runs are in `logs/`
(`label_agreement.log` is Part 1's analysis, whose tables are in `outputs/labels/agreement.json`).
Each arm is `python -m scripts.train_models --config docs/experiments/06_full_pipeline_and_labels.json` plus the
setting shown (docs/experiments/06_full_pipeline_and_labels.md §2.2).

| Arm | Setting | Results |
|---|---|---|
| per-labeler, logistic regression (the config) | – | `c5341cb6385e` |
| no augmented copy | `augmentation.copies=0` | `f6dcb84aaa6b` |
| pooled labelers | `engagement.labelers=pooled` | `fdb140e9477e` |
| random forest | `expert.name=random_forest` + params | `5df1987449dd` |
| k-NN | `expert.name=knn` + params | `75a55ca34835` |
| mixture of experts (to be redone separately) | `mixture.expert_scope=per_experiment` | `38c985c0f859` |

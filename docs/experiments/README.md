# Experiments

Records of what we did to the data and the components, why, and what we found. Where
`docs/guide/` explains how the code works, these documents explain the decisions behind it and
the evidence for them. They are the raw material for the write-up.

**The story so far.**
- 01–03 build and tune the components.
- 04 runs the whole pipeline on six sessions: at chance, but the features carry some signal.
- 05 moves training and evaluation to each labeler's own entries.
- 06 runs it at full scale and measures the labels. The labels allow little (a human ceiling barely
  above chance), and the model underfits with every classical decoder. That points at the
  features.
- 07 tunes k for k-NN, the most spread-out decoder in 06, on 06's input.
- 08 tries dimensionality reduction (PCA, and dropping the class context) with logistic regression,
  the random forest and 07's k-NN. 09 and 10 build
  on them: activity experts, then class-relative (clustering) features.

| # | Document | Status |
|---|---|---|
| 1 | [DIPSER: dataset work and manual annotation](01_dipser.md) | current; the label analysis is in experiment 06 Part 1 |
| 2 | [BoxMOT: detection and tracking](02_boxmot.md) | done: D3 sets `detector.conf` 0.1098 (augmented); T2 picks BoT-SORT + ReID + CMC (0.930 IDF1 on augmented data, against 0.914 without ReID and 0.877 for StrongSORT) |
| 3 | [SCB behaviour detection](03_scb.md) | current; feature-family ablation pending |
| 4 | [The first full-length run: six sessions, end to end](04_six_session_run.md) | done: the pipeline works at full length; engagement at chance, but the features carry consistent within-session signal the models do not use; directions for improvement |
| 5 | [Per-labeler training: first runs on the six sessions](05_per_labeler.md) | short: the per-labeler path runs; at chance by construction; one expert made 59% of the entries; constant 3 has the lowest MAE; a six-session augmentation gain that 06 does not confirm |
| 6 | [The labels and the full pipeline](06_full_pipeline_and_labels.md) ([config](06_full_pipeline_and_labels.json)) | **frozen**: the labels set a low ceiling (Krippendorff's α −0.06 to +0.04 on every unit and interpolation; the other experts predict an entry at AUC 0.564 and no human combination beats constant 3 on MAE); the model predicts 3 for 96% of entries (AUC 0.527, MAE 0.674 against constant 3's 0.619) and underfits, with logistic regression, a random forest or k-NN (SVM dropped as too slow). Includes the label analysis promised in row 1, perspectivism, and interpolation |
| 7 | [k-NN tuning: which k?](07_knn.md) ([config](07_knn.json)) | done: a larger k generalises better (test Spearman 0.017 at k 1, 0.038 at k 5, 0.083 at k 500); k = 500 chosen (k 1,000 degenerates for the smallest labeler); still below logistic regression; the self-report at chance for every k |
| 8 | [PCA and dimensionality reduction](08_pca.md) ([config](08_pca.json)) | started: the vector (95 inputs, every column against every labeler: the features look fine for what the labels allow, optical flow tracking the experts at ρ ≈ 0.08). PCA does not help logistic regression or the random forest (fewer components lower training and test ranking alike); the 27 per-student columns alone do as well as all 95. k-NN (k = 500) running |
| 9 | Mixture of experts against a single decoder | planned, on 08's features; a first run on the 06 cache tied (expert-entry AUC 0.526 / 0.527, results `38c985c0f859` / `c5341cb6385e`) |
| 10 | Dynamic-clustering synchrony features | planned (guide 14) |

Status as of 18 November 2026. Augmentation is covered in 06 §2.3 (one copy changes nothing; heavier augmentation not pursued). From experiment 06 on, each document has its frozen config next to it
(`NN_name.json`): run its commands with `--config docs/experiments/NN_name.json`.

Conventions: every number states the configuration or command that produced it; anything not
yet verified on data says so.

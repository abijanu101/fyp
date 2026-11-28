# Experiments

Records of what we did to the data and the components, why, and what we found. Where
`docs/guide/` explains how the code works, these documents explain the decisions behind it and
the evidence for them. They are the raw material for the write-up.

**The story so far.**
- 01–03 build and tune the components: the dataset, the person detector, the tracker. (The SCB
  behaviour detector was integrated and checked, not experimented on; that record is in guide
  Chapter 5.)
- 04 runs the whole pipeline on six sessions: at chance, but the features carry some signal.
- 05 moves training and evaluation to each labeler's own entries.
- 06 runs it at full scale and measures the labels. The labels allow little (a human ceiling barely
  above chance), and the model underfits with every classical decoder. That points at the
  features.
- 07 tunes k for k-NN, the most spread-out decoder in 06, on 06's input.
- 08 tries dimensionality reduction (PCA, and dropping the class context) with logistic regression,
  the random forest and 07's k-NN. Nothing moves the result by more than ±0.02: the limit is the
  signal in the input, not how it is transformed.
- 09 therefore adds new information: where each student's head points (visual focus of attention,
  VFoA), estimated from the classroom cameras. It also tries two strategies against the class
  imbalance that keeps every model near the middle of the scale: scale shrinking (1–2 LOW, 3 MID,
  4–5 HIGH) and filtered training data (each level's entries kept with a set probability).
- 10 describes each student against the class (synchrony and dynamic clustering). The groups are
  real, but they move the ranking by at most 0.01; after the run, the features turn out to carry a
  little information within a session and none between students.
- 11 gathers the class imbalance: it is not what limits the models, but it hid that their output
  never used the scale; cut points per labeler turn the same scores into 1–5 ratings with real
  agreement.

| # | Document | Status |
|---|---|---|
| 1 | [DIPSER: dataset work and manual annotation](01_dipser.md) | current; the label analysis is in experiment 06 Part 1 |
| 2 | [Person detection](02_detection.md) | done: D2 sets `detector.conf` 0.1098, the highest threshold that keeps every student within 10 recall points of the floor on augmented footage (0.22 lost 22); D3 (the pose model as detector, 1-point criterion, for experiment 09) sets 0.0535, at the floor; BoT-SORT's own `track_low_thresh` (0.1745) is the effective floor in the tracked pipeline |
| 3 | [Multi-object tracking](03_tracking.md) | done: T2 picks BoT-SORT + ReID + CMC (0.930 IDF1 on augmented data, against 0.914 without ReID and 0.877 for StrongSORT); T3 re-tunes it for experiment 09's pose detector (0.901 IDF1, coverage 0.982 against T2's 0.958; `track_low_thresh` rose to 0.2373, so the low detector threshold does not reach the tracks) |
| 4 | [The first full-length run: six sessions, end to end](04_six_session_run.md) | short: every stage runs at full length, at about real time (185 ms per frame); engagement at chance, in a design that cannot test the models fairly (one session per activity); the features carry a consistent within-session signal (class motion against attention) that the models do not use. Led to per-labeler training (05) and the full-scale run (06) |
| 5 | [Per-labeler training: first runs on the six sessions](05_per_labeler.md) | short: the per-labeler path runs; at chance by construction; one expert made 59% of the entries; constant 3 has the lowest MAE; a six-session augmentation gain that 06 does not confirm |
| 6 | [The labels and the full pipeline](06_full_pipeline_and_labels.md) ([config](06_full_pipeline_and_labels.json)) | **frozen**: the labels set a low ceiling (Krippendorff's α −0.06 to +0.04 on every unit and interpolation; the other experts predict an entry at AUC 0.564 and no human combination beats constant 3 on MAE); the model predicts 3 for 96% of entries (AUC 0.527, MAE 0.674 against constant 3's 0.619) and underfits, with logistic regression, a random forest or k-NN (SVM dropped as too slow). Includes the label analysis promised in row 1, perspectivism, and interpolation |
| 7 | [k-NN tuning: which k?](07_knn.md) ([config](07_knn.json)) | **frozen**: a larger k generalises better (test Spearman 0.017 at k 1, 0.038 at k 5, 0.083 at k 500); k = 500 chosen (k 1,000 degenerates for the smallest labeler); still below logistic regression; the self-report at chance for every k |
| 8 | [PCA and dimensionality reduction](08_pca.md) ([config](08_pca.json)) | **frozen**: the vector (95 inputs, every column against every labeler: the features look fine for what the labels allow, optical flow tracking the experts at ρ ≈ 0.08). PCA helps no model (logistic regression, the forest, k-NN at k = 500 and k = 5; PCA 5 worst everywhere); the 27 per-student columns alone tie or beat all 95, so the class context goes. Every arm within ±0.02, below the experts' mean, the self-report at chance: the limit is the input's information |
| 9 | [Visual focus of attention, and class-imbalance strategies](09_vfoa.md) ([config](09_vfoa.json)) | done (24 November): head direction from the context cameras (pose model as the detector, 9 head-pose columns) moves the experts' ranking by at most 0.01 and leaves the self-report at chance; scale shrinking and filtered training data widen the predictions without improving the ranking; k-NN regression beats the k-NN classifier at the same k but not logistic regression. 176 runs |
| 10 | [Class-relative and dynamic-clustering features](10_clustering.md) ([config](10_clustering.json): 09's, unchanged; [settings](10_clustering_settings.json)) | done (27 November): students move together on all 10 channels and the groups persist, but describing a student against the class moves the experts' ranking by at most +0.01–0.016 (agglomerative clustering; the pre-registered `sync+agglomerative` test is not supported). Per-track normalisation takes the self-report from below chance to chance (+0.115, supported): a fix to the baseline, not a gain. Constant 3 keeps the lowest MAE. After the run: entries that look alike in the features are rated no more alike than random pairs between students (Appendix C), and a model trained within the session does no better (Appendix D): further gains need new information in the input |
| 11 | [Class imbalance: what it does and how to report around it](11_class_imbalance.md) ([config](11_class_imbalance.json): 10's, unchanged) | done (28 November): the imbalance sits in the tails (4s and 5s are 16% of expert entries, 1s absent) and is not what limits the models (09); rounded, they answer 3 for 96% of entries with no agreement (QWK 0.000). Cut points per labeler on the same scores give QWK 0.10 (about 75% of another expert's 0.138), macro-F1 0.29 against 0.17, every class predicted; no gain on the self-report. Report QWK, macro-F1 and balanced accuracy; MAE favours a constant 3 |

**Delayed (18 November 2026).** The original plan after 08, kept here so the work is not lost; it
will likely change once 09 shows what new information buys.
- **Mixture of experts against a single decoder.** A gate predicts the instructional activity from
  scene features and weights activity-specific experts. Its one full-scale run tied with a single
  decoder (expert-entry AUC 0.526 against 0.527, results `38c985c0f859` / `c5341cb6385e`, archived
  with 06). To be redone on whichever input wins.
- **Dynamic-clustering synchrony features** (guide 15). Class-relative features: the student
  against the rest of the class (option 0, the baseline any clustering must beat), threshold-based
  agglomerative clustering, and a prototype codebook fitted per fold. Needs the per-track CFMU
  contract change (guide 15 §15.7). Head direction from 09 is its natural input: the classroom
  synchrony in the literature is head-rotation synchrony.

Status as of 18 November 2026. Augmentation is covered in 06 §2.3 (one copy changes nothing; heavier augmentation not pursued). From experiment 06 on, each document has its frozen config next to it
(`NN_name.json`): run its commands with `--config docs/experiments/NN_name.json`.

Conventions: every number states the configuration or command that produced it; anything not
yet verified on data says so.

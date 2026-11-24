# Visual focus of attention, and class-imbalance strategies

*18–24 November 2026. Config [`09_vfoa.json`](09_vfoa.json), hash `b1c2d485d3db`: experiment 08's
base with the head-pose unit, the pose model as the person detector, and 4 augmented copies. 176
training runs; their results, logs and settings are listed in
`outputs/experiments/09_vfoa/` (`all_arms.csv`, `logs/`, `arms.sh`).*

## 1. Why this experiment

**06–08 hit a wall that is not the model.** Experiment 08 tried every transformation of the 95
inputs within reach (PCA, dropping the class context, three model families, k-NN's k). None moved
the held-out ranking of the experts' labels by more than ±0.02, and the self-report stayed at
chance. Two things limit what any model can do with these labels:

- **The labels disagree.** The human ceiling is low: one expert's rating predicted by the other
  experts' mean reaches only Spearman 0.14 (AUC 0.56), and the experts predict the student's own
  self-report at Spearman 0.24 (AUC 0.62) (experiment 06 Part 1). A model is scored against one
  labeler at a time, so it cannot do much better than another person would.
- **The labels are imbalanced.** Ratings 2 and 3 are 85% of the expert entries (42% and 43%), 1 is
  almost never used and 5 is rare (06 §1.2). Every model so far answers with the middle: about 96%
  of predictions round to 3, and a constant 3 has the lowest MAE of all.

**Three comparisons**, each against the same reference:

1. **VFoA: new information.** None of the 95 inputs measures the head, and head direction is what
   the classroom attention literature leans on: gaze and head pose with a synchrony feature
   (Goldberg et al., *Educational Psychology Review*, 2019), head-rotation synchrony across a class
   (Raca, Tormey and Dillenbourg, *Journal of Learning Analytics*, 2016), and head pose alone carrying
   most of the signal in the one outside work on DIPSER (`docs/misc/benchmarks.md` §3.4). Comparison 1
   adds head direction, estimated from the classroom cameras, with 08's methodology.
2. **Scale shrinking: against imbalance.** Ratings 1–2 become LOW, 3 MID, 4–5 HIGH, so the thin
   tails are merged into classes of usable size.
3. **Filtered training data: against imbalance.** On the 1–5 scale, the 2s and 3s in the training
   data are thinned so the classes are less disparate.

## 2. Setup

**Head direction from the context cameras.** The pose model (`yolo11s-pose`, full frame at 1280)
replaces the person detector, so every track gets the keypoints of its own detection. It was
re-validated the way 02 and 03 validated YOLO11n: detector threshold D3 = 0.0535 (02, D3) and the
tracker re-tuned as T3 (IDF1 0.901, 98.2% of student-frames on a track; 03, T3). The head-pose unit
(`src/bfmu/head_pose.py`; design in guide 14) turns the keypoints into a yaw proxy (nose against the
ear midpoint), a head-height proxy (nose against the shoulders), face-keypoint confidence and ear
asymmetry. Per student and window: 9 columns (`trk.hp.*`: mean and SD of yaw and of head height,
mean |yaw|, share of frames with the head low, face confidence, ear asymmetry, share of frames
with a pose). A pilot chose the model and input (Appendix B): pose found for 99% of tracked students, 27 ms per
frame. The 52 recordings were re-extracted with 4 augmented copies each (260 recordings, about 17 h
on 3 workers).

**Inputs**, each with the class context (`vec.*`, the class mean and SD of every per-student
column) off and on:

| Input | Columns |
|---|---|
| 1. 08's inputs | the 27 per-student columns (head pose excluded) |
| 2. + head pose | 27 + 9 |
| 3. head pose alone | the 9 head-pose columns |
| 4. + class-relative head direction | input 2 + 4 columns: the student's yaw and head height against the window's class median, \|yaw\| deviation, and the share of the class turned the same way (`engagement.class_relative`) |

**Models and scoring, as in 06–08.** Experiment 08's four models: logistic regression (LR), the
random forest (RF: 200 trees, depth 6), and k-NN at k = 500 and k = 5. One model per labeler, three
folds by group; every number is on the held-out group, scored against that labeler's own entries,
averaged over folds. Two label sets: the **experts'** labels and the student's **self-report**. The
human-ceiling rows are the other experts' mean and median predicting the same entries (on all
entries of the track, so not fold by fold).

**What is read.** Ranking: AUC (engaged = rating ≥ 3) and Spearman. Spread: the predictions' SD
over the labels' SD, and the share of predictions that round to the middle. Calibration: MAE and
macro-F1, compared only within one scale.

## 3. The strategies and the k-NN variant

**Comparison 2, scale shrinking** (`labels.collapse=[[1,2],[3],[4,5]]`). The ratings are mapped
when the label files are read (`src/labels/session.py`), so training, the human ceiling and the
baselines all see three levels, at 1, 3 and 5 on the old scale. The engaged split (≥ 3) is
unchanged, so AUC compares with comparison 1. MAE is in rating units, where a level step is now 2,
so `mae_levels` (MAE in level steps) is reported as well, against a human ceiling rerun on the three levels
(`outputs/labels/agreement_lmh.json`). Run on inputs 1 and 2.

**Comparison 3, filtered training data** (`labels.train_keep={"2": p, "3": p}`). Each training
entry rated 2 or 3 is kept with probability p (per entry, seeded); 1, 4 and 5 keep every entry, and
held-out entries are always scored in full. p = 0.26 makes the 2s and 3s about as frequent as the
4s (9,875 and 10,214 training entries become 2,495 and 2,605, against 2,734 4s); p = 0.5 is milder.
Run on inputs 1 and 2.

**k-NN as regression** (`expert.name=knn_regression`). Since 06, k-NN has been a classifier: the
rating is cut at 3, the larger side undersampled, and P(engaged) read back as a rating, so 3, 4 and
5 count the same and the balancing pulls the output off the label distribution. The regressor
predicts the mean rating of the k nearest training rows instead, with no threshold and no
balancing. It runs at k = 5 and k = 50 in all 32 arms (four inputs, context off and on, in all
three comparisons), and the classifier is rerun at k = 50 in the same arms, so the two can be
compared at the same k (k = 500 and 15 were left out for compute time).

## 4. Results

### 4.1 Comparison 1: head direction adds nothing that ranks

**Experts' labels**, held-out, AUC / Spearman:

| Input | Context | LR | RF | k-NN 500 | k-NN 5 |
|---|---|---|---|---|---|
| 1. 08's inputs | off | 0.533 / 0.103 | 0.532 / 0.102 | 0.540 / 0.072 | 0.522 / 0.034 |
|  | on | 0.534 / 0.108 | 0.527 / 0.103 | 0.530 / 0.075 | 0.518 / 0.033 |
| 2. + head pose | off | 0.535 / 0.103 | 0.534 / 0.108 | 0.546 / 0.085 | 0.530 / 0.047 |
|  | on | 0.537 / 0.111 | 0.530 / 0.110 | 0.536 / 0.085 | 0.521 / 0.041 |
| 3. head pose alone | off | 0.502 / 0.044 | 0.511 / 0.067 | 0.511 / 0.017 | 0.503 / 0.004 |
|  | on | 0.506 / 0.052 | 0.508 / 0.067 | 0.514 / 0.021 | 0.511 / 0.017 |
| 4. + class-relative | off | 0.534 / 0.102 | 0.534 / 0.109 | 0.546 / 0.085 | 0.530 / 0.046 |
|  | on | 0.536 / 0.110 | 0.530 / 0.110 | 0.538 / 0.089 | 0.522 / 0.040 |

Human ceiling (the other experts' mean / median): AUC 0.563 / 0.543, Spearman 0.143 / 0.120. MAE: constant 3 0.619; human ceiling 0.668 (mean of experts) / 0.638 (median); the models (input 2, context off): LR 0.675, RF 0.672, k-NN 500 0.676, k-NN 5 0.963.

**Self-report**, held-out, AUC / Spearman:

| Input | Context | LR | RF | k-NN 500 | k-NN 5 |
|---|---|---|---|---|---|
| 1. 08's inputs | off | 0.481 / -0.034 | 0.471 / -0.025 | 0.510 / -0.024 | 0.520 / 0.032 |
|  | on | 0.507 / 0.009 | 0.491 / -0.005 | 0.499 / -0.023 | 0.513 / 0.016 |
| 2. + head pose | off | 0.457 / -0.086 | 0.472 / -0.046 | 0.503 / -0.037 | 0.502 / -0.007 |
|  | on | 0.482 / -0.042 | 0.494 / -0.024 | 0.501 / -0.032 | 0.511 / 0.014 |
| 3. head pose alone | off | 0.456 / -0.107 | 0.478 / -0.090 | 0.501 / -0.047 | 0.500 / -0.037 |
|  | on | 0.454 / -0.106 | 0.482 / -0.065 | 0.496 / -0.060 | 0.531 / 0.036 |
| 4. + class-relative | off | 0.444 / -0.112 | 0.468 / -0.048 | 0.503 / -0.036 | 0.513 / 0.022 |
|  | on | 0.475 / -0.053 | 0.488 / -0.020 | 0.505 / -0.028 | 0.513 / 0.026 |

Human ceiling (the other experts' mean / median): AUC 0.619 / 0.609, Spearman 0.241 / 0.213. MAE: constant 3 0.668; human ceiling 0.689 (mean of experts) / 0.661 (median); the models (input 2, context off): LR 0.713, RF 0.711, k-NN 500 0.722, k-NN 5 1.028.

- **Head pose adds at most 0.01.** From input 1 to 2, AUC moves by +0.002 to +0.008 and Spearman by
  0.000 to +0.013, for every model, with the context off and on. The class-relative columns
  (input 4) change nothing further, and on the experts' labels the class context changes no model
  by more than 0.01.
- **Head pose alone carries little.** Input 3 ranks the experts' labels at AUC 0.50–0.51 and
  Spearman 0.00–0.07, against 0.52–0.55 and 0.03–0.11 with 08's inputs. Optical flow and
  behaviour remain the signal.
- **On the experts' labels, the models sit at the median expert's level.** The best arms reach
  AUC 0.546 (k-NN 500) and Spearman 0.111 (LR), against 0.543 / 0.120 for the median of the other experts and
  0.563 / 0.143 for their mean. A constant 3 still has the lowest MAE (0.619), lower than any model
  and lower than the human ceiling.
- **The self-report stays at chance; LR and RF fall below it, most of all with head pose.** AUC is 0.44–0.53
  for every arm, while the experts predict the self-report at AUC 0.61–0.62 and Spearman 0.21–0.24.
  With head pose, LR and RF rank the held-out group's self-report in the wrong direction (LR,
  inputs 2–4 without the context: Spearman −0.09 to −0.11): what they learn from head pose on two
  groups' self-reports reverses on the third. This run does not show why.

### 4.2 Comparisons 2 and 3: the strategies change the spread, not the ranking

Input 2 (08's inputs plus head pose), context off; the other arms read the same
(`all_arms.csv`). **Experts' labels**, held-out:

| Model | Strategy | AUC | Spearman | Pred. SD / label SD | Share predicted mid | Macro F1 | MAE |
|---|---|---|---|---|---|---|---|
| LR | Comparison 1 (reference) | 0.535 | 0.103 | 0.25 | 0.96 | 0.138 | 0.675 |
|  | 2. Scale shrinking | 0.543 | 0.108 | 0.26 | 0.97 | 0.226 | 1.262 (0.631 levels) |
|  | 3. Filtering, p = 0.26 | 0.515 | 0.085 | 0.39 | 0.80 | 0.164 | 0.786 |
|  | 3. Filtering, p = 0.5 | 0.522 | 0.093 | 0.32 | 0.93 | 0.145 | 0.699 |
| RF | Comparison 1 (reference) | 0.534 | 0.108 | 0.22 | 0.98 | 0.132 | 0.672 |
|  | 2. Scale shrinking | 0.545 | 0.112 | 0.21 | 0.98 | 0.211 | 1.257 (0.628 levels) |
|  | 3. Filtering, p = 0.26 | 0.516 | 0.092 | 0.38 | 0.83 | 0.159 | 0.782 |
|  | 3. Filtering, p = 0.5 | 0.523 | 0.101 | 0.30 | 0.95 | 0.138 | 0.692 |
| k-NN 500 | Comparison 1 (reference) | 0.546 | 0.085 | 0.24 | 0.98 | 0.129 | 0.676 |
|  | 2. Scale shrinking | 0.546 | 0.083 | 0.13 | 1.00 | 0.198 | 1.210 (0.605 levels) |
|  | 3. Filtering, p = 0.26 | 0.530 | 0.081 | 0.25 | 0.98 | 0.129 | 0.672 |
|  | 3. Filtering, p = 0.5 | 0.535 | 0.081 | 0.23 | 0.99 | 0.129 | 0.676 |
| k-NN 5 | Comparison 1 (reference) | 0.530 | 0.047 | 1.07 | 0.55 | 0.183 | 0.963 |
|  | 2. Scale shrinking | 0.530 | 0.047 | 0.61 | 0.72 | 0.302 | 1.432 (0.716 levels) |
|  | 3. Filtering, p = 0.26 | 0.522 | 0.053 | 1.13 | 0.51 | 0.191 | 0.982 |
|  | 3. Filtering, p = 0.5 | 0.526 | 0.050 | 1.10 | 0.53 | 0.193 | 0.971 |
| *Constant 3* | *1â€“5* | 0.500 | â€“ | 0 | 1.00 | 0.119 | 0.619 |
| *Human ceiling, mean of experts* | *1â€“5* | 0.563 | 0.143 | â€“ | â€“ | 0.208 | 0.668 |
| *Human ceiling, median of experts* | *1â€“5* | 0.543 | 0.120 | â€“ | â€“ | 0.222 | 0.638 |
| *Constant MID* | *LOW / MID / HIGH* | 0.500 | â€“ | 0 | 1.00 | 0.198 | 1.153 (0.577 levels) |
| *Human ceiling, mean of experts* | *LOW / MID / HIGH* | 0.567 | 0.149 | â€“ | â€“ | 0.350 | 1.222 (0.611 levels) |
| *Human ceiling, median of experts* | *LOW / MID / HIGH* | 0.543 | 0.121 | â€“ | â€“ | 0.366 | 1.183 (0.592 levels) |

**Self-report**, held-out:

| Model | Strategy | AUC | Spearman | Pred. SD / label SD | Share predicted mid | Macro F1 | MAE |
|---|---|---|---|---|---|---|---|
| LR | Comparison 1 (reference) | 0.457 | -0.086 | 0.10 | 1.00 | 0.117 | 0.713 |
|  | 2. Scale shrinking | 0.468 | -0.088 | 0.10 | 1.00 | 0.194 | 1.277 (0.639 levels) |
|  | 3. Filtering, p = 0.26 | 0.448 | -0.078 | 0.19 | 0.79 | 0.143 | 0.904 |
|  | 3. Filtering, p = 0.5 | 0.452 | -0.082 | 0.15 | 0.99 | 0.121 | 0.755 |
| RF | Comparison 1 (reference) | 0.472 | -0.046 | 0.09 | 1.00 | 0.118 | 0.711 |
|  | 2. Scale shrinking | 0.486 | -0.032 | 0.09 | 1.00 | 0.195 | 1.275 (0.638 levels) |
|  | 3. Filtering, p = 0.26 | 0.472 | -0.011 | 0.17 | 0.89 | 0.136 | 0.875 |
|  | 3. Filtering, p = 0.5 | 0.472 | -0.034 | 0.12 | 0.99 | 0.122 | 0.745 |
| k-NN 500 | Comparison 1 (reference) | 0.503 | -0.037 | 0.18 | 1.00 | 0.118 | 0.722 |
|  | 2. Scale shrinking | 0.503 | -0.036 | 0.11 | 1.00 | 0.195 | 1.230 (0.615 levels) |
|  | 3. Filtering, p = 0.26 | 0.520 | 0.018 | 0.12 | 1.00 | 0.117 | 0.703 |
|  | 3. Filtering, p = 0.5 | 0.505 | -0.014 | 0.15 | 1.00 | 0.117 | 0.712 |
| k-NN 5 | Comparison 1 (reference) | 0.502 | -0.007 | 0.99 | 0.54 | 0.192 | 1.028 |
|  | 2. Scale shrinking | 0.502 | -0.006 | 0.60 | 0.71 | 0.292 | 1.486 (0.743 levels) |
|  | 3. Filtering, p = 0.26 | 0.495 | 0.004 | 1.03 | 0.51 | 0.183 | 1.058 |
|  | 3. Filtering, p = 0.5 | 0.502 | -0.001 | 1.02 | 0.52 | 0.192 | 1.049 |
| *Constant 3* | *1â€“5* | 0.500 | â€“ | 0 | 1.00 | 0.117 | 0.668 |
| *Human ceiling, mean of experts* | *1â€“5* | 0.619 | 0.241 | â€“ | â€“ | 0.179 | 0.689 |
| *Human ceiling, median of experts* | *1â€“5* | 0.609 | 0.213 | â€“ | â€“ | 0.179 | 0.661 |
| *Constant MID* | *LOW / MID / HIGH* | 0.500 | â€“ | 0 | 1.00 | 0.195 | 1.175 (0.588 levels) |
| *Human ceiling, mean of experts* | *LOW / MID / HIGH* | 0.622 | 0.235 | â€“ | â€“ | 0.285 | 1.204 (0.602 levels) |
| *Human ceiling, median of experts* | *LOW / MID / HIGH* | 0.609 | 0.217 | â€“ | â€“ | 0.299 | 1.155 (0.578 levels) |

- **Scale shrinking does not stop median hugging.** LR, RF and k-NN at k = 500 still predict MID
  for 97–100% of entries. Ranking rises slightly for LR and RF (AUC +0.008 to +0.011, Spearman
  +0.004 to +0.005); the k-NN classifiers do not change, since their engaged split is the same on
  both scales. Macro-F1 rises only because there are three classes: every model stays below the
  human ceiling (0.350 / 0.366), and in level steps every model's MAE (0.61–0.72) is worse than a
  constant MID (0.577).
- **Filtering widens the predictions and costs ranking.** At p = 0.26, LR's share of predictions
  at 3 falls from 0.96 to 0.80 and its spread rises from 0.25 to 0.39 of the labels' SD, but AUC
  drops by about 0.02, Spearman by 0.02 and MAE rises by 0.11 (RF the same); p = 0.5 is halfway.
  The k-NN classifiers' spread does not change, since they rebalance the engaged split
  themselves, but their AUC still drops by 0.01–0.02.
- **Neither strategy reaches the self-report.** Every arm stays at AUC 0.45–0.52.

So class imbalance is not what keeps the models at the middle: with the classes made less
disparate, the models spread their predictions, but the extra spread is not correct, because the
ranking does not improve.

### 4.3 k-NN: regression against the classifier

Comparison 1, mean over the six arms with 08's inputs (inputs 1, 2 and 4, context off and on),
held-out. **Experts' labels**:

| k-NN | AUC | Spearman | MAE | Pred. SD / label SD | Share predicted 3 |
|---|---|---|---|---|---|
| Classifier, k = 5 | 0.524 | 0.040 | 0.943 | 1.03 | 0.57 |
| Regression, k = 5 | 0.521 | 0.058 | 0.712 | 0.48 | 0.75 |
| Classifier, k = 50 | 0.531 | 0.062 | 0.727 | 0.45 | 0.85 |
| Regression, k = 50 | 0.523 | 0.087 | 0.679 | 0.28 | 0.94 |
| Classifier, k = 500 | 0.539 | 0.082 | 0.670 | 0.22 | 0.99 |
| *LR, for reference* | 0.535 | 0.106 | 0.678 | 0.28 | 0.95 |

**Self-report**:

| k-NN | AUC | Spearman | MAE | Pred. SD / label SD | Share predicted 3 |
|---|---|---|---|---|---|
| Classifier, k = 5 | 0.512 | 0.017 | 1.006 | 0.97 | 0.54 |
| Regression, k = 5 | 0.506 | 0.011 | 0.787 | 0.45 | 0.79 |
| Classifier, k = 50 | 0.519 | 0.014 | 0.784 | 0.39 | 0.86 |
| Regression, k = 50 | 0.512 | 0.023 | 0.726 | 0.19 | 0.99 |
| Classifier, k = 500 | 0.503 | -0.030 | 0.722 | 0.16 | 1.00 |
| *LR, for reference* | 0.474 | -0.053 | 0.722 | 0.14 | 0.99 |

- **At the same k, regression is better calibrated and ranks a little better.** At k = 5 and 50,
  regression has the lower MAE (0.712 against 0.943; 0.679 against 0.727) and the higher Spearman
  (+0.018, +0.025), at an AUC 0.003–0.008 lower. The classifier's wide spread at k = 5 (SD ratio
  1.03) is mostly error: its MAE is the worst of any model.
- **A larger k helps both, by smoothing.** From 5 to 50 (and 500 for the classifier), Spearman and
  MAE improve as the predictions shrink toward the middle, as 07 found for the classifier.
- **Regression at k = 50 matches LR's MAE (0.679 against 0.678), with a lower Spearman (0.087
  against 0.106).** k-NN does not beat the linear model in any form. In comparisons 2 and 3 the
  regressor reads like LR (all arms in `all_arms.csv`).
- **The self-report is at chance for every k-NN.**

## 5. Conclusion

- **Head direction from the classroom cameras adds no usable signal.** With 08's methodology, it
  moves the ranking of the experts' labels by at most 0.01 and does not reach the self-report.
  The pose model is a good detector and tracker and finds a pose for 99% of students, so the
  features exist and are measured; they do not track how the labelers rated attention.
- **The models already sit near the median expert on the experts' labels** (Spearman 0.11 against
  0.12). What is left is the disagreement between labelers, which no input can remove.
- **Class imbalance is not the bottleneck.** Shrinking the scale and filtering the training data
  change how spread out the predictions are, not whether they are right: ranking stays flat or
  drops, and a constant middle rating keeps the lowest error on both scales.
- **k-NN regression is the better k-NN** (lower MAE, higher Spearman than the classifier at the same
  k), but it does not beat logistic regression.
- **The self-report is the open gap.** The experts predict it at Spearman 0.21–0.24 from the same
  video; every model, with or without head pose, is at chance. What the experts see that relates to
  the student's own report is not in these features.

**Not run:** the augmentation-dose arm (0 / 1 / 4 copies), per-labeler tables, and rerunning the
feature study with the head-pose columns.

## Appendix A: reproducing

```
python -m scripts.parallel_runs extract --config docs/experiments/09_vfoa.json --workers 3   # resumable
python -m scripts.labels.label_agreement ceiling --config docs/experiments/09_vfoa.json \
    --set 'labels.collapse=[[1,2],[3],[4,5]]' --out outputs/labels/agreement_lmh.json
bash outputs/experiments/09_vfoa/arms.sh all        # comparisons 1–3, 08's four models (80 runs)
bash outputs/experiments/09_vfoa/arms.sh knnreg     # k-NN regression at 5 and 50, classifier at 50 (96 runs)
```

Each arm is `python -m scripts.train_models --config docs/experiments/09_vfoa.json --set` plus its
settings (`arms.sh`); arm names are `<model>_<comparison>_<input>[_ctx]`, and `all_arms.csv` gives
every arm's hash and held-out and training metrics on both label sets. The detector (D3) and tracker
(T3) scans are in experiments 02 and 03; the pose-input pilot is in `outputs/experiments/09_vfoa/pilot/`
(`scripts/analysis/head_pose_pilot.py`). Runtime: about 0.8 min of extraction per minute of footage
on 3 workers (tracking 41% of it), and 8–25 min per training run (`wallclock.py`).

## Appendix B: the pose-input pilot

`scripts/analysis/head_pose_pilot.py`: the unit in five variants on 40 frames from each of three
sessions (one per group), on experiment 06's cached tracks. Heads are about 35–50 px wide in the
1280×720 frames, so the input resolution was the open question.

| Variant | Seconds per frame | Pose matched | Yaw defined | Head height defined | Face confidence | Yaw p10 / p50 / p90 |
|---|---|---|---|---|---|---|
| nano, full frame at 640 | 0.026 | 0.97 | 0.97 | 0.97 | 0.92 | −0.39 / −0.04 / +0.35 |
| nano, full frame at 1280 | 0.026 | 0.97 | 0.95 | 0.95 | 0.95 | −0.37 / −0.01 / +0.33 |
| nano, upper-body crops at 256 | 0.061 | 1.00 | 0.89 | 0.89 | 0.88 | −0.37 / −0.02 / +0.34 |
| **small, full frame at 1280** | **0.027** | **0.99** | **0.97** | **0.97** | **0.94** | −0.36 / −0.04 / +0.32 |
| small, upper-body crops at 256 | 0.062 | 1.00 | 0.91 | 0.91 | 0.88 | −0.38 / −0.03 / +0.31 |

The small model on the full frame at 1280 was chosen: the best coverage at about the cost of the
nano model, while crops cost twice as much and leave yaw undefined more often. The overlays
(`outputs/experiments/09_vfoa/pilot/<variant>/`) show the signals behave: students turned to one
side read yaw ±0.3–0.5, students facing forward about 0, and heads down over a phone or notes read a
head height of −0.1 to −0.2, against 0.1–0.3 for heads up.

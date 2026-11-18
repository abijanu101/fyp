# k-NN tuning: which k?

*Started 18 November 2026. Config: [`07_knn.json`](07_knn.json), which is experiment 06's config
([`06_full_pipeline_and_labels.json`](06_full_pipeline_and_labels.json)) with the entered rating as
the training target (no interpolation). The input is unchanged from 06: all 95 columns, no PCA.
Done 18 November: **k = 500** is carried into experiment 08.*

**Why this experiment.** In experiment 06, k-NN had the most spread-out predictions of the expert
models (SD 0.74, against 0.19 for logistic regression) and the best held-out ranking (Spearman
0.094, AUC 0.532). But its k (25) was a default, never tuned. The right k depends on how noisy the
labels are and how many rows each model trains on:
- Each labeler's model trains on about 1,000 to 20,000 rows; k-NN also undersamples the majority
  class to balance them.
- Noisy labels favour a larger k, which averages the noise out; the √n heuristic gives about 30 to
  140. A small k follows individual training rows, noise included.

This experiment chooses k on 06's input as it stands. Whether k-NN does better in a smaller space
(the curse of dimensionality: in 95 noisy dimensions, the nearest and farthest training rows are
almost equally far away) is a question about the input. It belongs to experiment 08, which uses the
k chosen here.

## 1. Design

Five arms, identical except for k (`expert.name=knn 'expert.params={"n_neighbors": k}'`), with k ∈ {5,
25, 50, 100, 250}, extended to 500 and 1,000 when the first five had not levelled off, and to
k = 1 (pure memorisation: each prediction copies the single nearest training row) as the other end. Leave one group
out, per-labeler, scored like experiment 06.

```
bash tmp/exp07/knn.sh        # the 8 arms, resumable (log: tmp/exp07/knn.log, with each arm's hash)
python -m scripts.reporting.results_table "k 5=<hash>" "k 25=<hash>" ...
python -m scripts.reporting.results_table "k ...=<hash>" --track self
python -m scripts.reporting.results_table "k ...=<hash>" --by-labeler
```

**Choosing k.** By held-out rank metrics (Spearman, AUC), macro over labelers as well as pooled,
with the gap between training and held-out as the overfitting check. A small k should fit the
training groups closely and generalise worst.

**Reading the metrics.** k-NN is trained on rating ≥ 3 and outputs P(rating ≥ 3), read back as
1 + 4p (`SklearnExpert.SOFT`). Its rank metrics (AUC, Spearman) are comparable with every other
model; its profile, MAE, RMSE and the discretised metrics measure a different quantity, and are
reported for completeness only.

## 2. Result

| k | 1 | 5 | 25 | 50 | 100 | 250 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|
| Results | `0f2d8f2841a9` | `e79c4c117090` | `75a782eb8951` | `efa51da0cd61` | `2fd336ace408` | `dbc58c4983ba` | `2fcb719b3a54` | `386791e78fd1` |

Every data cell is train / test: the same fold's model on its own training groups, then on its
held-out group. Rows in italics are references with one value per cell, on all entries of the track
(experiment 06 Part 1): the self-report ratings and the median of the expert ratings (profile), and
the other experts' mean and median predicting the same labels (metrics). \* threshold (AUC: labels
split at rating ≥ 3); \*\* discretisation (prediction rounded to a whole rating, halves toward 3);
unmarked: the raw prediction. Only the rank metrics (AUC, Spearman) compare with other models.

### 2.1 Expert entries

**Profile** (train / test)

| | Mean | Median | SD |
|---|---|---|---|
| k 1 | 2.982 / 3.042 | 3.000 / 3.000 | 1.902 / 1.611 |
| k 5 | 2.997 / 3.019 | 3.000 / 3.000 | 0.803 / 0.742 |
| k 25 | 2.991 / 2.999 | 2.973 / 2.991 | 0.401 / 0.376 |
| k 50 | 2.991 / 2.997 | 2.991 / 3.000 | 0.309 / 0.291 |
| k 100 | 2.987 / 2.992 | 2.978 / 2.993 | 0.243 / 0.229 |
| k 250 | 2.981 / 2.985 | 2.979 / 2.984 | 0.179 / 0.162 |
| k 500 | 2.979 / 2.984 | 2.983 / 2.995 | 0.141 / 0.126 |
| k 1000 | 2.976 / 2.981 | 2.983 / 2.990 | 0.112 / 0.102 |
| Constant 3 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 |
| *Self-report labels* | 2.885 | 3.000 | 0.895 |
| *Median of expert labels* | 2.895 | 3.000 | 0.510 |

**Metrics** (train / test)

| | MAE | RMSE | AUC* | Spearman | Macro F1** | Accuracy** |
|---|---|---|---|---|---|---|
| k 1 | 1.277 / 1.531 | 1.430 / 1.811 | 0.964 / 0.509 | 0.812 / 0.017 | 0.084 / 0.115 | 0.090 / 0.147 |
| k 5 | 0.730 / 0.890 | 0.909 / 1.102 | 0.744 / 0.517 | 0.427 / 0.038 | 0.257 / 0.185 | 0.427 / 0.360 |
| k 25 | 0.671 / 0.730 | 0.829 / 0.900 | 0.643 / 0.521 | 0.276 / 0.049 | 0.203 / 0.166 | 0.426 / 0.399 |
| k 50 | 0.663 / 0.701 | 0.821 / 0.870 | 0.622 / 0.526 | 0.246 / 0.059 | 0.173 / 0.149 | 0.427 / 0.414 |
| k 100 | 0.656 / 0.680 | 0.820 / 0.852 | 0.604 / 0.528 | 0.219 / 0.068 | 0.149 / 0.135 | 0.426 / 0.421 |
| k 250 | 0.647 / 0.661 | 0.819 / 0.839 | 0.589 / 0.528 | 0.196 / 0.072 | 0.127 / 0.121 | 0.427 / 0.423 |
| k 500 | 0.642 / 0.649 | 0.822 / 0.833 | 0.576 / 0.533 | 0.175 / 0.083 | 0.121 / 0.119 | 0.427 / 0.423 |
| k 1000 | 0.637 / 0.642 | 0.824 / 0.831 | 0.563 / 0.534 | 0.153 / 0.085 | 0.120 / 0.119 | 0.427 / 0.423 |
| Constant 3 | 0.618 / 0.619 | 0.840 / 0.839 | 0.500 / 0.500 | - / - | 0.120 / 0.119 | 0.427 / 0.423 |
| *Mean of experts* | 0.668 | 0.860 | 0.563 | 0.143 | 0.208 | 0.447 |
| *Median of experts* | 0.638 | 0.913 | 0.543 | 0.120 | 0.222 | 0.451 |

### 2.2 Self-report entries

**Profile** (train / test)

| | Mean | Median | SD |
|---|---|---|---|
| k 1 | 3.052 / 3.113 | 3.000 / 3.000 | 1.876 / 1.590 |
| k 5 | 3.039 / 3.043 | 3.000 / 3.089 | 0.856 / 0.775 |
| k 25 | 3.034 / 3.030 | 3.053 / 3.027 | 0.392 / 0.375 |
| k 50 | 3.031 / 3.029 | 3.027 / 3.013 | 0.286 / 0.284 |
| k 100 | 3.032 / 3.028 | 3.020 / 3.018 | 0.215 / 0.209 |
| k 250 | 3.027 / 3.026 | 3.016 / 3.013 | 0.147 / 0.144 |
| k 500 | 3.029 / 3.031 | 3.027 / 3.023 | 0.110 / 0.103 |
| k 1000 | 3.036 / 3.039 | 3.038 / 3.039 | 0.076 / 0.070 |
| Constant 3 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 |
| *Self-report labels* | 2.885 | 3.000 | 0.895 |
| *Median of expert labels* | 2.876 | 3.000 | 0.428 |

**Metrics** (train / test)

| | MAE | RMSE | AUC* | Spearman | Macro F1** | Accuracy** |
|---|---|---|---|---|---|---|
| k 1 | 1.238 / 1.541 | 1.420 / 1.843 | 0.961 / 0.500 | 0.759 / 0.004 | 0.113 / 0.127 | 0.124 / 0.158 |
| k 5 | 0.750 / 0.959 | 0.942 / 1.192 | 0.808 / 0.527 | 0.489 / 0.033 | 0.280 / 0.187 | 0.415 / 0.336 |
| k 25 | 0.727 / 0.799 | 0.905 / 0.993 | 0.669 / 0.499 | 0.273 / -0.016 | 0.194 / 0.156 | 0.408 / 0.381 |
| k 50 | 0.723 / 0.770 | 0.905 / 0.964 | 0.630 / 0.500 | 0.214 / -0.023 | 0.158 / 0.135 | 0.406 / 0.393 |
| k 100 | 0.717 / 0.740 | 0.906 / 0.938 | 0.605 / 0.511 | 0.178 / -0.005 | 0.129 / 0.125 | 0.407 / 0.405 |
| k 250 | 0.709 / 0.723 | 0.908 / 0.927 | 0.583 / 0.503 | 0.134 / -0.020 | 0.118 / 0.116 | 0.412 / 0.410 |
| k 500 | 0.704 / 0.711 | 0.911 / 0.922 | 0.564 / 0.508 | 0.102 / -0.012 | 0.117 / 0.116 | 0.412 / 0.411 |
| k 1000 | 0.700 / 0.704 | 0.914 / 0.920 | 0.549 / 0.515 | 0.069 / -0.008 | 0.117 / 0.116 | 0.412 / 0.411 |
| Constant 3 | 0.668 / 0.670 | 0.910 / 0.910 | 0.500 / 0.500 | - / - | 0.117 / 0.116 | 0.412 / 0.411 |
| *Mean of experts* | 0.689 | 0.884 | 0.619 | 0.241 | 0.179 | 0.435 |
| *Median of experts* | 0.661 | 0.914 | 0.609 | 0.213 | 0.179 | 0.439 |

### 2.3 Reading

- **k = 1 memorises and carries nothing.** Each prediction copies one neighbour's label, so it is a
  1 or a 5 (SD 1.6–1.9). It fits its training groups almost perfectly (AUC 0.964, Spearman 0.812;
  not 1, because training rows were undersampled to balance the classes and entries are averaged
  over tracks) and is at chance on new groups (AUC 0.509, Spearman 0.017). One neighbour's label is
  worth nothing here: neighbouring rows share their features, not their ratings.
- **A larger k generalises better, all the way up.** Test Spearman rises from 0.038 (k 5) to 0.083
  (k 500) and 0.085 (k 1,000); AUC from 0.517 to 0.533–0.534. The training fit falls the other way
  (Spearman 0.427 → 0.153), so a small k memorises its training groups and a large one smooths the
  noisy labels. With labels this noisy, averaging over hundreds of rows is what works.
- **The gain flattens after k 500,** and k 1,000 breaks down for the smallest labeler. labeler_05's
  model trains on fewer balanced rows than 1,000, so it predicts a constant (its Spearman becomes
  undefined), and the macro mean over labelers swings with that one labeler (−0.45 to undefined).
  Pooled, k 1,000 is ahead of k 500 by 0.002.
- **So k = 500.** It is within 0.002 of the best pooled score and still a k-NN for every labeler.
  Experiment 08 uses it (`expert.params={"n_neighbors": 500}`).
- **The spread collapses as k grows.** The test prediction SD falls from 0.74 (k 5) to 0.13 (k 500):
  a large-k k-NN predicts close to the local mean, which is close to the overall mean. It ends up
  ranking like a smoothed linear model, and still below logistic regression on the same input
  (test Spearman 0.097, experiment 08 §3).
- **Against the human ceiling.** At k 500 the test Spearman (0.083) is about half the other
  experts' mean (0.143) and below their median (0.120); AUC 0.533 against 0.563.
- **The self-report stays at chance for every k** (test AUC 0.50–0.53), where the experts' mean
  reaches AUC 0.619.

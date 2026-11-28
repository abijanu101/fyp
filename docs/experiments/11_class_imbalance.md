# Class imbalance: what it does and how to report around it

*28 November 2026. Config [`11_class_imbalance.json`](11_class_imbalance.json): experiment 10's
config, unchanged (the same hash, `b1c2d485d3db`, as [`09_vfoa.json`](09_vfoa.json)). No model is
trained: the analysis reads experiment 10's saved held-out predictions
(`outputs/experiments/10_clustering_v2/ablation/ablation_predictions.csv.gz`, group split) with
`python -m scripts.analysis.imbalance_scoring`; results in
`outputs/experiments/10_clustering_v2/imbalance/`. It brings together what experiments 06, 09 and 10
found about the imbalance, and adds a way to turn the models' scores into 1–5 ratings that use the
scale as the labelers do.*

## 1. Why this experiment

**Every model so far answers 3.** About 96% of the experts' predictions round to 3 (experiments
06–10), and a constant 3 has the lowest MAE of every model and of the human ceiling. Two causes are
possible, and they call for different fixes:
- **The imbalance:** the ratings bunch at 2 and 3, so a model fitted to them stays in the middle.
- **The signal:** the inputs predict the ratings only weakly, so staying near the mean is what the
  squared error asks for (experiment 06 §1.2: the error-optimal spread is r × the label SD).

Experiment 09 tried the first fix and found it was not the cause (§3). This experiment asks what is
left: **can the models' existing scores be turned into ratings that use the whole scale, with real
agreement, and which metrics should report them?**

## 2. The imbalance

**The labels** (experiment 06 §1.2: every entry, the session start excluded; shares of 1 / 2 / 3 / 4 / 5):

| Labeler | Entries | Share of expert entries | Shares 1 / 2 / 3 / 4 / 5 |
|---|---|---|---|
| labeler_01 | 14,258 | 61.5% | 0.00 / 0.41 / 0.47 / 0.10 / 0.02 |
| labeler_02 | 1,263 | 5.4% | 0.00 / 0.33 / 0.47 / 0.19 / 0.01 |
| labeler_03 | 5,702 | 24.6% | 0.00 / 0.46 / 0.32 / 0.11 / 0.10 |
| labeler_04 | 1,370 | 5.9% | 0.00 / 0.37 / 0.42 / 0.16 / 0.05 |
| labeler_05 | 586 | 2.5% | 0.00 / 0.32 / 0.39 / 0.24 / 0.05 |
| **all experts** | 23,179 | 100% | 0.00 / 0.42 / 0.43 / 0.11 / 0.04 |
| self_labeling | 2,634 | – | 0.01 / 0.37 / 0.41 / 0.14 / 0.07 |

**The held-out entries of experiment 10** (group split; experts 21,519, self-report 2,463):

| | 1 | 2 | 3 | 4 | 5 | Engaged (≥ 3) | Predictions that round to 3 (`base`) |
|---|---|---|---|---|---|---|---|
| Experts | 0.0% (3 entries) | 41.7% | 42.7% | 11.1% | 4.5% | 58.3% | 96.4% |
| Self-report | 1.3% | 36.7% | 41.4% | 14.0% | 6.6% | 62.0% | 99.9% |

- **The imbalance is in the tails, not the middle.** 2 and 3 are about equally common, so the engaged
  split that AUC uses (≥ 3 against ≤ 2) is close to balanced (58 / 42), and the ranking metrics (AUC,
  Spearman) are not distorted by it.
- **The experts' rating 1 does not exist** (3 of 21,519 entries): their scale is 2–5. Below it is
  merged into 2.
- **4 and 5 are the minority** (16% of expert entries), and the labelers use them very differently:
  labeler_03 gives a 5 to 10% of entries, labeler_01 to 2.4%, labeler_02 to 0.8%.

## 3. What has been tried

**Experiment 09 §4.2, against the imbalance** (logistic regression, input 2, experts' labels; the
other models read the same):

| Strategy | AUC | Spearman | Pred. SD / label SD | Share predicted mid | Macro F1 | MAE |
|---|---|---|---|---|---|---|
| None (reference) | 0.535 | 0.103 | 0.25 | 0.96 | 0.138 | 0.675 |
| Scale shrinking (1–2 LOW, 3 MID, 4–5 HIGH) | 0.543 | 0.108 | 0.26 | 0.97 | 0.226 | 1.262 (0.631 levels) |
| Filtering the 2s and 3s, p = 0.5 | 0.522 | 0.093 | 0.32 | 0.93 | 0.145 | 0.699 |
| Filtering the 2s and 3s, p = 0.26 | 0.515 | 0.085 | 0.39 | 0.80 | 0.164 | 0.786 |

Shrinking the scale left 97% of predictions at MID; filtering spread the predictions (80% at 3) but
cost ranking (Spearman −0.02) and calibration (MAE +0.11). 09's conclusion: *class imbalance is not
what keeps the models at the middle*.

**Experiment 10 §3.3, the spread on the same input:** `base` spreads its predictions over 0.25 of the
labels' SD, 96% rounding to 3; with Pearson r 0.058 against the labels, the error-optimal SD (r × 0.81
= 0.047) is a fifth of the actual 0.232. The models are not too cautious: with this little signal,
their spread is mostly noise.

## 4. Design

**From score to rating, three ways,** for each arm of experiment 10 (`base`, `normalised`,
`agglomerative`, `sync+agglomerative`, `normalised+sync`), held-out group by held-out group:
- **rounding:** the nearest whole rating, halves toward 3, what every table in 06–10 reports;
- **cut points per labeler:** the labeler's scores in the held-out group are cut at the quantiles
  that give the labeler's class shares in the **other** groups' ratings (the top share-of-5s becomes
  a 5, and so on). The cut points use the held-out group's own predictions, which use no labels, and
  the other groups' labels: no held-out label is used (tested: changing them leaves every mapped
  class the same). Fitting the cut points on the other groups' predictions instead would leak, since
  their models were trained on this group's labels. Within a labeler the order is kept, so ranking
  cannot change;
- **random with the same shares:** ratings drawn at random with the labeler's shares (the mean of 20
  draws), the reference the cut points must beat: matching the shares alone is not skill.

**Metrics.**
- Quadratic weighted kappa (QWK, the standard agreement measure for ordinal ratings: 0 is chance, 1
  perfect).
- Macro-F1 and balanced accuracy (mean recall per class), the recall of each class.
- MAE, macro-MAE (MAE per true class, averaged), and the share within one point.
- From the raw score: Spearman, AUC for engaged (≥ 3), and for "highly engaged" (≥ 4) the ROC AUC and
  precision-recall AUC, against the base rate.
- Classes with fewer than 10 entries are left out of the per-class averages.

**Human reference, on the same entries:** another expert's rating in force at the entry's time, as
it is, and the mean of the other experts' ratings in force, mapped by the same cut points. The rating
tracks are rebuilt from the saved entries, which lack moments when a student was not tracked: an
approximation (the mean comes out at Spearman 0.156 against experiment 06's 0.143).

**Intervals:** bootstrap over class sessions (1,000), paired for the differences.

## 5. Results

### 5.1 The experts' labels

**Every arm, rounding against cut points:**

| Arm | Spearman (score) | QWK: rounding → cut points | Macro-F1: rounding → cut points | Balanced accuracy: rounding → cut points | MAE: rounding → cut points |
|---|---|---|---|---|---|
| `base` | +0.055 | +0.000 → **+0.103** | 0.172 → 0.286 | 0.256 → 0.287 | 0.631 → 0.781 |
| `normalised` | +0.045 | +0.006 → **+0.085** | 0.165 → 0.280 | 0.253 → 0.280 | 0.625 → 0.794 |
| `agglomerative` | +0.065 | +0.005 → **+0.103** | 0.176 → 0.287 | 0.256 → 0.288 | 0.634 → 0.783 |
| `sync+agglomerative` | +0.063 | +0.007 → **+0.098** | 0.180 → 0.284 | 0.256 → 0.285 | 0.636 → 0.785 |
| `normalised+sync` | +0.048 | +0.005 → **+0.081** | 0.169 → 0.282 | 0.254 → 0.282 | 0.629 → 0.795 |

**The differences** (95% interval):

| Arm | Cut points − rounding: QWK | Macro-F1 | Balanced accuracy | Cut points − random with the same shares: QWK | Macro-F1 | Balanced accuracy |
|---|---|---|---|---|---|---|
| `base` | +0.102 [+0.072, +0.132] | +0.113 [+0.094, +0.131] | +0.031 [+0.017, +0.044] | +0.106 [+0.079, +0.132] | +0.034 [+0.024, +0.046] | +0.036 [+0.024, +0.048] |
| `normalised` | +0.078 [+0.053, +0.106] | +0.114 [+0.098, +0.131] | +0.027 [+0.016, +0.039] | +0.081 [+0.052, +0.113] | +0.023 [+0.013, +0.036] | +0.024 [+0.013, +0.036] |
| `agglomerative` | +0.097 [+0.068, +0.126] | +0.110 [+0.092, +0.127] | +0.031 [+0.018, +0.045] | +0.096 [+0.062, +0.128] | +0.031 [+0.018, +0.045] | +0.033 [+0.018, +0.047] |
| `sync+agglomerative` | +0.091 [+0.063, +0.119] | +0.103 [+0.085, +0.119] | +0.028 [+0.015, +0.040] | +0.100 [+0.073, +0.126] | +0.032 [+0.021, +0.043] | +0.033 [+0.021, +0.044] |
| `normalised+sync` | +0.076 [+0.054, +0.100] | +0.112 [+0.098, +0.126] | +0.028 [+0.019, +0.039] | +0.074 [+0.051, +0.097] | +0.026 [+0.015, +0.035] | +0.026 [+0.016, +0.037] |

**`base` in full,** with the references:

| `base` | QWK | Macro-F1 | Balanced accuracy | Recall of 2 / 3 / 4 / 5 | Predicted shares of 2 / 3 / 4 / 5 | MAE | Macro-MAE | Within 1 point |
|---|---|---|---|---|---|---|---|---|
| Constant 3 | +0.000 | 0.150 | 0.250 | 0% / 100% / 0% / 0% | 0.0% / 100.0% / 0.0% / 0.0% | 0.617 | 1.000 | 0.955 |
| Rounding (the current output) | +0.000 | 0.172 | 0.256 | 2% / 97% / 4% / 0% | 1.5% / 96.4% / 2.1% / 0.0% | 0.631 | 1.007 | 0.944 |
| Random, the labeler's shares (mean of 20) | +0.001 | 0.254 | 0.254 | – | as the labels | 0.840 | 1.223 | 0.829 |
| **Cut points per labeler** | +0.103 | 0.286 | 0.287 | 44% / 44% / 18% / 9% | 40.9% / 42.1% / 12.4% / 4.6% | 0.781 | 1.115 | 0.855 |
| *Human: another expert's rating in force* | +0.138 | 0.351 | 0.350 | 44% / 60% / 27% / 9% | 35.9% / 49.9% / 11.3% / 2.9% | 0.682 | 1.041 | 0.870 |
| *Human: the other experts' mean, cut points* | +0.129 | 0.314 | 0.315 | 50% / 46% / 17% / 13% | 42.5% / 40.7% / 12.0% / 4.7% | 0.750 | 1.100 | 0.858 |

- **The current output has no agreement at all.** Rounding gives QWK 0.000, like the constant: it
  almost never produces a 2 (1.5%), a 4 (2.1%) or a 5 (none).
- **Cut points give real agreement, not just a spread.** QWK 0.103 for `base` and `agglomerative`,
  +0.10 over rounding and over random ratings with the same shares (both intervals well clear of 0).
  That is about **75% of another expert's agreement** (0.138), and macro-F1 0.286 is about 80% of
  theirs (0.351). A 5 is recognised 9% of the time, as often as the other expert manages.
- **The price is MAE.** It rises from 0.63 to 0.78, and the share within one point falls from 94% to
  86%. That is unavoidable when an output commits to 2s, 4s and 5s on a weak signal, and the other
  expert pays it too (0.682). Macro-MAE shows the same: even the other expert scores worse on it than a
  constant 3 (1.041 against 1.000). MAE and macro-MAE reward staying in the middle here and should
  not be the headline.
- **Ranking is unchanged** (the mapping keeps each labeler's order), and the arm makes little
  difference: `base`, `agglomerative` and `sync+agglomerative` are within 0.005 of each other; the
  `normalised` arms are lower (0.081–0.085).

**Per labeler** (`base`):

| Labeler | Entries | QWK: rounding | QWK: cut points (`base`) | Its shares of 2 / 3 / 4 / 5 | The cut points' shares |
|---|---|---|---|---|---|
| labeler_01 | 13,130 | −0.003 | +0.099 | 41.6% / 46.5% / 9.5% / 2.4% | 41.3% / 46.4% / 9.9% / 2.4% |
| labeler_02 | 1,208 | +0.127 | +0.281 | 33.2% / 46.8% / 19.2% / 0.8% | 31.0% / 46.7% / 21.6% / 0.7% |
| labeler_03 | 5,337 | +0.007 | +0.098 | 46.0% / 33.0% / 10.7% / 10.2% | 46.2% / 32.4% / 11.8% / 9.6% |
| labeler_04 | 1,326 | −0.024 | +0.024 | 37.5% / 42.0% / 15.6% / 4.9% | 27.9% / 33.7% / 28.3% / 10.1% |
| labeler_05 | 518 | −0.118 | −0.171 | 30.5% / 39.0% / 25.3% / 5.2% | 33.6% / 43.4% / 20.5% / 2.5% |

The cut points work for labeler_01, labeler_02 and labeler_03 (QWK 0.10–0.28, against −0.003 to 0.13
by rounding), which include the two with the most entries. labeler_04 (the third largest) barely
gains: it rated only groups 1 and 2, so with one of them held out its shares come from the other alone,
and its style differed between them (the cut points' 28% of 4s against its own 16%). labeler_05, with
518 entries, gets worse.

**Detecting "highly engaged" (≥ 4):**

| | Spearman | AUC, engaged (≥ 3) | AUC, highly engaged (≥ 4) | PR-AUC, ≥ 4 | Base rate of ≥ 4 |
|---|---|---|---|---|---|
| `base` | +0.055 | 0.518 | 0.558 | 0.194 | 0.156 |
| `normalised` | +0.045 | 0.515 | 0.543 | 0.180 | 0.156 |
| `agglomerative` | +0.065 | 0.522 | 0.569 | 0.196 | 0.156 |
| `sync+agglomerative` | +0.063 | 0.520 | 0.569 | 0.196 | 0.156 |
| `normalised+sync` | +0.048 | 0.516 | 0.548 | 0.184 | 0.156 |
| *Human: the other experts' mean* | +0.156 | 0.573 | 0.612 | 0.226 | 0.157 |

The scores detect it only weakly: precision-recall AUC 0.18–0.20 against a base rate of 0.156 (`base`:
0.194, a 1.24-fold lift), against the other experts' 0.226.

### 5.2 The self-report

| Arm | Spearman (score) | QWK: rounding → cut points | Macro-F1: rounding → cut points | Balanced accuracy: rounding → cut points | MAE: rounding → cut points |
|---|---|---|---|---|---|
| `base` | −0.130 | +0.003 → **−0.079** | 0.117 → 0.194 | 0.200 → 0.194 | 0.665 → 0.986 |
| `normalised` | −0.015 | +0.009 → **+0.044** | 0.126 → 0.201 | 0.203 → 0.201 | 0.663 → 0.915 |
| `agglomerative` | −0.075 | +0.003 → **−0.023** | 0.119 → 0.198 | 0.201 → 0.198 | 0.665 → 0.957 |
| `sync+agglomerative` | −0.043 | +0.003 → **+0.018** | 0.125 → 0.203 | 0.202 → 0.203 | 0.666 → 0.936 |
| `normalised+sync` | +0.017 | +0.012 → **+0.051** | 0.129 → 0.210 | 0.203 → 0.210 | 0.665 → 0.909 |

| Arm | Cut points − rounding: QWK | Macro-F1 | Balanced accuracy | Cut points − random with the same shares: QWK | Macro-F1 | Balanced accuracy |
|---|---|---|---|---|---|---|
| `base` | −0.080 [−0.149, −0.017] | +0.077 [+0.058, +0.100] | −0.004 [−0.025, +0.020] | −0.093 [−0.175, −0.012] | −0.005 [−0.031, +0.026] | −0.004 [−0.031, +0.030] |
| `normalised` | +0.036 [−0.019, +0.084] | +0.075 [+0.063, +0.087] | −0.001 [−0.015, +0.012] | +0.043 [−0.009, +0.091] | +0.011 [−0.015, +0.032] | +0.012 [−0.014, +0.033] |
| `agglomerative` | −0.026 [−0.091, +0.029] | +0.079 [+0.065, +0.095] | −0.002 [−0.015, +0.017] | −0.031 [−0.105, +0.041] | +0.004 [−0.014, +0.024] | +0.004 [−0.015, +0.027] |
| `sync+agglomerative` | +0.014 [−0.043, +0.067] | +0.078 [+0.065, +0.095] | +0.002 [−0.011, +0.020] | +0.038 [−0.019, +0.103] | +0.023 [+0.010, +0.037] | +0.024 [+0.010, +0.040] |
| `normalised+sync` | +0.040 [−0.010, +0.089] | +0.081 [+0.069, +0.093] | +0.007 [−0.005, +0.020] | +0.062 [−0.002, +0.117] | +0.013 [−0.006, +0.031] | +0.013 [−0.005, +0.032] |

| `base` | QWK | Macro-F1 | Balanced accuracy | Recall of 1 / 2 / 3 / 4 / 5 | Predicted shares of 1 / 2 / 3 / 4 / 5 | MAE | Macro-MAE | Within 1 point |
|---|---|---|---|---|---|---|---|---|
| Constant 3 | +0.000 | 0.117 | 0.200 | 0% / 0% / 100% / 0% / 0% | 0.0% / 0.0% / 100.0% / 0.0% / 0.0% | 0.665 | 1.200 | 0.921 |
| Rounding (the current output) | +0.003 | 0.117 | 0.200 | 0% / 0% / 100% / 0% / 0% | 0.0% / 0.1% / 99.9% / 0.0% / 0.0% | 0.665 | 1.194 | 0.921 |
| Random, the labeler's shares (mean of 20) | +0.003 | 0.198 | 0.198 | – | as the labels | 0.953 | 1.368 | 0.776 |
| **Cut points per labeler** | −0.079 | 0.194 | 0.194 | 3% / 34% / 44% / 11% / 6% | 1.3% / 36.7% / 41.3% / 14.1% / 6.7% | 0.986 | 1.490 | 0.757 |
| *Human: another expert's rating in force* | +0.197 | 0.310 | 0.310 | 0% / 43% / 70% / 27% / 16% | 0.0% / 28.1% / 53.9% / 13.9% / 4.2% | 0.690 | 1.102 | 0.852 |
| *Human: the other experts' mean, cut points* | +0.211 | 0.265 | 0.266 | 0% / 45% / 53% / 16% / 20% | 0.0% / 34.5% / 45.3% / 13.8% / 6.4% | 0.771 | 1.236 | 0.845 |

| | Spearman | AUC, engaged (≥ 3) | AUC, highly engaged (≥ 4) | PR-AUC, ≥ 4 | Base rate of ≥ 4 |
|---|---|---|---|---|---|
| `base` | −0.130 | 0.438 | 0.410 | 0.181 | 0.206 |
| `normalised` | −0.015 | 0.512 | 0.448 | 0.194 | 0.206 |
| `agglomerative` | −0.075 | 0.469 | 0.439 | 0.193 | 0.206 |
| `sync+agglomerative` | −0.043 | 0.481 | 0.464 | 0.204 | 0.206 |
| `normalised+sync` | +0.017 | 0.520 | 0.485 | 0.211 | 0.206 |
| *Human: the other experts' mean* | +0.230 | 0.618 | 0.631 | 0.322 | 0.207 |

- **The cut points cannot help where there is no signal.** No arm's QWK is clearly above random with
  the same shares (the best, `normalised+sync`, +0.051; its interval against random touches 0), and
  `base`, which ranks the self-report backwards (§3.3 of experiment 10), turns negative (−0.079).
- **The experts reach QWK 0.20–0.21 on the self-report** and detect a self-reported 4 or 5 at
  precision-recall AUC 0.32 against a base rate of 0.21; every model is at its base rate.

### 5.3 Can the cut points do better?

*After §5.1–5.2: `python -m scripts.analysis.imbalance_variants` (results `variants.json`). Each
variant was fixed before it was run, and all are scored on the same entries: those that every source
predicted (21,285 expert entries and 2,380 self-report entries, the labeler-sessions that experiment 10
Appendix D keeps). The reference is §5.1's mapping on these entries (QWK 0.104, against 0.103 on all
21,519).*

Three ideas, each a way to get more from the same scores:
1. **Other target shares.** With a weak signal, predicting the rare classes more often than they occur
   can raise macro-F1 and balanced accuracy. Cut points at **tempered** shares (halfway between the
   labeler's own and equal shares) and at **equal** shares.
2. **One pooled model.** The models are one per labeler, and labeler_04 and labeler_05 have few
   entries (1,326 and 518). The pooled ridge of experiment 10 Appendix D (W0p: one model for every
   expert, normalised inputs, the rating as target), against the per-labeler model on the same inputs
   (`normalised`), so that only pooling differs.
3. **Averaging models.** The mean percentile rank of `base`, `agglomerative` and `sync+agglomerative`
   (ranked within each labeler and held-out group, so the models' scales do not matter).

**Experts' labels:**

| Variant | QWK | Macro-F1 | Balanced accuracy | Recall per class | MAE | QWK against the reference (95%) |
|---|---|---|---|---|---|---|
| **Reference:** `base`, own shares (§5.1) | +0.104 | 0.287 | 0.288 | 44% / 44% / 18% / 9% | 0.778 | – |
| 1a. `base`, tempered shares | +0.099 | 0.264 | 0.287 | 36% / 35% / 20% / 23% | 0.964 | −0.005 [−0.013, +0.004] |
| 1b. `base`, equal shares | +0.077 | 0.234 | 0.285 | 28% / 26% / 25% / 35% | 1.165 | −0.027 [−0.040, −0.012] |
| 2a. `normalised`, one model per labeler, own shares | +0.086 | 0.281 | 0.281 | 44% / 44% / 14% / 10% | 0.790 | −0.018 [−0.042, +0.006] |
| 2b. pooled ridge, one model for all, own shares | +0.074 | 0.283 | 0.283 | 44% / 45% / 14% / 10% | 0.792 | −0.030 [−0.055, −0.004] |
| 3. average of three models, own shares | +0.105 | 0.286 | 0.287 | 44% / 44% / 18% / 9% | 0.779 | +0.001 [−0.008, +0.010] |

**QWK per labeler:**

| Variant | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 |
|---|---|---|---|---|---|
| **Reference:** `base`, own shares (§5.1) | +0.099 | +0.270 | +0.096 | +0.039 | −0.124 |
| 1a. `base`, tempered shares | +0.098 | +0.275 | +0.088 | +0.057 | −0.112 |
| 1b. `base`, equal shares | +0.072 | +0.229 | +0.074 | +0.068 | −0.080 |
| 2a. `normalised`, one model per labeler, own shares | +0.062 | +0.211 | +0.106 | −0.023 | +0.037 |
| 2b. pooled ridge, one model for all, own shares | +0.064 | +0.035 | +0.082 | +0.021 | +0.022 |
| 3. average of three models, own shares | +0.099 | +0.264 | +0.093 | +0.042 | −0.044 |

**Self-report:**

| Variant | QWK | Macro-F1 | Balanced accuracy | Recall per class | MAE | QWK against the reference (95%) |
|---|---|---|---|---|---|---|
| **Reference:** `base`, own shares (§5.1) | −0.083 | 0.192 | 0.192 | 3% / 33% / 44% / 11% / 6% | 0.995 | – |
| 1a. `base`, tempered shares | −0.068 | 0.178 | 0.178 | 3% / 27% / 33% / 12% / 14% | 1.199 | +0.016 [−0.004, +0.035] |
| 1b. `base`, equal shares | −0.060 | 0.164 | 0.187 | 13% / 19% / 22% / 19% / 20% | 1.401 | +0.023 [+0.006, +0.041] |
| 2a. `normalised`, one model per labeler, own shares | +0.047 | 0.202 | 0.202 | 0% / 40% / 45% / 12% / 4% | 0.918 | +0.131 [+0.076, +0.191] |
| 2b. pooled ridge, one model for all, own shares | +0.041 | 0.203 | 0.203 | 0% / 39% / 44% / 13% / 5% | 0.920 | +0.125 [+0.065, +0.196] |
| 3. average of three models, own shares | −0.034 | 0.189 | 0.189 | 3% / 35% / 42% / 9% / 5% | 0.974 | +0.050 [+0.009, +0.090] |

- **Predicting the rare classes more often does not pay.** With tempered or equal shares a 5 is
  predicted more often (recall 23% and 35%, against 9%), but most of the extra 5s are wrong: QWK and
  macro-F1 fall, balanced accuracy stays the same, and MAE rises from 0.78 to 0.96 and 1.17.
- **Pooling does not help.** The pooled model is lower (QWK 0.074, −0.030). It rescues the smallest
  labeler (labeler_05: −0.124 → +0.022) and loses labeler_02 (0.270 → 0.035). The per-labeler model on
  the same normalised inputs is between them (0.086). That `base`, on raw inputs, is ahead of both fits
  experiment 10's seat check: raw inputs vary with where a student sits, and the main labelers' ratings
  do slightly too. It is not a gain to build on.
- **Averaging changes nothing** (QWK 0.105 against 0.104): the three models make almost the same errors.
  Its Spearman (+0.107, against 0.055) is not a gain either: ranking within each labeler and group
  removes the level differences that a Spearman pooled over labelers and groups otherwise includes;
  within each labeler the order is that of the three models.
- **On the self-report every variant stays near chance.** The normalised arms (QWK +0.047 and +0.041)
  are ahead only of `base`, which ranks the self-report backwards (−0.083).

So §5.1's mapping is the best 1–5 output these predictions give. What is left needs other training
(an ordinal model, a consensus target) or new information, not another way to read the same scores.

## 6. Conclusion

- **The imbalance is not what limits the models,** as 09 found: it sits in the tails (4s and 5s are
  16% of expert entries, 1s absent), the engaged split is close to balanced, and balancing the
  training data spreads the predictions without making them better. The limit is the weak signal.
- **But it hid that the output never used the scale.** Rounded, the models answer 3 for 96% of
  entries, with no agreement (QWK 0.000), however good their ranking.
- **Cut points per labeler fix the output without touching the model.** On the experts' labels, the
  same scores become ratings with each labeler's shares and real agreement: QWK 0.10 against 0.00, about
  75% of another expert's; macro-F1 0.29 against 0.17; every class predicted, a 5 recognised as often as
  by a human. On the self-report there is no signal for them to map.
- **Nothing else read from the same scores does better** (§5.3): tempered or equal shares, a pooled
  model, or averaging three models all tie or lose. The cut points at each labeler's own shares are the
  best output these predictions give; more would need other training or new information.
- **Metrics:** report QWK, macro-F1, balanced accuracy and per-class recall next to Spearman and AUC,
  and the human reference on the same entries. Do not headline MAE or macro-MAE: they favour a
  constant 3, even over a human.
- **Not recommended:** resampling or filtering the training data (09: it costs ranking), synthetic
  oversampling (it would invent rows from noise-dominated features), class weights on the regression.
- **For a deployed 1–5 output:** cut points per labeler on the model's score. For "highly engaged"
  alone, the scores are too weak to be useful (a 1.24-fold lift).

**Not run:** the activity split, an ordinal (cumulative-logit) model, training on the experts'
consensus, and a detector trained for ≥ 4 with class weights.

## Appendix A: reproducing

```
python -m scripts.analysis.imbalance_scoring        # §5.1-5.2, ~8 min; reads experiment 10's saved predictions
python -m scripts.analysis.imbalance_variants       # §5.3, ~3 min; also reads Appendix D's saved predictions
```

Options: `--predictions`, `--arms`, `--boot`, `--out` (and `--within` for the variants). Tests, with
known answers (the mappings, the share rules, the leakage guard, the metrics against hand and library
values): `tests/test_imbalance_scoring.py`, `tests/test_imbalance_variants.py`.

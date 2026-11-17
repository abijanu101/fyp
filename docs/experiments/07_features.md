# Features: what the model can and cannot read

*Started 18 November 2026. §1.1 is the feature vector as experiment 06 trained on it; §1.2 tries
changes to it, one trial at a time. Base config: [`07_features.json`](07_features.json), which is
experiment 06's config ([`06_full_pipeline_and_labels.json`](06_full_pipeline_and_labels.json))
with the entered rating as the training target (no interpolation) and the new `expert.pca` key off.*

**Why this experiment.** Experiment 06 ended with a model that underfits with every classical
decoder (training AUC 0.55–0.59). The decoder, the training target, the window length and
augmentation were all varied there, and none moved the result. What is left is the input.

---

## 1. The vector

Script: `python -m scripts.analysis.feature_study` (`scripts/analysis/feature_study.py`). Output in
`outputs/experiments/07_features/`: `vector_stats.csv` (per-column distributions, redundancy,
variance shares), `correlations.csv`, `components.csv`, `feature_correlations.png`, `deciles.png`.
Rows: the 45,166 rows experiment 06 trains on, original recordings only.

### 1.1 Layout

Each row is one student in a 4 s window centred on a rating entry. The per-labeler model (an
expert) receives 95 numbers; the activity gate, a separate model, receives 26 more.

| Block | Columns | Family | What each column is | Source |
|---|---|---|---|---|
| `trk.*`: the student | 10 | motion (Kalman) | box-centre velocity, speed, acceleration, energy, share of still frames, sign flips per second, trends of area, aspect and height; divided by the box diagonal | the tracker's Kalman state, about 5 fps |
| | 9 | flow (residual) | optical-flow magnitude in a 3 × 3 grid inside the box, with the box's own mean flow removed; window mean | Farneback flow on the frame at half resolution (640 × 360) |
| | 8 | behaviour (SCB) | hand raising, reading, writing, and objectness scores from the SCB YOLOv7 detectors, mean and peak over the window | detector at 640 px |
| `vec.*`: the class | 54 | the same 27 | mean and SD across the visible students | from the `trk.*` values |
| | 4 | thresholds | share of students still, hand up, reading, writing | |
| | 9 | occupancy | share of students in each cell of a 3 × 3 grid over the frame | box centres |
| | 1 | count | `n_tracks`, the number of visible students | |
| **Model input** | **95** | | | |
| `scene.*`: the gate | 26 | scene | whole-frame motion, saturation, brightness, edges, a 3 × 3 brightness grid; mean and SD over the window | frame at 64 × 36 |

**The features themselves look fine for what these labels allow.** Every column's correlation with
every labeler is in the Appendix. The strongest per-student signal, optical flow inside the box,
tracks the experts' ratings at Spearman ρ ≈ 0.07–0.08, within students as well as across them.
That is about what one expert achieves predicting another (0.083, experiment 06 Appendix A).
labeler_02 responds to different columns (hand raising, head lowering) than the others. No column
tracks the self-report (|ρ| ≤ 0.05).

### 1.2 Trials

Each trial changes one thing on [`07_features.json`](07_features.json), with logistic regression
experts unless stated, and is scored like experiment 06 (held-out and training groups, per labeler,
the self-report track alongside).

#### 1.2.1 PCA and dimensionality reduction

The 95 inputs are highly redundant: 61 have a partner at |ρ| > 0.9, and 29 principal components
hold 90% of the variance (`vector_stats.csv`). The trial adds PCA inside each expert's pipeline
(impute → standardise → **PCA** → classifier, `expert.pca`), fitted with the rest of the expert on
each fold's training rows only. An integer keeps that many components; a fraction keeps enough
for that share of the variance. Each setting is tried with the class context (`vec.*`, 68 columns)
and without it (`engagement.late_context=false`, the 27 per-student columns only).

**What the components are** (fitted on all 95 standardised inputs, label-free; `components.csv`;
ρ within student):

| Component | Variance | Mostly | ρ with experts | ρ with self-report |
|---|---|---|---|---|
| PC1 | 25.5% | class flow (`vec.rf.*.mean`) | +0.036 | +0.007 |
| PC2 | 11.2% | class reading (`vec.scb.read`, objectness) | −0.010 | +0.004 |
| PC3 | 7.5% | **the student's own flow** (`trk.rf.*`) | **+0.053** | +0.017 |
| PC4 | 5.0% | class hand raising | +0.024 | +0.007 |
| PC5 | 4.5% | occupancy (seat layout) | −0.018 | −0.009 |
| PC6 | 3.4% | the student's motion (stillness against speed) | +0.004 | +0.000 |
| PC7–PC12 | 1.9–3.2% each | class writing, head lowering, `n_tracks` and events, the student's SCB | ≤ 0.031 | ≤ 0.041 |

The component that carries the experts' signal, the student's own flow, is the third by variance.
The first two are class-level and carry little. Keeping only the top few components by variance
would keep it; a cut below three would lose it.

| Arm | Setting | Components kept | Results |
|---|---|---|---|
| base | – | all 95 inputs | *running* |
| PCA 5 / 10 / 20 | `expert.pca=5` / `10` / `20` | 5 / 10 / 20 | *running* |
| PCA 90% / 95% | `expert.pca=0.9` / `0.95` | enough for 90% / 95% of the variance | *running* |
| no context | `engagement.late_context=false` | the 27 per-student inputs | *running* |
| no context + PCA 5 / 10 / 90% | both | | *running* |

Results to follow in the standard tables (`python -m scripts.reporting.results_table`).

---

## Appendix: every column against every labeler

Spearman ρ of each model input with each labeler's ratings, all experts pooled, the self-report, and
the model's own-labeler prediction (in-sample, experts' rows). |ρ| ≥ 0.05 in bold. One table per
feature category, first within student, then raw. Source:
`outputs/experiments/07_features/correlations.csv`; §1.5 explains both views.

### Within student (feature and rating demeaned per student-session and labeler)

**Student: motion (Kalman)** (10 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `trk.kg.accel` | **+0.054** | +0.029 | +0.011 | +0.016 | +0.036 | +0.037 | +0.011 | **+0.223** |
| `trk.kg.area_trend` | +0.011 | +0.002 | +0.030 | −0.007 | +0.015 | +0.016 | −0.026 | −0.036 |
| `trk.kg.aspect_trend` | −0.027 | **−0.155** | **−0.060** | −0.045 | +0.031 | −0.043 | −0.021 | −0.018 |
| `trk.kg.cy_trend` | −0.016 | **−0.177** | **−0.066** | **−0.069** | +0.036 | −0.041 | −0.022 | +0.029 |
| `trk.kg.energy` | **+0.063** | +0.027 | +0.006 | +0.039 | +0.040 | +0.042 | +0.017 | **+0.268** |
| `trk.kg.events` | −0.023 | −0.017 | −0.002 | **−0.051** | −0.011 | −0.017 | −0.014 | **−0.109** |
| `trk.kg.speed` | **+0.067** | +0.034 | +0.003 | +0.040 | +0.042 | +0.043 | +0.019 | **+0.278** |
| `trk.kg.stillness` | −0.036 | +0.012 | −0.019 | −0.003 | −0.002 | −0.022 | −0.004 | **−0.069** |
| `trk.kg.vx` | +0.002 | **+0.063** | +0.008 | **+0.076** | +0.011 | +0.012 | +0.002 | **+0.056** |
| `trk.kg.vy` | −0.024 | **−0.181** | **−0.060** | **−0.089** | +0.049 | −0.045 | −0.013 | −0.001 |

**Student: flow (residual)** (9 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `trk.rf.mag_00` | **+0.084** | +0.000 | +0.045 | +0.049 | **+0.052** | **+0.065** | +0.013 | **+0.279** |
| `trk.rf.mag_01` | **+0.070** | −0.024 | +0.029 | −0.017 | +0.010 | +0.047 | −0.007 | **+0.207** |
| `trk.rf.mag_02` | **+0.077** | +0.019 | +0.039 | +0.019 | +0.048 | **+0.058** | +0.019 | **+0.250** |
| `trk.rf.mag_10` | **+0.092** | **+0.055** | +0.042 | **+0.063** | **+0.055** | **+0.073** | +0.018 | **+0.294** |
| `trk.rf.mag_11` | **+0.093** | +0.044 | +0.045 | +0.030 | +0.043 | **+0.072** | +0.006 | **+0.306** |
| `trk.rf.mag_12` | **+0.085** | +0.042 | +0.038 | +0.040 | **+0.056** | **+0.065** | +0.014 | **+0.280** |
| `trk.rf.mag_20` | **+0.078** | **+0.061** | +0.034 | **+0.072** | +0.040 | **+0.062** | +0.012 | **+0.281** |
| `trk.rf.mag_21` | **+0.081** | **+0.074** | +0.042 | **+0.073** | **+0.055** | **+0.068** | +0.015 | **+0.309** |
| `trk.rf.mag_22` | **+0.072** | +0.046 | +0.031 | **+0.051** | +0.048 | **+0.057** | +0.011 | **+0.254** |

**Student: behaviour (SCB)** (8 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `trk.scb.hand_raising` | +0.034 | **+0.076** | +0.023 | +0.047 | **+0.103** | +0.033 | +0.018 | **+0.128** |
| `trk.scb.hand_raising.max` | +0.040 | **+0.073** | +0.033 | +0.049 | **+0.088** | +0.040 | +0.009 | **+0.144** |
| `trk.scb.m0.obj` | +0.020 | **+0.113** | −0.001 | −0.001 | +0.022 | +0.016 | +0.007 | **+0.124** |
| `trk.scb.m0.obj.max` | +0.031 | **+0.109** | +0.001 | +0.027 | +0.047 | +0.027 | +0.007 | **+0.178** |
| `trk.scb.read` | +0.012 | **+0.098** | −0.008 | −0.020 | −0.017 | +0.007 | −0.000 | **+0.082** |
| `trk.scb.read.max` | +0.022 | **+0.095** | −0.008 | +0.009 | +0.018 | +0.016 | −0.002 | **+0.121** |
| `trk.scb.write` | +0.009 | **+0.094** | +0.002 | −0.024 | +0.016 | +0.007 | +0.000 | **+0.119** |
| `trk.scb.write.max` | +0.022 | **+0.082** | +0.020 | −0.017 | +0.042 | +0.020 | +0.010 | **+0.126** |

**Class: motion (Kalman)** (21 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `vec.kg.accel.mean` | +0.029 | **+0.062** | −0.001 | +0.038 | +0.039 | +0.022 | +0.005 | **+0.142** |
| `vec.kg.accel.std` | +0.018 | **+0.052** | +0.003 | +0.011 | +0.049 | +0.015 | −0.007 | +0.047 |
| `vec.kg.area_trend.mean` | −0.003 | **−0.054** | −0.006 | −0.028 | −0.041 | −0.009 | −0.037 | −0.029 |
| `vec.kg.area_trend.std` | +0.031 | +0.043 | +0.001 | +0.048 | +0.008 | +0.023 | +0.022 | **+0.135** |
| `vec.kg.aspect_trend.mean` | −0.016 | −0.050 | −0.014 | −0.039 | +0.009 | −0.017 | +0.019 | −0.018 |
| `vec.kg.aspect_trend.std` | +0.027 | +0.021 | +0.000 | +0.044 | −0.020 | +0.019 | +0.001 | **+0.140** |
| `vec.kg.cy_trend.mean` | −0.015 | **−0.112** | −0.031 | **−0.066** | +0.001 | −0.027 | −0.042 | −0.007 |
| `vec.kg.cy_trend.std` | +0.021 | +0.028 | −0.007 | +0.044 | **+0.050** | +0.016 | +0.019 | **+0.098** |
| `vec.kg.energy.mean` | +0.047 | **+0.065** | +0.001 | **+0.059** | +0.047 | +0.034 | +0.016 | **+0.155** |
| `vec.kg.energy.std` | +0.039 | **+0.054** | +0.014 | +0.024 | +0.028 | +0.031 | −0.003 | +0.032 |
| `vec.kg.events.mean` | −0.016 | −0.021 | −0.004 | −0.041 | −0.003 | −0.013 | −0.035 | **−0.163** |
| `vec.kg.events.std` | +0.002 | −0.025 | −0.008 | −0.012 | −0.002 | −0.003 | −0.003 | **−0.073** |
| `vec.kg.speed.mean` | **+0.051** | **+0.065** | +0.000 | **+0.059** | +0.047 | +0.036 | +0.018 | **+0.157** |
| `vec.kg.speed.std` | +0.043 | **+0.052** | +0.015 | +0.027 | +0.025 | +0.034 | +0.001 | +0.036 |
| `vec.kg.stillness.frac` | −0.019 | **−0.056** | −0.002 | −0.028 | +0.017 | −0.013 | −0.010 | −0.039 |
| `vec.kg.stillness.mean` | −0.022 | −0.046 | +0.005 | −0.025 | −0.016 | −0.014 | −0.012 | −0.043 |
| `vec.kg.stillness.std` | +0.019 | **+0.050** | −0.007 | +0.016 | +0.019 | +0.011 | +0.008 | +0.021 |
| `vec.kg.vx.mean` | −0.018 | −0.004 | +0.005 | **+0.053** | −0.036 | −0.006 | −0.002 | +0.004 |
| `vec.kg.vx.std` | **+0.053** | **+0.053** | +0.024 | **+0.054** | +0.028 | +0.043 | +0.000 | **+0.060** |
| `vec.kg.vy.mean` | −0.004 | **−0.130** | −0.015 | **−0.051** | +0.027 | −0.015 | −0.022 | +0.008 |
| `vec.kg.vy.std` | +0.022 | **+0.050** | −0.012 | +0.018 | +0.005 | +0.013 | +0.000 | **+0.073** |

**Class: flow (residual)** (18 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `vec.rf.mag_00.mean` | +0.040 | +0.038 | −0.001 | +0.035 | **+0.064** | +0.028 | +0.013 | **+0.191** |
| `vec.rf.mag_00.std` | +0.028 | +0.017 | −0.003 | +0.007 | **+0.062** | +0.017 | −0.004 | **+0.085** |
| `vec.rf.mag_01.mean` | +0.034 | +0.020 | −0.005 | +0.013 | +0.018 | +0.020 | +0.006 | **+0.175** |
| `vec.rf.mag_01.std` | +0.023 | +0.014 | +0.005 | −0.016 | +0.049 | +0.015 | −0.014 | **+0.091** |
| `vec.rf.mag_02.mean` | +0.039 | **+0.052** | +0.001 | +0.025 | **+0.060** | +0.028 | +0.012 | **+0.201** |
| `vec.rf.mag_02.std` | +0.023 | +0.024 | +0.000 | +0.006 | **+0.058** | +0.016 | −0.006 | **+0.100** |
| `vec.rf.mag_10.mean` | +0.043 | **+0.059** | −0.003 | +0.039 | **+0.058** | +0.030 | +0.022 | **+0.223** |
| `vec.rf.mag_10.std` | +0.029 | +0.032 | −0.002 | +0.006 | **+0.056** | +0.019 | +0.003 | **+0.114** |
| `vec.rf.mag_11.mean` | +0.043 | +0.047 | −0.001 | +0.033 | **+0.050** | +0.030 | +0.010 | **+0.253** |
| `vec.rf.mag_11.std` | +0.028 | +0.031 | +0.001 | −0.010 | **+0.062** | +0.018 | −0.020 | **+0.149** |
| `vec.rf.mag_12.mean` | +0.041 | **+0.063** | +0.000 | +0.040 | **+0.081** | +0.030 | +0.005 | **+0.241** |
| `vec.rf.mag_12.std` | +0.025 | +0.038 | +0.003 | +0.002 | **+0.081** | +0.019 | −0.019 | **+0.137** |
| `vec.rf.mag_20.mean` | +0.039 | **+0.055** | −0.004 | +0.046 | **+0.080** | +0.028 | +0.012 | **+0.208** |
| `vec.rf.mag_20.std` | +0.028 | +0.046 | −0.004 | +0.007 | **+0.104** | +0.019 | +0.003 | **+0.098** |
| `vec.rf.mag_21.mean` | +0.035 | **+0.051** | −0.004 | +0.036 | **+0.085** | +0.026 | +0.009 | **+0.245** |
| `vec.rf.mag_21.std` | +0.021 | +0.043 | −0.007 | +0.001 | **+0.086** | +0.014 | −0.007 | **+0.129** |
| `vec.rf.mag_22.mean` | +0.033 | +0.045 | +0.002 | +0.033 | **+0.093** | +0.025 | +0.011 | **+0.200** |
| `vec.rf.mag_22.std` | +0.019 | +0.043 | +0.001 | +0.002 | **+0.088** | +0.015 | −0.006 | **+0.090** |

**Class: behaviour (SCB)** (19 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `vec.scb.hand_raising.frac` | +0.037 | **+0.092** | +0.016 | **+0.084** | **+0.093** | +0.038 | +0.012 | **+0.106** |
| `vec.scb.hand_raising.max.mean` | +0.040 | **+0.097** | +0.021 | **+0.062** | **+0.099** | +0.040 | +0.005 | **+0.174** |
| `vec.scb.hand_raising.max.std` | +0.031 | **+0.077** | +0.014 | +0.047 | **+0.092** | +0.031 | +0.008 | **+0.145** |
| `vec.scb.hand_raising.mean` | +0.045 | **+0.082** | +0.024 | **+0.075** | **+0.109** | +0.045 | +0.004 | **+0.145** |
| `vec.scb.hand_raising.std` | +0.039 | **+0.078** | +0.021 | **+0.075** | **+0.101** | +0.040 | +0.006 | **+0.123** |
| `vec.scb.m0.obj.max.mean` | +0.027 | **+0.076** | −0.003 | +0.037 | **+0.136** | +0.024 | +0.011 | **+0.150** |
| `vec.scb.m0.obj.max.std` | +0.017 | **+0.071** | −0.003 | +0.043 | **+0.137** | +0.018 | +0.023 | **+0.141** |
| `vec.scb.m0.obj.mean` | +0.019 | **+0.052** | +0.002 | +0.022 | **+0.132** | +0.019 | +0.002 | **+0.082** |
| `vec.scb.m0.obj.std` | +0.013 | **+0.063** | +0.002 | +0.037 | **+0.129** | +0.017 | +0.006 | **+0.059** |
| `vec.scb.read.frac` | +0.007 | −0.008 | +0.005 | −0.002 | **+0.075** | +0.007 | −0.010 | −0.047 |
| `vec.scb.read.max.mean` | +0.019 | +0.041 | −0.009 | +0.021 | **+0.115** | +0.015 | +0.006 | **+0.104** |
| `vec.scb.read.max.std` | +0.007 | +0.049 | −0.010 | +0.029 | **+0.126** | +0.009 | +0.012 | **+0.086** |
| `vec.scb.read.mean` | +0.011 | +0.008 | −0.005 | +0.001 | **+0.102** | +0.008 | −0.002 | +0.037 |
| `vec.scb.read.std` | −0.001 | +0.014 | −0.007 | +0.011 | **+0.100** | +0.001 | −0.002 | −0.025 |
| `vec.scb.write.frac` | −0.038 | +0.008 | +0.011 | −0.049 | −0.014 | −0.027 | −0.012 | −0.042 |
| `vec.scb.write.max.mean` | +0.008 | +0.033 | +0.009 | −0.015 | **+0.084** | +0.009 | −0.021 | +0.025 |
| `vec.scb.write.max.std` | +0.007 | +0.027 | +0.008 | −0.010 | **+0.063** | +0.007 | −0.025 | −0.002 |
| `vec.scb.write.mean` | −0.006 | +0.016 | +0.002 | −0.031 | **+0.079** | −0.003 | −0.022 | −0.010 |
| `vec.scb.write.std` | −0.012 | +0.015 | +0.001 | −0.036 | **+0.067** | −0.008 | −0.024 | **−0.051** |

**Class: occupancy and count** (10 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `vec.n_tracks` | −0.005 | −0.009 | +0.000 | −0.026 | +0.019 | −0.004 | +0.002 | **−0.207** |
| `vec.occ_00` | +0.009 | −0.002 | +0.008 | +0.013 | −0.010 | +0.005 | +0.008 | **+0.088** |
| `vec.occ_01` | +0.004 | −0.014 | +0.004 | −0.003 | +0.014 | +0.002 | +0.011 | **+0.065** |
| `vec.occ_02` | +0.004 | +0.007 | −0.005 | −0.015 | −0.001 | +0.000 | +0.003 | −0.024 |
| `vec.occ_10` | −0.003 | −0.014 | +0.004 | −0.003 | +0.016 | −0.001 | +0.001 | −0.029 |
| `vec.occ_11` | −0.001 | +0.010 | +0.004 | +0.013 | −0.007 | +0.002 | −0.008 | −0.021 |
| `vec.occ_12` | −0.000 | −0.006 | +0.002 | +0.012 | −0.005 | +0.001 | −0.010 | +0.010 |
| `vec.occ_20` | +0.001 | +0.004 | +0.019 | +0.012 | −0.028 | +0.007 | +0.004 | **−0.134** |
| `vec.occ_21` | +0.005 | −0.004 | −0.002 | −0.039 | +0.037 | +0.003 | −0.015 | **−0.102** |
| `vec.occ_22` | −0.024 | −0.002 | +0.003 | −0.018 | +0.021 | −0.010 | −0.014 | **−0.104** |

### Raw

**Student: motion (Kalman)** (10 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `trk.kg.accel` | **+0.083** | −0.037 | +0.036 | −0.004 | +0.041 | **+0.064** | +0.020 | **+0.238** |
| `trk.kg.area_trend` | +0.013 | +0.005 | +0.039 | −0.000 | +0.042 | +0.020 | −0.020 | −0.030 |
| `trk.kg.aspect_trend` | −0.030 | **−0.100** | **−0.066** | −0.040 | **+0.059** | −0.043 | −0.020 | −0.017 |
| `trk.kg.cy_trend` | −0.019 | **−0.138** | **−0.069** | **−0.070** | **+0.076** | −0.042 | −0.035 | +0.019 |
| `trk.kg.energy` | **+0.094** | **−0.052** | +0.035 | +0.018 | +0.047 | **+0.073** | +0.029 | **+0.267** |
| `trk.kg.events` | −0.038 | +0.006 | −0.008 | **−0.060** | −0.028 | −0.032 | −0.032 | **−0.100** |
| `trk.kg.speed` | **+0.099** | −0.047 | +0.034 | +0.018 | **+0.051** | **+0.076** | +0.033 | **+0.274** |
| `trk.kg.stillness` | −0.040 | **+0.082** | −0.005 | +0.000 | −0.006 | −0.023 | +0.000 | **−0.076** |
| `trk.kg.vx` | +0.003 | +0.043 | +0.010 | **+0.051** | +0.012 | +0.009 | +0.000 | +0.036 |
| `trk.kg.vy` | −0.033 | **−0.180** | **−0.054** | **−0.086** | **+0.057** | −0.049 | −0.020 | −0.008 |

**Student: flow (residual)** (9 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `trk.rf.mag_00` | **+0.081** | **−0.065** | **+0.058** | +0.015 | +0.001 | **+0.070** | −0.007 | **+0.246** |
| `trk.rf.mag_01` | **+0.077** | **−0.110** | +0.042 | −0.035 | −0.032 | **+0.058** | −0.012 | **+0.194** |
| `trk.rf.mag_02` | **+0.079** | **−0.060** | **+0.052** | +0.026 | +0.017 | **+0.068** | −0.008 | **+0.226** |
| `trk.rf.mag_10` | **+0.088** | −0.002 | **+0.054** | +0.028 | +0.029 | **+0.076** | −0.004 | **+0.264** |
| `trk.rf.mag_11` | **+0.096** | −0.020 | **+0.057** | +0.000 | +0.024 | **+0.080** | −0.008 | **+0.280** |
| `trk.rf.mag_12` | **+0.083** | −0.032 | +0.045 | +0.007 | +0.030 | **+0.069** | −0.006 | **+0.251** |
| `trk.rf.mag_20` | **+0.083** | +0.021 | **+0.059** | +0.034 | +0.033 | **+0.075** | +0.001 | **+0.264** |
| `trk.rf.mag_21` | **+0.079** | +0.042 | **+0.064** | +0.025 | +0.014 | **+0.074** | +0.006 | **+0.283** |
| `trk.rf.mag_22` | **+0.079** | +0.005 | **+0.050** | +0.001 | +0.018 | **+0.068** | +0.003 | **+0.243** |

**Student: behaviour (SCB)** (8 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `trk.scb.hand_raising` | +0.035 | **+0.164** | +0.010 | +0.004 | +0.049 | +0.031 | +0.021 | **+0.155** |
| `trk.scb.hand_raising.max` | +0.043 | **+0.174** | +0.017 | +0.006 | +0.037 | +0.040 | +0.021 | **+0.175** |
| `trk.scb.m0.obj` | +0.045 | **+0.109** | +0.010 | +0.016 | −0.003 | +0.031 | +0.002 | **+0.101** |
| `trk.scb.m0.obj.max` | **+0.054** | **+0.106** | +0.019 | +0.018 | −0.015 | +0.041 | −0.001 | **+0.121** |
| `trk.scb.read` | +0.042 | **+0.077** | +0.004 | +0.012 | −0.027 | +0.026 | −0.005 | **+0.076** |
| `trk.scb.read.max` | +0.047 | **+0.068** | +0.010 | +0.015 | −0.039 | +0.031 | −0.008 | **+0.085** |
| `trk.scb.write` | **+0.063** | **+0.070** | +0.015 | −0.005 | **+0.050** | +0.042 | +0.012 | **+0.132** |
| `trk.scb.write.max` | **+0.074** | **+0.053** | +0.022 | −0.013 | **+0.071** | **+0.052** | +0.012 | **+0.157** |

**Class: motion (Kalman)** (21 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `vec.kg.accel.mean` | **+0.056** | +0.040 | +0.027 | +0.037 | **+0.076** | **+0.052** | +0.031 | **+0.153** |
| `vec.kg.accel.std` | +0.026 | +0.024 | +0.014 | +0.014 | **+0.051** | +0.025 | +0.010 | **+0.054** |
| `vec.kg.area_trend.mean` | −0.002 | **−0.064** | −0.001 | −0.008 | −0.019 | −0.005 | −0.032 | −0.013 |
| `vec.kg.area_trend.std` | +0.036 | +0.039 | +0.021 | +0.033 | +0.043 | +0.037 | +0.034 | **+0.120** |
| `vec.kg.aspect_trend.mean` | −0.019 | −0.030 | −0.014 | −0.048 | **+0.052** | −0.018 | +0.012 | −0.016 |
| `vec.kg.aspect_trend.std` | +0.037 | −0.009 | +0.020 | +0.030 | +0.017 | +0.033 | +0.028 | **+0.117** |
| `vec.kg.cy_trend.mean` | −0.013 | **−0.099** | −0.028 | **−0.067** | −0.010 | −0.025 | −0.035 | +0.003 |
| `vec.kg.cy_trend.std` | +0.030 | −0.020 | +0.019 | +0.014 | +0.050 | +0.029 | +0.025 | **+0.077** |
| `vec.kg.energy.mean` | **+0.062** | **+0.058** | +0.028 | **+0.066** | **+0.074** | **+0.058** | +0.044 | **+0.176** |
| `vec.kg.energy.std` | +0.034 | +0.015 | +0.010 | +0.024 | +0.027 | +0.029 | +0.018 | **+0.062** |
| `vec.kg.events.mean` | −0.027 | −0.035 | −0.035 | **−0.081** | +0.010 | −0.037 | **−0.054** | **−0.163** |
| `vec.kg.events.std` | −0.001 | **−0.051** | −0.028 | −0.038 | +0.005 | −0.016 | −0.015 | **−0.098** |
| `vec.kg.speed.mean` | **+0.064** | **+0.055** | +0.027 | **+0.066** | **+0.075** | **+0.060** | +0.048 | **+0.183** |
| `vec.kg.speed.std` | +0.039 | +0.011 | +0.008 | +0.026 | +0.022 | +0.031 | +0.025 | **+0.073** |
| `vec.kg.stillness.frac` | −0.017 | −0.015 | −0.003 | −0.015 | −0.008 | −0.015 | −0.004 | −0.042 |
| `vec.kg.stillness.mean` | −0.020 | −0.025 | −0.005 | −0.021 | −0.021 | −0.020 | −0.004 | −0.042 |
| `vec.kg.stillness.std` | +0.017 | +0.028 | +0.005 | +0.021 | +0.016 | +0.017 | +0.002 | +0.028 |
| `vec.kg.vx.mean` | −0.007 | +0.021 | +0.014 | +0.045 | +0.030 | +0.004 | −0.017 | +0.013 |
| `vec.kg.vx.std` | **+0.056** | **+0.082** | +0.025 | **+0.099** | +0.034 | **+0.055** | +0.047 | **+0.158** |
| `vec.kg.vy.mean` | −0.006 | **−0.148** | −0.001 | **−0.068** | +0.044 | −0.014 | −0.029 | −0.000 |
| `vec.kg.vy.std` | +0.035 | −0.019 | +0.008 | +0.000 | +0.032 | +0.028 | +0.019 | **+0.062** |

**Class: flow (residual)** (18 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `vec.rf.mag_00.mean` | +0.048 | +0.047 | +0.037 | **+0.089** | **+0.072** | **+0.055** | +0.042 | **+0.199** |
| `vec.rf.mag_00.std` | +0.035 | +0.029 | +0.021 | **+0.061** | +0.047 | +0.039 | +0.019 | **+0.135** |
| `vec.rf.mag_01.mean` | +0.046 | +0.035 | +0.034 | **+0.057** | +0.041 | **+0.051** | +0.036 | **+0.189** |
| `vec.rf.mag_01.std` | +0.030 | −0.001 | +0.025 | +0.014 | +0.047 | +0.033 | +0.019 | **+0.122** |
| `vec.rf.mag_02.mean` | **+0.051** | +0.046 | +0.036 | **+0.061** | **+0.059** | **+0.055** | +0.039 | **+0.196** |
| `vec.rf.mag_02.std` | +0.040 | +0.018 | +0.023 | +0.047 | **+0.051** | +0.041 | +0.020 | **+0.138** |
| `vec.rf.mag_10.mean` | **+0.053** | **+0.091** | +0.036 | **+0.087** | **+0.076** | **+0.060** | +0.046 | **+0.227** |
| `vec.rf.mag_10.std` | +0.040 | **+0.066** | +0.023 | **+0.055** | **+0.051** | +0.044 | +0.029 | **+0.165** |
| `vec.rf.mag_11.mean` | **+0.056** | **+0.075** | +0.040 | **+0.075** | **+0.083** | **+0.061** | +0.039 | **+0.247** |
| `vec.rf.mag_11.std` | +0.039 | +0.044 | +0.026 | +0.036 | **+0.075** | +0.043 | +0.011 | **+0.180** |
| `vec.rf.mag_12.mean` | **+0.056** | **+0.079** | +0.034 | **+0.061** | **+0.070** | **+0.059** | +0.035 | **+0.226** |
| `vec.rf.mag_12.std` | +0.042 | **+0.054** | +0.023 | +0.036 | **+0.055** | +0.043 | +0.008 | **+0.166** |
| `vec.rf.mag_20.mean` | **+0.050** | **+0.076** | +0.036 | **+0.090** | **+0.096** | **+0.057** | +0.039 | **+0.208** |
| `vec.rf.mag_20.std` | +0.037 | +0.043 | +0.022 | **+0.054** | **+0.095** | +0.041 | +0.026 | **+0.132** |
| `vec.rf.mag_21.mean` | **+0.057** | **+0.090** | +0.043 | **+0.076** | **+0.104** | **+0.064** | +0.044 | **+0.245** |
| `vec.rf.mag_21.std` | +0.039 | **+0.067** | +0.029 | +0.045 | **+0.103** | +0.046 | +0.021 | **+0.169** |
| `vec.rf.mag_22.mean` | **+0.056** | **+0.074** | +0.039 | **+0.060** | **+0.090** | **+0.060** | +0.042 | **+0.207** |
| `vec.rf.mag_22.std` | +0.036 | **+0.052** | +0.024 | +0.040 | **+0.070** | +0.041 | +0.021 | **+0.128** |

**Class: behaviour (SCB)** (19 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `vec.scb.hand_raising.frac` | +0.014 | **+0.138** | +0.000 | **+0.089** | +0.011 | +0.022 | +0.004 | **+0.099** |
| `vec.scb.hand_raising.max.mean` | +0.016 | **+0.132** | +0.005 | **+0.079** | +0.020 | +0.025 | +0.007 | **+0.154** |
| `vec.scb.hand_raising.max.std` | +0.011 | **+0.124** | +0.007 | **+0.069** | +0.028 | +0.022 | +0.006 | **+0.128** |
| `vec.scb.hand_raising.mean` | +0.007 | **+0.126** | −0.001 | **+0.088** | +0.006 | +0.016 | +0.001 | **+0.122** |
| `vec.scb.hand_raising.std` | −0.001 | **+0.115** | +0.000 | **+0.080** | +0.011 | +0.012 | −0.000 | **+0.096** |
| `vec.scb.m0.obj.max.mean` | +0.027 | +0.038 | +0.013 | **+0.080** | −0.007 | +0.022 | −0.011 | +0.041 |
| `vec.scb.m0.obj.max.std` | +0.015 | **+0.097** | +0.019 | **+0.104** | +0.016 | +0.025 | −0.002 | **+0.087** |
| `vec.scb.m0.obj.mean` | +0.017 | +0.030 | +0.004 | **+0.083** | −0.017 | +0.011 | −0.010 | +0.010 |
| `vec.scb.m0.obj.std` | −0.002 | +0.036 | +0.006 | **+0.084** | −0.034 | +0.002 | −0.013 | −0.019 |
| `vec.scb.read.frac` | −0.004 | **−0.059** | −0.001 | +0.037 | **−0.056** | −0.011 | −0.021 | **−0.099** |
| `vec.scb.read.max.mean` | +0.023 | −0.003 | +0.010 | **+0.071** | −0.009 | +0.015 | −0.015 | +0.018 |
| `vec.scb.read.max.std` | +0.007 | +0.031 | +0.012 | **+0.088** | +0.001 | +0.013 | −0.011 | +0.032 |
| `vec.scb.read.mean` | +0.012 | −0.009 | +0.000 | **+0.069** | −0.027 | +0.004 | −0.015 | −0.016 |
| `vec.scb.read.std` | −0.007 | −0.014 | −0.001 | **+0.069** | −0.045 | −0.008 | −0.018 | **−0.060** |
| `vec.scb.write.frac` | −0.003 | −0.037 | +0.005 | −0.016 | −0.020 | −0.006 | −0.029 | **−0.068** |
| `vec.scb.write.max.mean` | +0.046 | −0.012 | +0.021 | +0.041 | **+0.082** | +0.034 | +0.013 | +0.043 |
| `vec.scb.write.max.std` | +0.037 | −0.026 | +0.020 | +0.027 | **+0.085** | +0.027 | +0.007 | +0.016 |
| `vec.scb.write.mean` | +0.030 | −0.007 | +0.009 | +0.045 | **+0.055** | +0.019 | +0.007 | −0.007 |
| `vec.scb.write.std` | +0.024 | −0.017 | +0.010 | +0.026 | **+0.070** | +0.014 | +0.002 | −0.036 |

**Class: occupancy and count** (10 columns)

| Column | labeler_01 | labeler_02 | labeler_03 | labeler_04 | labeler_05 | All experts | Self-report | Model |
|---|---|---|---|---|---|---|---|---|
| `vec.n_tracks` | **−0.070** | **−0.124** | +0.017 | −0.012 | **−0.063** | −0.046 | −0.037 | **−0.257** |
| `vec.occ_00` | +0.009 | +0.023 | +0.023 | **+0.104** | +0.040 | +0.020 | +0.012 | **+0.128** |
| `vec.occ_01` | +0.013 | +0.021 | +0.021 | **+0.091** | **+0.064** | +0.022 | +0.023 | **+0.114** |
| `vec.occ_02` | −0.017 | **−0.067** | +0.023 | **+0.065** | −0.021 | −0.005 | +0.004 | −0.021 |
| `vec.occ_10` | +0.006 | −0.016 | +0.005 | +0.030 | **+0.118** | +0.008 | +0.015 | −0.013 |
| `vec.occ_11` | +0.002 | +0.042 | −0.024 | **−0.053** | **−0.076** | −0.008 | −0.007 | −0.042 |
| `vec.occ_12` | −0.013 | +0.016 | −0.028 | **−0.094** | **−0.105** | −0.023 | −0.026 | −0.041 |
| `vec.occ_20` | −0.024 | −0.012 | +0.000 | −0.015 | −0.025 | −0.016 | −0.019 | **−0.191** |
| `vec.occ_21` | −0.030 | **−0.088** | −0.009 | **−0.127** | +0.024 | −0.034 | −0.028 | **−0.208** |
| `vec.occ_22` | −0.040 | **−0.096** | +0.009 | −0.041 | +0.018 | −0.028 | −0.030 | **−0.171** |

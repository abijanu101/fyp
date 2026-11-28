# Class-relative and dynamic-clustering features

*25–27 November 2026. Config [`10_clustering.json`](10_clustering.json): experiment 09's
[`09_vfoa.json`](09_vfoa.json), unchanged (the same hash, `b1c2d485d3db`; the run itself passed
`09_vfoa.json`). Every setting of experiment 10's own steps (the screens, the choices between them, the
candidates, the ablation, the analyses after the run) is in
[`10_clustering_settings.json`](10_clustering_settings.json), a record, not a config. Experiment
09's tracks (`2a11240a2d42`) and features (`4c761244568f`), 52 recordings with 4 augmented copies each.
No new extraction. The design is guide 15 (§15.4 the options, §15.9 the screens). Two full runs: the
first (25–26 November, run report [`docs/run_reports/10_clustering.md`](../run_reports/10_clustering.md))
and the rerun with the fixed design (26–27 November, `python -m scripts.clustering.run_all`, commit
`086f28e`; results `outputs/experiments/10_clustering_v2/`, ablation `b1c2d485d3db-ff78da54`). Every
number below is from the rerun unless it says otherwise.*

## 1. Why this experiment

**09 added the head and found nothing that ranks.** Head direction moved the experts' ranking by at
most 0.01 and left the self-report at chance (09 §5). Every input so far describes a student on their
own. But a rating is a judgement made while watching a whole class: a student who writes while
everyone else writes looks different from one who writes while everyone else looks up. The classroom
literature measures exactly this, as synchrony across the class (Raca, Tormey and Dillenbourg, 2016;
guide 15 §15.1).

**So 10 describes each student against the class**, in two ways (guide 15 §15.4):

- **Synchrony (option 0):** how closely a student's signal follows the class's over the last 24 s
  (correlation, lag, how many classmates they move with), and how far their descriptors sit from the
  class's.
- **Clustering (options 1, 2 and 4):** group the students of each window, and describe each student by
  their group: the share of the class in it, whether they are alone, whether they are in the largest
  group, how far from its centre. Three families: threshold-based agglomerative clustering (Ward, τ),
  HDBSCAN, and a prototype codebook fitted per training fold.

**And it tests per-track normalisation** (each student's descriptors standardised against their own
track), which the clustering needs as input and which can also be given to the model directly.

The question: **does describing a student relative to the class improve the ranking of either label
set over 08–09's inputs?**

## 2. Setup

### 2.1 Screens before any label is used

Six label-free screens choose the inputs by fixed rules, written before the run (`run_all.py`;
guide 15 §15.9). Nothing in them sees a rating.

| Screen | Question | Rule | Chosen |
|---|---|---|---|
| 1–3. `signal_screen` | which channels move together across the class, beyond a surrogate null | per-recording Wilcoxon test, BH over channels (`pass_sync`); repeated on the calmest third of windows by scene motion (camera check) | all 10 channels: `rf.total`, `scb.read`, `scb.write`, `scb.m0.obj`, `scb.hand_raising`, `kg.speed`, `hp.head_height`, `hp.ear_asym`, `hp.yaw`, `hp.face_conf` |
| 4. `event_check` | do students react together at the big moments of a recording | share above threshold at events against random times, per recording; camera-like events removed | (reported, no choice) |
| 5. `descriptor_subset` | which descriptors to cluster on | within-student share ≥ 0.3, redundancy < 0.9, Laplacian score, 10 kept | `trk.kg.events`, `trk.rf.mag_01`, `trk.scb.write`, `trk.scb.write.max`, `trk.rf.mag_21`, `trk.scb.m0.obj.max`, `trk.kg.energy`, `trk.rf.mag_22`, `trk.hp.yaw.std`, `trk.rf.mag_20` |
| 6. `stability` | which threshold τ gives groups that persist | the highest temporal ARI in excess of a null (windows ≥ 20 s apart), agglomerative, usable | τ-desc **2.0**, τ-sync **0.7** |

### 2.2 Candidates and arms

**Candidates.** `candidates` computes 151 features per student and window, on the windows centred on
the rating entries (the same rows the models train on, augmented copies included): per student
`trk.cluster.*`, per window `cluster.*` (repeated on its students). Synchrony uses a 24 s context
ending with the window, a lag of up to ±3 s, and pairs of students sharing at least half its frames.
Descriptors are standardised per track, with the SD floored at 0.25 of the typical student's so a
student who barely moves is not inflated.

**Arms.** Twelve, fixed before the run: each is 09's input (input 1 + head pose, context off; the
`base` arm) plus one family.

| Arm | Adds to `base` |
|---|---|
| `base` | nothing (09's input 2) |
| `normalised` | the same columns, standardised per track |
| `sync` | option 0: synchrony, lag, degree per channel; deviation in descriptor space |
| `agglomerative` | option 1: agglomerative cluster features, descriptor and synchrony space |
| `hdbscan` | option 2 |
| `prototype` | option 4: the codebook, fitted on each training fold |
| `sync+agglomerative`, `sync+hdbscan`, `sync+prototype` | option 0 and one clustering family |
| `normalised+sync`, `normalised+sync+agglomerative` | normalised base columns, with option 0 (and 1) |
| `window` | the per-window `cluster.*` features only, centred per recording |

**Models and scoring, as in 06–09.** Logistic regression, one model per labeler, soft targets, with
the shared expert. Two splits:
- **group** (primary): leave one group out, 3 folds;
- **activity:** train on 4 of the 6 activities, test on the other 2, all 15 pairs (the activity is
  given to the model as an oracle).

Augmented copies are used for training only. Every number is on held-out entries.

**Two pre-registered primary tests**, group split, paired against `base` on the same entries:
1. `sync+agglomerative` on the experts' labels;
2. `normalised` on the self-report.

The difference in pooled Spearman is tested with a cluster bootstrap over recordings (2,000
resamples), at 98.75% (Bonferroni over the two tests). Every other comparison is exploratory, at 95%,
and not corrected.

**What is read.** Ranking: Spearman and AUC (engaged = rating ≥ 3), fold means as in 06–09, and the
paired pooled Spearman difference. Spread, as in 09: the predictions' SD over the labels' SD and the
share of predictions that round to 3 (halves toward 3), fold means (`src/evaluation/metrics.py`
`spread_metrics`, on the saved held-out predictions). Calibration: MAE. The human ceiling and the constant baselines are
09's (same config).

## 3. Results

### 3.1 The screens: students do move together

**Synchrony** (screen 1–3), per channel: the excess of the real class correlation over the surrogate
null, the share of recordings where it is positive, and the same excess on the calmest third of
windows.

| Channel | Excess | Recordings positive | Excess, calm windows | ρ with camera motion |
|---|---|---|---|---|
| `rf.total` | 0.145 | 52 / 52 | 0.133 | 0.046 |
| `scb.read` | 0.060 | 52 / 52 | 0.059 | −0.055 |
| `scb.write` | 0.058 | 52 / 52 | 0.059 | −0.073 |
| `scb.m0.obj` | 0.057 | 52 / 52 | 0.056 | −0.052 |
| `scb.hand_raising` | 0.052 | 52 / 52 | 0.052 | −0.029 |
| `kg.speed` | 0.033 | 52 / 52 | 0.027 | 0.121 |
| `hp.head_height` | 0.013 | 49 / 52 | 0.012 | 0.053 |
| `hp.ear_asym` | 0.012 | 48 / 52 | 0.013 | 0.034 |
| `hp.yaw` | 0.011 | 44 / 52 | 0.013 | 0.015 |
| `hp.face_conf` | 0.010 | 49 / 52 | 0.011 | 0.011 |

Every channel passes (q ≈ 0 over recordings), in the calm windows too, and none tracks camera motion
(|ρ| ≤ 0.12). Optical flow is the most synchronous, then behaviour; head pose is weak (0.01) but
consistent. (The first run passed 7 channels, failing three of the four head-pose ones, with a sign test
over pooled windows; the rerun tests recordings, the unit that is independent.)

**Event reactions** (screen 4): at the biggest moments of a recording, the share of students above
threshold against random times.

| Channel | Ratio, all events | Camera-like events | Ratio without them | p | q |
|---|---|---|---|---|---|
| `rf.total` | 1.60 | 78% | 1.20 | 0.047 | 0.139 |
| `scb.m0.obj` | 1.21 | 56% | 1.18 | 0.011 | 0.087 |
| `scb.hand_raising` | 1.18 | 51% | 1.11 | 0.017 | 0.087 |
| `scb.read` | 1.19 | 54% | 1.10 | 0.056 | 0.139 |
| `scb.write` | 1.25 | 53% | 1.08 | 0.199 | 0.397 |
| `kg.speed` | 1.26 | 61% | 1.04 | 0.486 | 0.694 |
| head pose (4 channels) | 1.08–1.15 | 52–58% | 0.97–1.04 | 0.26–0.86 | 0.44–0.86 |

Every channel reacts when all events are counted, but more than half of the events look like camera
movement. Without them, behaviour and flow keep a reaction of 1.1–1.2, and none survives the
correction (every q ≥ 0.087). Head pose shows no reaction.

**Stability** (screen 6), agglomerative clustering on the descriptor subset:

| τ-desc | 0.4 | 0.6 | 0.8 | 1.0 | 1.5 | **2.0** | 3.0 |
|---|---|---|---|---|---|---|---|
| Temporal ARI | 0.230 | 0.219 | 0.246 | 0.264 | 0.308 | **0.382** | 0.595 |
| Null ARI | 0.058 | 0.007 | 0.006 | 0.005 | 0.007 | **0.049** | 0.348 |
| Excess | 0.172 | 0.211 | 0.240 | 0.259 | 0.301 | **0.332** | 0.247 |
| Clusters per student | 0.84 | 0.66 | 0.52 | 0.41 | 0.25 | **0.17** | 0.11 |

The excess peaks inside the grid. The first run chose τ on the temporal ARI alone, which keeps rising
as the clusters merge (3.0 scores 0.595, mostly null); the null fixes that. In synchrony space the
excess peaks at τ 0.7 (0.146, mean over channels), with 0.9 at 0.143.

So the groups are real and they persist: what follows is whether they say anything about the ratings.

### 3.2 The feature screen

Each of the 151 candidates against the ratings, on training folds only: Spearman with a within-student
circular-shift null (200 shifts), BH within each fold.

- **Experts, group split: 13 candidates significant in every fold**, all agglomerative or prototype
  features, all |ρ| ≤ 0.036. The strongest: `trk.cluster.agg.desc.in_largest` (ρ −0.036),
  `agg.desc.share` (−0.032), `proto.subset.conformity` (−0.030), `cluster.agg.desc.largest_share`
  (−0.030). The sign says that students in the class's largest, most typical group are rated a little
  lower, and students away from it a little higher.
- **Experts, activity split:** the same features lead, significant in 11 of 15 folds; none in every
  fold.

The 13 candidates significant (q < 0.05) in every group fold, on the experts' labels. ρ: mean over
the training folds, with its SD; held-out: the same candidate on each fold's held-out group, mean over
folds; activity: the mean over the 15 activity folds, and in how many it is significant.

| Candidate | ρ, group split | SD | ρ, held-out | ρ, activity split | Activity folds significant |
|---|---|---|---|---|---|
| `trk.cluster.agg.desc.in_largest` | −0.036 | 0.010 | −0.036 | −0.035 | 11 / 15 |
| `trk.cluster.agg.desc.share` | −0.032 | 0.007 | −0.033 | −0.031 | 11 / 15 |
| `trk.cluster.proto.subset.conformity` | −0.030 | 0.004 | −0.030 | −0.030 | 11 / 15 |
| `cluster.agg.desc.largest_share` | −0.030 | 0.006 | −0.030 | −0.029 | 11 / 15 |
| `cluster.proto.subset.proto_7` | +0.027 | 0.008 | +0.038 | +0.012 | 5 / 15 |
| `cluster.proto.subset.entropy` | +0.026 | 0.005 | +0.026 | +0.026 | 11 / 15 |
| `cluster.agg.scb.write.n_clusters` | −0.026 | 0.004 | −0.027 | −0.025 | 11 / 15 |
| `trk.cluster.proto.subset.nearest_dist` | +0.026 | 0.007 | +0.026 | +0.026 | 11 / 15 |
| `trk.cluster.proto.subset.hard_share` | −0.022 | 0.004 | −0.022 | −0.022 | 11 / 15 |
| `cluster.agg.desc.n_clusters` | +0.022 | 0.002 | +0.022 | +0.021 | 10 / 15 |
| `cluster.agg.desc.fragmentation` | +0.020 | 0.004 | +0.020 | +0.019 | 10 / 15 |
| `trk.cluster.agg.desc.centroid_dist` | +0.018 | 0.004 | +0.018 | +0.017 | 9 / 15 |
| `trk.cluster.agg.scb.write.singleton` | −0.016 | 0.001 | −0.014 | −0.017 | 6 / 15 |

They tell one story: a student in the largest, most typical group, near a prototype, scores lower;
a student far from it, in a fragmented class, scores higher. They hold on the held-out groups at the
size seen in training. Every one is below 0.04 in absolute value.
- **Self-report: no candidate is significant in any fold**, on either split. The largest is
  `trk.cluster.sync.desc.dev` (ρ −0.047, median q 0.19).
- **Missing values carry no recording identity.** 1.3% of values are NaN, and the NaN pattern predicts
  the recording at 4.8% accuracy (the largest recording is 3.6% of rows).

### 3.3 The ablation: group split

Every data cell is train / test: the same fold's model on its own training groups' original
entries, then on its held-out group, as in 07–08; fold means (3 folds). Rows in italics are
references with one value per cell, on all entries of the track (experiment 06 Part 1, as in
07–09). \* threshold (AUC: labels split at rating ≥ 3); \*\* discretisation (prediction rounded
to a whole rating, halves toward 3). Share predicted 3: the same rounding (09's median-hugging
check). Constant 3 on the training entries is computed from their rating counts.

#### Expert entries (21,519 held-out)

**Profile** (train / test)

| | Mean | Median | SD | Share predicted 3 |
|---|---|---|---|---|
| `base` | 2.794 / 2.798 | 2.761 / 2.748 | 0.176 / 0.204 | 0.99 / 0.96 |
| `normalised` | 2.788 / 2.793 | 2.753 / 2.746 | 0.153 / 0.171 | 0.99 / 0.98 |
| `sync` | 2.794 / 2.798 | 2.760 / 2.751 | 0.190 / 0.220 | 0.98 / 0.95 |
| `agglomerative` | 2.800 / 2.803 | 2.768 / 2.757 | 0.190 / 0.221 | 0.98 / 0.95 |
| `hdbscan` | 2.794 / 2.798 | 2.761 / 2.749 | 0.177 / 0.205 | 0.98 / 0.96 |
| `prototype` | 2.793 / 2.798 | 2.759 / 2.749 | 0.179 / 0.206 | 0.98 / 0.96 |
| `sync+agglomerative` | 2.800 / 2.802 | 2.768 / 2.757 | 0.199 / 0.231 | 0.97 / 0.94 |
| `sync+hdbscan` | 2.795 / 2.799 | 2.760 / 2.751 | 0.190 / 0.220 | 0.97 / 0.95 |
| `sync+prototype` | 2.795 / 2.799 | 2.760 / 2.752 | 0.191 / 0.221 | 0.97 / 0.95 |
| `normalised+sync` | 2.792 / 2.797 | 2.760 / 2.754 | 0.168 / 0.190 | 0.98 / 0.97 |
| `normalised+sync+agglomerative` | 2.802 / 2.802 | 2.775 / 2.762 | 0.180 / 0.204 | 0.97 / 0.96 |
| `window` | 2.795 / 2.800 | 2.761 / 2.755 | 0.189 / 0.216 | 0.97 / 0.95 |
| Constant 3 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 | 1.00 / 1.00 |
| *Self-report labels* | 2.885 | 3.000 | 0.895 | |
| *Median of expert labels* | 2.895 | 3.000 | 0.510 | |

**Metrics** (train / test)

| | MAE | RMSE | AUC* | Spearman | Macro F1** | Accuracy** |
|---|---|---|---|---|---|---|
| `base` | 0.639 / 0.675 | 0.788 / 0.835 | 0.574 / 0.535 | 0.199 / 0.103 | 0.136 / 0.138 | 0.433 / 0.422 |
| `normalised` | 0.642 / 0.671 | 0.794 / 0.828 | 0.568 / 0.533 | 0.171 / 0.093 | 0.131 / 0.133 | 0.431 / 0.423 |
| `sync` | 0.635 / 0.676 | 0.783 / 0.837 | 0.582 / 0.535 | 0.217 / 0.104 | 0.144 / 0.143 | 0.435 / 0.421 |
| `agglomerative` | 0.635 / 0.676 | 0.783 / 0.836 | 0.583 / 0.536 | 0.221 / 0.106 | 0.143 / 0.142 | 0.434 / 0.421 |
| `hdbscan` | 0.638 / 0.675 | 0.787 / 0.835 | 0.575 / 0.536 | 0.201 / 0.104 | 0.137 / 0.139 | 0.433 / 0.423 |
| `prototype` | 0.638 / 0.675 | 0.787 / 0.835 | 0.577 / 0.538 | 0.204 / 0.107 | 0.138 / 0.139 | 0.433 / 0.422 |
| `sync+agglomerative` | 0.632 / 0.678 | 0.779 / 0.839 | 0.589 / 0.535 | 0.234 / 0.105 | 0.149 / 0.145 | 0.436 / 0.420 |
| `sync+hdbscan` | 0.635 / 0.677 | 0.783 / 0.837 | 0.582 / 0.535 | 0.218 / 0.104 | 0.145 / 0.143 | 0.435 / 0.422 |
| `sync+prototype` | 0.635 / 0.677 | 0.783 / 0.837 | 0.583 / 0.536 | 0.220 / 0.106 | 0.145 / 0.143 | 0.435 / 0.421 |
| `normalised+sync` | 0.637 / 0.671 | 0.789 / 0.830 | 0.575 / 0.531 | 0.191 / 0.090 | 0.139 / 0.136 | 0.434 / 0.422 |
| `normalised+sync+agglomerative` | 0.633 / 0.672 | 0.784 / 0.832 | 0.585 / 0.533 | 0.214 / 0.094 | 0.143 / 0.138 | 0.435 / 0.421 |
| `window` | 0.635 / 0.676 | 0.784 / 0.838 | 0.581 / 0.535 | 0.215 / 0.103 | 0.145 / 0.142 | 0.434 / 0.420 |
| Constant 3 | 0.618 / 0.619 | 0.841 / 0.839 | 0.500 / 0.500 | - / - | 0.120 / 0.119 | 0.426 / 0.423 |
| *Mean of experts* | 0.668 | 0.860 | 0.563 | 0.143 | 0.208 | 0.447 |
| *Median of experts* | 0.638 | 0.913 | 0.543 | 0.120 | 0.222 | 0.451 |

#### Self-report entries (2,463 held-out)

**Profile** (train / test)

| | Mean | Median | SD | Share predicted 3 |
|---|---|---|---|---|
| `base` | 2.870 / 2.883 | 2.861 / 2.875 | 0.091 / 0.090 | 1.00 / 1.00 |
| `normalised` | 2.865 / 2.862 | 2.871 / 2.872 | 0.109 / 0.113 | 0.99 / 0.99 |
| `sync` | 2.872 / 2.885 | 2.870 / 2.884 | 0.148 / 0.141 | 0.98 / 0.99 |
| `agglomerative` | 2.879 / 2.897 | 2.871 / 2.892 | 0.133 / 0.129 | 0.99 / 1.00 |
| `hdbscan` | 2.870 / 2.884 | 2.862 / 2.875 | 0.099 / 0.097 | 1.00 / 1.00 |
| `prototype` | 2.869 / 2.880 | 2.867 / 2.874 | 0.113 / 0.109 | 0.99 / 1.00 |
| `sync+agglomerative` | 2.880 / 2.893 | 2.874 / 2.893 | 0.163 / 0.156 | 0.98 / 0.98 |
| `sync+hdbscan` | 2.872 / 2.886 | 2.871 / 2.884 | 0.151 / 0.143 | 0.98 / 0.99 |
| `sync+prototype` | 2.873 / 2.885 | 2.870 / 2.883 | 0.152 / 0.144 | 0.98 / 0.99 |
| `normalised+sync` | 2.873 / 2.871 | 2.878 / 2.875 | 0.150 / 0.151 | 0.98 / 0.98 |
| `normalised+sync+agglomerative` | 2.883 / 2.876 | 2.884 / 2.880 | 0.163 / 0.164 | 0.98 / 0.97 |
| `window` | 2.867 / 2.879 | 2.861 / 2.874 | 0.128 / 0.129 | 0.99 / 0.99 |
| Constant 3 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 | 1.00 / 1.00 |
| *Self-report labels* | 2.885 | 3.000 | 0.895 | |
| *Median of expert labels* | 2.876 | 3.000 | 0.428 | |

**Metrics** (train / test)

| | MAE | RMSE | AUC* | Spearman | Macro F1** | Accuracy** |
|---|---|---|---|---|---|---|
| `base` | 0.691 / 0.712 | 0.890 / 0.920 | 0.549 / 0.461 | 0.120 / -0.081 | 0.118 / 0.117 | 0.415 / 0.412 |
| `normalised` | 0.684 / 0.703 | 0.888 / 0.912 | 0.589 / 0.546 | 0.151 / 0.060 | 0.124 / 0.127 | 0.419 / 0.418 |
| `sync` | 0.684 / 0.711 | 0.880 / 0.916 | 0.591 / 0.511 | 0.194 / 0.029 | 0.128 / 0.124 | 0.419 / 0.415 |
| `agglomerative` | 0.687 / 0.713 | 0.883 / 0.922 | 0.583 / 0.489 | 0.175 / -0.024 | 0.121 / 0.119 | 0.415 / 0.414 |
| `hdbscan` | 0.690 / 0.712 | 0.889 / 0.919 | 0.558 / 0.475 | 0.133 / -0.054 | 0.118 / 0.118 | 0.414 / 0.413 |
| `prototype` | 0.689 / 0.714 | 0.887 / 0.921 | 0.564 / 0.484 | 0.140 / -0.037 | 0.122 / 0.119 | 0.417 / 0.413 |
| `sync+agglomerative` | 0.683 / 0.716 | 0.875 / 0.921 | 0.600 / 0.506 | 0.217 / 0.017 | 0.127 / 0.125 | 0.418 / 0.415 |
| `sync+hdbscan` | 0.684 / 0.711 | 0.879 / 0.915 | 0.593 / 0.516 | 0.197 / 0.036 | 0.128 / 0.123 | 0.419 / 0.413 |
| `sync+prototype` | 0.683 / 0.713 | 0.879 / 0.918 | 0.594 / 0.511 | 0.197 / 0.024 | 0.129 / 0.123 | 0.421 / 0.414 |
| `normalised+sync` | 0.679 / 0.705 | 0.880 / 0.911 | 0.608 / 0.548 | 0.201 / 0.079 | 0.130 / 0.129 | 0.422 / 0.417 |
| `normalised+sync+agglomerative` | 0.678 / 0.712 | 0.875 / 0.918 | 0.613 / 0.540 | 0.224 / 0.063 | 0.132 / 0.132 | 0.422 / 0.420 |
| `window` | 0.689 / 0.724 | 0.884 / 0.931 | 0.572 / 0.446 | 0.166 / -0.104 | 0.122 / 0.119 | 0.416 / 0.412 |
| Constant 3 | 0.666 / 0.668 | 0.908 / 0.908 | 0.500 / 0.500 | - / - | 0.117 / 0.117 | 0.413 / 0.412 |
| *Mean of experts* | 0.689 | 0.884 | 0.619 | 0.241 | 0.179 | 0.435 |
| *Median of experts* | 0.661 | 0.914 | 0.609 | 0.213 | 0.179 | 0.439 |

**Paired against `base`**, pooled Spearman over all held-out entries, cluster bootstrap over
recordings (the two primary tests at 98.75%, the rest at 95%):

| Arm | Experts: difference [interval] | Self-report: difference [interval] |
|---|---|---|
| `normalised` | −0.010 [−0.026, +0.008] | **+0.115 [+0.053, +0.184]** (primary 2: **supported**) |
| `sync` | +0.003 [−0.004, +0.010] | +0.100 [+0.059, +0.142] |
| `agglomerative` | +0.010 [+0.001, +0.019] | +0.055 [+0.016, +0.098] |
| `hdbscan` | +0.001 [−0.000, +0.003] | +0.023 [+0.004, +0.043] |
| `prototype` | +0.003 [−0.001, +0.006] | +0.029 [−0.000, +0.058] |
| `sync+agglomerative` | +0.008 [−0.003, +0.020] (primary 1: **not supported**) | +0.088 [+0.035, +0.144] |
| `sync+hdbscan` | +0.004 [−0.004, +0.011] | +0.107 [+0.061, +0.154] |
| `sync+prototype` | +0.004 [−0.003, +0.012] | +0.090 [+0.045, +0.138] |
| `normalised+sync` | −0.007 [−0.022, +0.011] | +0.147 [+0.089, +0.205] |
| `normalised+sync+agglomerative` | −0.000 [−0.019, +0.018] | +0.128 [+0.063, +0.192] |
| `window` | +0.003 [−0.002, +0.008] | −0.015 [−0.046, +0.015] |

Pooled `base`: experts 0.055, self-report −0.130. The primary intervals are at 98.75%; the 95%
interval of primary 1 is [−0.002, +0.018].

- **Primary 1 is not supported.** `sync+agglomerative` adds +0.008 to the experts' pooled Spearman,
  and its interval includes zero. In fold means it adds +0.002.
- **Primary 2 is supported, but it is a correction, not a gain.** `base` ranks the held-out group's
  self-report in the wrong direction (pooled −0.130, fold mean −0.081), as LR did in 09 §4.1.
  Normalising each student's columns against their own track takes away whatever differs between
  groups and brings the ranking to chance (pooled −0.015, fold mean +0.060). Nothing ranks the
  self-report as well as the experts do (Spearman 0.21–0.24).
- **On the experts, only agglomerative clustering helps, by about 0.01.** It is the one arm whose
  interval excludes zero, and it uses the features the screen picked (§3.2). Normalisation costs the
  experts about 0.01.
- **On the self-report, every arm with synchrony features lifts the ranking from below chance to
  chance or a little above** (+0.09 to +0.15). Against `sync` itself, only `normalised+sync` is
  clearly better (+0.047 [+0.008, +0.089]); the clustering families add nothing on top of `sync`
  (−0.012 to +0.007).
- **The per-window features help neither label set** (experts +0.003; self-report −0.015, the lowest arm).
- **A constant 3 still has the lowest MAE** on both label sets (0.619 and 0.668 against 0.671–0.678 and
  0.703–0.724). No arm improves calibration.
- **The predictions still sit at the middle.** `base` spreads its predictions over 0.25 of the
  labels' SD with 96% rounding to 3, exactly 09's logistic regression on the same input (09 §4.2:
  0.25 and 0.96); on the self-report 0.10 and 100% (09: 0.10 and 1.00). The class-relative features
  widen the spread a little (up to 0.29 on the experts, 0.18 on the self-report) without improving
  the ranking. By 06's rule (§1.2: the squared-error-optimal spread is r × the label SD), even that
  is too wide: pooled, `base` has Pearson r 0.058 on the experts' labels, an optimum of 0.047
  against an actual SD of 0.232. So a constant 3 wins on MAE because the models spread noise, not
  because they are too cautious.

### 3.4 The ablation: activity split

Held-out, fold means (15 folds), and the pooled difference against `base` (95%):

| Arm | Experts: Spearman | Experts: difference | Self-report: Spearman | Self-report: difference |
|---|---|---|---|---|
| `base` | 0.107 | – | −0.046 | – |
| `normalised` | 0.108 | −0.008 [−0.029, +0.014] | 0.039 | +0.105 [+0.039, +0.176] |
| `sync` | 0.108 | +0.005 [−0.002, +0.013] | 0.037 | +0.125 [+0.062, +0.194] |
| `agglomerative` | 0.118 | +0.016 [+0.006, +0.027] | −0.003 | +0.076 [+0.028, +0.132] |
| `hdbscan` | 0.108 | +0.002 [−0.000, +0.003] | −0.025 | +0.027 [+0.005, +0.049] |
| `prototype` | 0.109 | +0.003 [−0.001, +0.008] | −0.016 | +0.050 [+0.011, +0.091] |
| `sync+agglomerative` | 0.117 | +0.016 [+0.004, +0.030] | 0.026 | +0.114 [+0.050, +0.180] |
| `sync+hdbscan` | 0.109 | +0.006 [−0.001, +0.014] | 0.040 | +0.128 [+0.063, +0.198] |
| `sync+prototype` | 0.108 | +0.005 [−0.002, +0.013] | 0.035 | +0.124 [+0.054, +0.196] |
| `normalised+sync` | 0.100 | −0.012 [−0.031, +0.009] | 0.067 | +0.147 [+0.077, +0.219] |
| `normalised+sync+agglomerative` | 0.109 | −0.001 [−0.021, +0.020] | 0.048 | +0.132 [+0.065, +0.201] |
| `window` | 0.103 | −0.003 [−0.010, +0.002] | −0.048 | +0.009 [−0.024, +0.045] |

The activity split repeats the group split: agglomerative features add +0.016 to the experts,
synchrony and normalisation lift the self-report to chance, and the per-window features do nothing.
Constant 3's MAE (0.615 and 0.669) again beats every arm (0.648–0.650 and 0.700–0.717).

The full profile and metrics tables of the activity split, train / test, are in Appendix B. On
the experts' labels its models spread less than the group split's (test SD 0.14–0.18 against
0.17–0.23); on the self-report about the same.

### 3.5 Importance and overfitting

**Importance** (`importance`, 2 repeats). The model is `base` plus every candidate family (sync,
agglomerative, HDBSCAN, prototype). Each of the 51 feature groups is shuffled across the held-out rows
and the fall in Spearman is recorded: positive, the model uses the group on unseen data; negative,
the group hurts. The SD is over folds. `base:` marks the original inputs; `win:` per-window features,
centred per recording.

**Experts' labels**, group split:

| Rank | Feature group | Spearman fall | SD | MAE rise |
|---|---|---|---|---|
| 1 | `base:trk.rf` (optical flow) | 0.038 | 0.005 | 0.003 |
| 2 | `base:trk.kg` (kinematics) | 0.023 | 0.006 | 0.003 |
| 3 | `base:trk.hp` (head pose) | 0.010 | 0.010 | 0.004 |
| 4 | `agg.desc` | 0.006 | 0.011 | 0.001 |
| 5 | `sync.scb.m0.obj` | 0.004 | 0.002 | 0.001 |
| 6 | `win:sync.scb.m0.obj` | 0.004 | 0.002 | 0.001 |
| 7 | `agg.rf.total` | 0.004 | 0.005 | 0.000 |
| 8 | `win:agg.desc` | 0.004 | 0.001 | 0.000 |
| 9 | `sync.desc` | 0.003 | 0.002 | 0.001 |
| 10 | `base:trk.scb` (behaviour) | 0.003 | 0.003 | 0.001 |

**Self-report**, group split:

| Rank | Feature group | Spearman fall | SD | MAE rise |
|---|---|---|---|---|
| 1 | `sync.desc` | 0.036 | 0.006 | 0.008 |
| 2 | `sync.scb.m0.obj` | 0.017 | 0.020 | 0.008 |
| 3 | `base:trk.rf` (optical flow) | 0.011 | 0.018 | 0.002 |
| 4 | `agg.scb.write` | 0.011 | 0.009 | 0.000 |
| 5 | `win:hdb.desc` | 0.009 | 0.005 | 0.003 |
| 6 | `base:trk.kg` (kinematics) | 0.008 | 0.018 | 0.001 |
| 7 | `agg.rf.total` | 0.007 | 0.004 | 0.002 |
| 8 | `agg.hp.head_height` | 0.007 | 0.004 | 0.001 |
| 9 | `win:sync.hp.head_height` | 0.005 | 0.002 | 0.001 |
| 10 | `sync.kg.speed` | 0.005 | 0.008 | 0.000 |
| 16 | `base:trk.scb` (behaviour) | 0.003 | 0.003 | |
| 51 (last) | `base:trk.hp` (head pose) | **−0.033** | 0.014 | |

**The base inputs on both splits** (fall in Spearman, rank of 51):

| Feature group | Experts, group | Experts, activity | Self-report, group | Self-report, activity |
|---|---|---|---|---|
| `base:trk.rf` (optical flow) | 0.038 (1) | 0.023 (2) | 0.011 (3) | −0.005 (47) |
| `base:trk.kg` (kinematics) | 0.023 (2) | 0.030 (1) | 0.008 (6) | 0.006 (6) |
| `base:trk.hp` (head pose) | 0.010 (3) | 0.011 (3) | −0.033 (51) | −0.005 (50) |
| `base:trk.scb` (behaviour) | 0.003 (10) | 0.010 (4) | 0.003 (16) | −0.005 (49) |

- **The base columns carry the experts' models.** Optical flow and kinematics lead on both splits,
  and head pose is third. The best clustering group, `agg.desc`, is fourth on the group split, at
  0.006, with an SD (0.011) larger than its effect.
- **On the self-report, synchrony in descriptor space leads** (`sync.desc`, 0.036 on the group split,
  0.026 on the activity split), which matches the gain of the synchrony arms. The base groups matter
  little.
- **Head pose hurts the self-report on new students.** Shuffling it raises the held-out group's
  Spearman by 0.033, the only clearly negative group. Whatever the model learns from head pose on
  two groups' self-reports reverses on the third, as in 09 §4.1. With the students already seen (the
  activity split) it is about neutral (−0.005).
- **Every arm overfits by the same margin.** Training Spearman on the experts is 0.17–0.23 against
  0.09–0.11 held-out, and the class-relative features widen the gap (`base` 0.199 → 0.103;
  `sync+agglomerative` 0.234 → 0.105).
- **Side result: the cluster features help predict the activity.** The gate's activity accuracy rises
  from 0.335 (scene features only) to 0.401 (scene and cluster features).

### 3.6 Against the first run

The first run (25–26 November) tested four arms on fold means. Its design had faults that the rerun
fixed:
- a `sync+best` arm chosen on the test folds;
- τ chosen at the edge of the grid;
- channels tested over pooled windows;
- a mutual-information screen that reports about 4 nats on independent tied data.

| Group split, fold-mean Spearman, difference against `base` | First run | Rerun |
|---|---|---|
| `normalised`, self-report | +0.145 | +0.142 |
| `normalised`, experts | −0.012 | −0.010 |
| `sync`, self-report | +0.073 | +0.110 |
| `sync+agglomerative`, experts | +0.010 | +0.002 |

Normalisation's effect repeats almost exactly. The small expert gain of the first run's
`sync+agglomerative` shrinks once τ is chosen against a null and nothing is chosen on test folds.

### 3.7 After the run: the self-report within students

*Not pre-registered. This analysis was done after the results above, on the saved held-out predictions
(`ablation_predictions.csv.gz`, group split; `python -m scripts.analysis.self_within_student`). No
model is trained. Student = group and subject, the same person across activities.*

**Is the self-report a trait or a state?** If students gave themselves the same score every time,
nothing within a student would be left to predict.

| | |
|---|---|
| Self-report entries | 2,463 from 56 students (277 student-sessions, 3 groups) |
| Entries per student | median 40 (9–121); one about every 8 s within a session |
| Distinct scores per student | 2: 7 students; 3: 19; 4: 24; 5: 6. None gives a single score |
| Variance within a student | **86%** (0.693 of 0.809); within a student-session, 75% |
| Within-student SD | median 0.79 (quartiles 0.69–0.92) |
| Drift with time in session | Spearman −0.027 (experts' labels: −0.015) |

So the self-report is mostly a state: students move up and down their own scale, and not with the
time since the start.

**Who tracks it within students?** Held-out Spearman, pooled over all entries and within students
(rating and prediction each minus the student's mean):

| Predictor | Pooled | Within student |
|---|---|---|
| The nearest expert rating (same student and session, within 5 s; 2,231 entries) | **+0.232** | **+0.234** |
| … within 15 s (2,414 entries) | +0.219 | +0.225 |
| The expert models' prediction at that entry (`base`, within 5 s) | +0.006 | +0.049 |
| Self-report models: `base` | −0.130 | −0.056 |
| `normalised` | −0.015 | +0.053 |
| `sync` | −0.030 | +0.027 |
| `normalised+sync` | +0.017 | **+0.065** |
| `normalised+sync+agglomerative` | −0.003 | +0.045 |
| `sync+agglomerative` | −0.043 | +0.009 |
| `agglomerative`, `hdbscan`, `prototype` | −0.107 to −0.075 | −0.028 to −0.019 |
| `window` | −0.145 | −0.073 |

- **What changes in a student's own report is visible on video.** When a student rates themselves
  above their own mean, an expert watching the same moment tends to rate them higher too (+0.234
  within students). That is as strong as the experts' agreement with the self-report overall.
- **The models catch about a quarter of it at best.** `normalised+sync` reaches +0.065 within
  students; `base` is below zero. Normalisation is what turns the sign, here as in §3.3.
- **Training on the experts does not transfer.** The expert models' prediction tracks the
  self-report at +0.049 within students, no better than the self-report models, although the
  experts' own ratings reach +0.234. The gap is not a lack of labels: the experts see something
  the features do not carry.

This pointed to one model not yet tried, on the existing features: one trained within students
(the target is the student's score minus their own mean; the inputs are per-track normalised).
Appendix D runs it, and it does not do better (W1 against W0′: experts +0.002, self-report −0.032);
Appendix C shows why there is little to gain.

## 4. Conclusion

- **Students in these classrooms do move together, and the groups persist.** All 10 channels are
  synchronous beyond a null in every recording, the camera does not explain it, and agglomerative
  groups at τ 2.0 persist well beyond chance (temporal ARI excess 0.33).
- **But describing a student against the class does not improve the experts' ranking by more than
  0.01.** The pre-registered test (`sync+agglomerative`) is not supported. Agglomerative clustering
  alone gives about +0.01 to +0.016 on both splits, consistently and with the screen's backing, but too
  little to matter: the models stay below the median expert (Spearman 0.11 against 0.12).
- **Per-track normalisation fixes the self-report baseline, but doesn't make the self-report
  predictable.** Supported (+0.115 [+0.053, +0.184]), and repeated from the first run. It removes the
  between-group difference that made `base` rank the self-report backwards; the result is chance, not
  prediction. A self-report model should use normalised (or normalised+sync) inputs, so that it at
  least does not reverse.
- **Neither the per-window features nor HDBSCAN nor the prototype codebook helps.**
- **The self-report is mostly a state, and the video shows it** (§3.7, after the run). 86% of its
  variance is within students, and the experts' ratings follow it there at +0.23. The best model
  reaches +0.065, and an expert-trained model +0.05. This is the one gap left that is not set by the
  labels.
- **Between students, the features carry none of what the labelers rate** (Appendix C, after the
  run). Entries that look nearly the same in another group are rated no more alike than random pairs
  (−0.002), with the columns weighted equally or by what predicts ratings, while two experts on the
  same moment agree at 0.09–0.15. So the limit is not only the labels' agreement, as 08 and 09
  concluded: these features do not hold the between-student information that the labelers' (weak)
  agreement does. Within a session they hold a little (+0.03 to +0.08 beyond time).
- **Training on the within-session part does not help** (Appendix D, fixed before the run). A model
  trained on each rating's deviation from its labeler-session mean ties one trained on the rating on
  the experts' labels (+0.002 [−0.003, +0.007]) and is worse on the self-report (−0.032). The models
  already capture the within-session signal: about 0.08 against the humans' 0.137 for the experts,
  0.06 against 0.271 for the self-report. A causal version, using only each student's past, loses
  nothing.
- **Calibration is not the way either.** A constant 3 keeps the lowest MAE, and every arm overfits its
  training groups by about two to one. Further gains need new information in the input (the face,
  expression, where the student looks), not new models or targets on these features.

**Not run:** per-labeler tables, the classifier and k-NN models of 06–09, and a single pre-registered
test of `agg.desc.in_largest` on its own.

## Appendix A: reproducing

```
python -m scripts.clustering.run_all --config docs/experiments/10_clustering.json \
    --out outputs/experiments/10_clustering_v2          # resumable; steps 4-7 of scripts/clustering/README.md
```

The runner chains `signal_screen`, `event_check`, `descriptor_subset`, `stability`, `candidates`,
`feature_screen`, `importance --gate` and `ablation`, and then `report`. It picks the channels and
τ by the rules in §2.1 and saves them in `run_choices.json`. Each step resumes inside itself; a step
that finished with the same command is skipped. The results are in
`outputs/experiments/10_clustering_v2/`:
- `clustering_report.json` (with the completeness check);
- `ablation/ablation.csv` (fold means);
- `ablation/ablation_intervals.csv` (paired differences);
- `feature_screen.csv`;
- `importance_summary.csv`.

Runtime on the run machine (Ryzen 5 3600, 16 GB, CPU only; 10 workers for the per-recording steps, 1
for the label steps), in minutes, from the runner's markers:

| Step | Minutes |
|---|---|
| `signal_screen` | 3 |
| `event_check` | 1 |
| `descriptor_subset` | < 1 |
| `stability` | 39 |
| `candidates` | 160 |
| `feature_screen` | 3 |
| `importance` | 24 |
| `ablation` | 199 |
| `report` | < 1 |

The `feature_screen` and `importance` times are for the last invocation: both resumed from per-fold
caches left by an earlier, interrupted one.

Unit tests: `tests/test_clustering_fixes.py`. The real-data end-to-end test (two recordings, with and
without augmented copies) is `tests/test_clustering_sample.py`, run with `FYP_SAMPLE_TEST=1`.

**After the run** (§3.7, Appendices C and D), on the same outputs:

```
python -m scripts.analysis.self_within_student           # §3.7, from ablation/ablation_predictions.csv.gz
python -m scripts.clustering.feature_similarity --out outputs/experiments/10_clustering_v2   # Appendix C, ~15 min
python -m scripts.clustering.within_session --out outputs/experiments/10_clustering_v2       # Appendix D, ~35 min
```

Their tests, on synthetic data with known answers: `tests/test_feature_similarity.py` and
`tests/test_within_session.py`.

## Appendix B: full tables, activity split

Every data cell is train / test: the same fold's model on its own training groups' original
entries, then on its held-out group, as in 07–08; fold means (15 folds; an entry is held out in 5 of them). Rows in italics are
references with one value per cell, on all entries of the track (experiment 06 Part 1, as in
07–09). \* threshold (AUC: labels split at rating ≥ 3); \*\* discretisation (prediction rounded
to a whole rating, halves toward 3). Share predicted 3: the same rounding (09's median-hugging
check). Constant 3 on the training entries is computed from their rating counts.

### Expert entries

**Profile** (train / test)

| | Mean | Median | SD | Share predicted 3 |
|---|---|---|---|---|
| `base` | 2.787 / 2.779 | 2.758 / 2.750 | 0.163 / 0.154 | 0.99 / 0.99 |
| `normalised` | 2.782 / 2.779 | 2.752 / 2.748 | 0.140 / 0.138 | 0.99 / 0.99 |
| `sync` | 2.788 / 2.782 | 2.757 / 2.753 | 0.177 / 0.167 | 0.98 / 0.98 |
| `agglomerative` | 2.793 / 2.787 | 2.765 / 2.761 | 0.179 / 0.171 | 0.98 / 0.98 |
| `hdbscan` | 2.787 / 2.779 | 2.757 / 2.751 | 0.164 / 0.155 | 0.99 / 0.99 |
| `prototype` | 2.787 / 2.780 | 2.756 / 2.752 | 0.167 / 0.157 | 0.99 / 0.99 |
| `sync+agglomerative` | 2.794 / 2.788 | 2.765 / 2.762 | 0.188 / 0.179 | 0.97 / 0.97 |
| `sync+hdbscan` | 2.788 / 2.782 | 2.758 / 2.753 | 0.178 / 0.168 | 0.98 / 0.98 |
| `sync+prototype` | 2.788 / 2.781 | 2.758 / 2.753 | 0.179 / 0.169 | 0.98 / 0.98 |
| `normalised+sync` | 2.785 / 2.783 | 2.759 / 2.756 | 0.157 / 0.155 | 0.98 / 0.98 |
| `normalised+sync+agglomerative` | 2.795 / 2.793 | 2.772 / 2.769 | 0.170 / 0.169 | 0.98 / 0.98 |
| `window` | 2.789 / 2.781 | 2.757 / 2.754 | 0.178 / 0.167 | 0.98 / 0.98 |
| Constant 3 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 | 1.00 / 1.00 |
| *Self-report labels* | 2.885 | 3.000 | 0.895 | |
| *Median of expert labels* | 2.895 | 3.000 | 0.510 | |

**Metrics** (train / test)

| | MAE | RMSE | AUC* | Spearman | Macro F1** | Accuracy** |
|---|---|---|---|---|---|---|
| `base` | 0.639 / 0.648 | 0.787 / 0.802 | 0.568 / 0.541 | 0.188 / 0.107 | 0.133 / 0.129 | 0.433 / 0.431 |
| `normalised` | 0.641 / 0.649 | 0.793 / 0.803 | 0.563 / 0.542 | 0.161 / 0.108 | 0.128 / 0.126 | 0.432 / 0.430 |
| `sync` | 0.636 / 0.649 | 0.783 / 0.802 | 0.575 / 0.541 | 0.205 / 0.108 | 0.140 / 0.132 | 0.434 / 0.431 |
| `agglomerative` | 0.635 / 0.648 | 0.783 / 0.801 | 0.578 / 0.546 | 0.211 / 0.118 | 0.140 / 0.133 | 0.435 / 0.430 |
| `hdbscan` | 0.638 / 0.648 | 0.787 / 0.801 | 0.569 / 0.541 | 0.190 / 0.108 | 0.133 / 0.129 | 0.433 / 0.431 |
| `prototype` | 0.638 / 0.648 | 0.787 / 0.801 | 0.571 / 0.542 | 0.193 / 0.109 | 0.134 / 0.130 | 0.433 / 0.431 |
| `sync+agglomerative` | 0.633 / 0.648 | 0.779 / 0.802 | 0.583 / 0.545 | 0.222 / 0.117 | 0.146 / 0.136 | 0.436 / 0.430 |
| `sync+hdbscan` | 0.635 / 0.648 | 0.783 / 0.802 | 0.576 / 0.541 | 0.206 / 0.109 | 0.141 / 0.133 | 0.435 / 0.431 |
| `sync+prototype` | 0.635 / 0.649 | 0.782 / 0.802 | 0.577 / 0.541 | 0.208 / 0.108 | 0.141 / 0.133 | 0.435 / 0.431 |
| `normalised+sync` | 0.638 / 0.650 | 0.788 / 0.805 | 0.570 / 0.539 | 0.182 / 0.100 | 0.135 / 0.129 | 0.435 / 0.431 |
| `normalised+sync+agglomerative` | 0.634 / 0.649 | 0.784 / 0.805 | 0.579 / 0.543 | 0.205 / 0.109 | 0.140 / 0.132 | 0.436 / 0.431 |
| `window` | 0.635 / 0.649 | 0.783 / 0.804 | 0.576 / 0.539 | 0.204 / 0.103 | 0.141 / 0.133 | 0.435 / 0.430 |
| Constant 3 | 0.617 / 0.615 | 0.840 / 0.836 | 0.500 / 0.500 | - / - | 0.120 / 0.120 | 0.428 / 0.428 |
| *Mean of experts* | 0.668 | 0.860 | 0.563 | 0.143 | 0.208 | 0.447 |
| *Median of experts* | 0.638 | 0.913 | 0.543 | 0.120 | 0.222 | 0.451 |

### Self-report entries

**Profile** (train / test)

| | Mean | Median | SD | Share predicted 3 |
|---|---|---|---|---|
| `base` | 2.869 / 2.866 | 2.861 / 2.862 | 0.092 / 0.088 | 1.00 / 1.00 |
| `normalised` | 2.865 / 2.871 | 2.870 / 2.876 | 0.111 / 0.109 | 0.99 / 0.99 |
| `sync` | 2.871 / 2.874 | 2.869 / 2.874 | 0.147 / 0.137 | 0.99 / 0.99 |
| `agglomerative` | 2.879 / 2.884 | 2.873 / 2.879 | 0.133 / 0.128 | 1.00 / 1.00 |
| `hdbscan` | 2.869 / 2.867 | 2.861 / 2.863 | 0.099 / 0.093 | 1.00 / 1.00 |
| `prototype` | 2.869 / 2.871 | 2.864 / 2.871 | 0.109 / 0.101 | 1.00 / 1.00 |
| `sync+agglomerative` | 2.879 / 2.887 | 2.877 / 2.884 | 0.161 / 0.152 | 0.98 / 0.99 |
| `sync+hdbscan` | 2.871 / 2.875 | 2.869 / 2.873 | 0.149 / 0.139 | 0.99 / 0.99 |
| `sync+prototype` | 2.873 / 2.877 | 2.870 / 2.877 | 0.150 / 0.140 | 0.98 / 0.99 |
| `normalised+sync` | 2.872 / 2.878 | 2.877 / 2.879 | 0.151 / 0.149 | 0.98 / 0.99 |
| `normalised+sync+agglomerative` | 2.881 / 2.891 | 2.883 / 2.893 | 0.163 / 0.163 | 0.98 / 0.98 |
| `window` | 2.866 / 2.865 | 2.859 / 2.859 | 0.125 / 0.127 | 1.00 / 0.99 |
| Constant 3 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 | 1.00 / 1.00 |
| *Self-report labels* | 2.885 | 3.000 | 0.895 | |
| *Median of expert labels* | 2.876 | 3.000 | 0.428 | |

**Metrics** (train / test)

| | MAE | RMSE | AUC* | Spearman | Macro F1** | Accuracy** |
|---|---|---|---|---|---|---|
| `base` | 0.693 / 0.709 | 0.891 / 0.912 | 0.546 / 0.483 | 0.109 / -0.046 | 0.118 / 0.117 | 0.414 / 0.410 |
| `normalised` | 0.684 / 0.700 | 0.888 / 0.905 | 0.590 / 0.533 | 0.151 / 0.039 | 0.124 / 0.120 | 0.418 / 0.413 |
| `sync` | 0.685 / 0.704 | 0.880 / 0.906 | 0.586 / 0.524 | 0.187 / 0.037 | 0.127 / 0.122 | 0.419 / 0.413 |
| `agglomerative` | 0.687 / 0.708 | 0.885 / 0.912 | 0.578 / 0.504 | 0.164 / -0.003 | 0.119 / 0.118 | 0.414 / 0.411 |
| `hdbscan` | 0.692 / 0.708 | 0.890 / 0.910 | 0.555 / 0.493 | 0.125 / -0.025 | 0.118 / 0.117 | 0.414 / 0.410 |
| `prototype` | 0.691 / 0.705 | 0.889 / 0.909 | 0.557 / 0.500 | 0.124 / -0.016 | 0.119 / 0.118 | 0.414 / 0.411 |
| `sync+agglomerative` | 0.683 / 0.709 | 0.876 / 0.910 | 0.597 / 0.517 | 0.213 / 0.026 | 0.128 / 0.121 | 0.419 / 0.411 |
| `sync+hdbscan` | 0.685 / 0.705 | 0.879 / 0.906 | 0.589 / 0.525 | 0.192 / 0.040 | 0.127 / 0.121 | 0.419 / 0.412 |
| `sync+prototype` | 0.685 / 0.704 | 0.880 / 0.907 | 0.588 / 0.524 | 0.189 / 0.035 | 0.128 / 0.121 | 0.419 / 0.412 |
| `normalised+sync` | 0.679 / 0.703 | 0.879 / 0.905 | 0.610 / 0.542 | 0.206 / 0.067 | 0.129 / 0.124 | 0.421 / 0.415 |
| `normalised+sync+agglomerative` | 0.678 / 0.707 | 0.875 / 0.910 | 0.617 / 0.531 | 0.229 / 0.048 | 0.131 / 0.124 | 0.422 / 0.413 |
| `window` | 0.691 / 0.716 | 0.886 / 0.918 | 0.565 / 0.478 | 0.151 / -0.048 | 0.120 / 0.118 | 0.415 / 0.409 |
| Constant 3 | 0.666 / 0.669 | 0.908 / 0.908 | 0.500 / 0.500 | - / - | 0.117 / 0.116 | 0.413 / 0.410 |
| *Mean of experts* | 0.689 | 0.884 | 0.619 | 0.241 | 0.179 | 0.435 |
| *Median of experts* | 0.661 | 0.914 | 0.609 | 0.213 | 0.179 | 0.439 |

## Appendix C: how much do the features tell rating situations apart?

*After the run, not pre-registered. `python -m scripts.clustering.feature_similarity --out
outputs/experiments/10_clustering_v2` (script at commit `87b40e5`; results
`feature_similarity.json`, run on 28 November, commit `49c4c05`).*

**The question.** When two entries have nearly the same feature vector, do they get the same
rating? Each entry's vector is its window's model inputs, standardised. Its rating is compared with
the rating of its nearest neighbour, and that agreement with the agreement of a **reference**: a
random partner that shares everything with the neighbour except the feature match. Only the
difference between the two counts.
- **Between students:** the nearest entry in another group (other students, other recordings),
  against a random entry of another group; also with both rated by the entry's own labeler.
- **Classmates:** where the nearest is a classmate, against a random classmate at the same time gap.
- **Within a session:** the student's own nearest moment at least 20 s away, against a random moment
  of the same session at the same time gap (nearby moments agree in rating through time alone).

Each entry counts once (one camera per student-session; 99.7% of entries come from the student's main
camera), and 15 entries with a missing value are left out. The feature spaces are `base` (experiment
10's `base` arm), `normalised` and `base+agglomerative`, each also **learned**: every column weighted
by the size of its ridge coefficient predicting the rating, fitted on the other two groups only, so
that "near" means alike in what predicts ratings for other students. Intervals: a cluster bootstrap
over class sessions, computed two ways. The two-way version (each pair weighs its entry's class
times its partner's) is valid but about 1.5 times too wide on synthetic data with known answers; the
one-sided version (the entry's class only) covered only 87–93% there where pairs share dependence (a
shared moment, drift in time). A difference is **firm** if the
two-way interval excludes 0, **borderline** if only the one-sided one does.

**The labels' own agreement, the yardstick** (Spearman, 95% interval):

| | Experts' labels | Self-report |
|---|---|---|
| Another expert's rating in force at the same moment, same student | 0.089 [0.041, 0.134] | – |
| The mean of the other experts in force (experiment 06: 0.143) | **0.150** [0.076, 0.201] | – |
| The mean of the experts in force (experiment 06: 0.241) | – | **0.244** [0.151, 0.319] |

**Experts' labels**, nearest neighbour minus its reference (two-way interval; verdict):

| Space | Another group | Another group, same labeler | Within the session |
|---|---|---|---|
| `base` | −0.002 [−0.040, +0.030] none | +0.031 [−0.019, +0.077] none | **+0.047 [+0.013, +0.073] firm** |
| `base@learned` | +0.009 [−0.026, +0.042] none | +0.029 borderline | **+0.032 [+0.003, +0.059] firm** |
| `normalised` | +0.001 [−0.042, +0.025] none | +0.013 none | **+0.044 [+0.005, +0.068] firm** |
| `normalised@learned` | −0.004 [−0.044, +0.032] none | +0.027 borderline | **+0.046 [+0.017, +0.065] firm** |
| `base+agglomerative` | +0.014 [−0.017, +0.050] none | **+0.037 [+0.007, +0.065] firm** | +0.025 borderline |
| `base+agglomerative@learned` | +0.005 [−0.027, +0.034] none | +0.032 borderline | +0.030 borderline |

In `base`, the nearest entry of another group agrees at 0.006 and a random one at 0.008. Within a
session, the nearest moment agrees at 0.116 and a random moment at the same time gap at 0.069. A
classmate who looks alike agrees no more than a random classmate at the same time gap (+0.026, none).

**Self-report**, the same:

| Space | Another group | Within the session |
|---|---|---|
| `base` | +0.031 [−0.106, +0.116] none | +0.037 [−0.038, +0.116] none |
| `base@learned` | −0.006 none | **+0.076 [+0.028, +0.118] firm** |
| `normalised` | +0.008 none | +0.054 borderline |
| `normalised@learned` | +0.023 none | **+0.079 [+0.042, +0.151] firm** |
| `base+agglomerative` | +0.059 borderline | **+0.054 [+0.003, +0.106] firm** |
| `base+agglomerative@learned` | −0.007 none | +0.038 borderline |

The nearest entry of another group is 0.24 below the experts' reading of the self-report (`base`:
−0.238 [−0.344, −0.129], firm).

**The features are not degenerate.** The 35 `base` columns need 11 principal components for 80% of
their variance and 20 for 95% (the first holds 27%), and 42% of entries are someone's nearest
neighbour. **But look-alikes are common and rated as if random:** 39% of expert entries have a
near-duplicate in another group (nearer than a quarter of the median random distance; 69% in the
learned space), and such a near-duplicate's rating differs from the entry's by 0.76 on average,
against 0.79 for a random partner. The learned weights pick sensible columns (`base`, experts:
`trk.scb.m0.obj.max` 2.5, `trk.scb.read` 1.8, `trk.rf.mag_11` 1.8, `trk.scb.read.max` 1.6), and
neighbours in that space still agree at chance.

**What it shows.**
- **Between students, the features carry none of what the labelers rate.** Students who look the
  same to every feature, in another group, are rated no more alike than random students, with the
  columns weighted equally or by what predicts ratings. Two experts on the same moment agree at
  0.09–0.15, so the information is in the video, but not in these features in a form that carries
  from one student to another. The one exception is small and labeler-specific: with the clustering
  columns, the same labeler rates look-alikes slightly more alike (+0.037, firm), consistent with
  §3.3's +0.01.
- **Within a student's session, they carry a small amount** beyond what time alone gives: +0.03 to
  +0.05 for the experts, +0.05 to +0.08 for the self-report with learned weights. This is the
  within-against-between split that the re-scoring of the predictions found, by an independent
  method.
- **A null between students does not prove that no model could use the features:** this is a
  look-alike test, not a trained model, although the learned weighting narrows that gap. Ratings and
  features are autocorrelated in time, so the effective sample is smaller than the entry count; the
  intervals allow for that by resampling whole class sessions.

## Appendix D: a model trained within the session

*After the run; its test was fixed before it was run. `python -m scripts.clustering.within_session
--out outputs/experiments/10_clustering_v2` (script at commit `0ede76e`; results `within_session/`,
run on 28 November, commit `d098f25`).*

**The question.** Appendix C and §3.7 say the features carry a little information within a
student's session and none between students. Does a model trained only on the within-session part
(the rating minus that labeler's mean for that student in that session) rank a student's own moments
better than a model trained on the rating itself?

**Design.** Pooled ridge regression, one model for all experts and one for the self-report; the
ridge strength is chosen by an inner cross-validation grouped by class session, inside the training
groups only. Only labeler-sessions with at least 3 entries and 2 distinct ratings are used: 21,285 of
21,519 expert entries and 2,380 of 2,463 self-report entries. An entry seen by several cameras is
scored once. The score is **within-session Spearman**: prediction and rating each centred on their
labeler-session mean. It is also computed **detrended** (each labeler-session's linear trend in time
removed from both), so drift that features and ratings share cannot pass for skill.

| Arm | Model | Inputs | Target |
|---|---|---|---|
| W0 | experiment 10's `normalised` arm (per-labeler logistic regression) | `normalised` | rating |
| **W0′** | pooled ridge | `normalised` | rating |
| **W1** | pooled ridge | `normalised` | **deviation** from the labeler-session mean |
| W2 | pooled ridge | + the synchrony candidates | deviation |
| W3 | pooled ridge | + the student's own features over the trailing 30 s and 120 s | deviation |
| W1c | pooled ridge | **causal**: each student standardised by their own past only | deviation |
| T | pooled ridge | the time since the student's first tracked window only | deviation |

**Fixed before the run:** W1 against W0′ (only the target differs), group split, experts and
self-report, each at 98.75%. A gain counts only if it also holds detrended, W1 beats T, and it holds
for each of the three labelers with the most entries.

**Results**, group split, within-session Spearman (95% interval):

| | Experts | Self-report |
|---|---|---|
| Human ceiling (the other experts' ratings in force, on the same entries) | 0.137 [0.083, 0.180] | 0.271 [0.219, 0.323] |
| W0, the current model | 0.079 [0.052, 0.108] | 0.058 [0.005, 0.110] |
| **W0′** | **0.075** [0.043, 0.102] | **0.056** [0.003, 0.108] |
| **W1** | **0.077** [0.048, 0.102] | **0.024** [−0.030, 0.078] |
| W2 | 0.077 | 0.003 |
| W3 | 0.082 | 0.037 |
| W1c (causal) | 0.076 | 0.027 |
| T (time only) | 0.011 | 0.018 |
| **W1 − W0′** (98.75%) | **+0.002 [−0.003, +0.007]** | **−0.032 [−0.059, −0.004]** |
| W1 − W0′, detrended | +0.001 | −0.032 |
| W1 − T | +0.066 [+0.043, +0.086] | +0.006 [−0.044, +0.053] |
| W3 − W1 (detrended) | +0.005 (−0.000) | +0.013 (+0.012) |
| W1c − W1 | −0.001 [−0.009, +0.005] | +0.003 [−0.028, +0.038] |

On the experts' labels, W1 against W0′ per labeler: labeler_01 +0.006, labeler_03 +0.003, labeler_04
−0.012. The activity split agrees: on the experts' labels W1 and W0′ both score 0.070, and on the
self-report W1 is lower (−0.020); there the current model, with its activity gate, is ahead of the
pooled ridge (W0 0.088, +0.018).

- **The test is not supported.** On the experts' labels, training on the deviation ties training on
  the rating (+0.002, a tight interval around 0), and labeler_04 goes the other way. On the
  self-report it is **worse** (−0.032, its interval excludes 0): a self-report session has about 10
  entries, so subtracting its noisy mean adds noise to every target.
- **The within-session skill is real, not drift.** Detrending leaves every score in place (experts
  W1 0.077 → 0.078), time alone gives 0.011, and every feature arm beats it (+0.066, firm).
- **More inputs add nothing.** Synchrony (W2) and longer context (W3) are within 0.005 of W1, and
  W3's +0.005 disappears after detrending.
- **The causal version loses nothing** (W1c − W1: −0.001): standardising each student by their own
  past only, as a live system would, ranks their moments as well as using the whole session.
- **Models reach about 55–60% of human agreement within a session** on the experts' labels
  (0.075–0.082 against 0.137), and about 20% on the self-report (0.056 against 0.271). Over all
  entries this is lower than the re-scoring of the predictions suggested (0.14–0.15 against
  0.17–0.23), which looked only at moments where two experts entered a rating within the same 4 s.

**Limit.** For 7 of the 18 group-split fits, the chosen ridge strength was the strongest tested
(10⁵), with the inner score still rising by about 0.002–0.004 from 10⁴. Stronger settings converge to
a limit, so at most a few thousandths are left, for W1 and W0′ alike.

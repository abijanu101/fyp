# Per-labeler training: first runs on the six sessions

On 16 November 2026 the training and evaluation decisions of `docs/misc/targets.md` §7 were
implemented (guide Chapter 7 §7.9) and made the default (`engagement.labelers = "per_labeler"`).
This document records the first runs, on the six full-length sessions of
`04_six_session_run.md`, and whether augmented training copies help.

**In short:** the new path runs end to end on real data (training, per-label evaluation, inference
in both prediction modes, held-out plots, the output viewer, expert tuning). Every prediction is at
chance level, as in 04: with one session per activity, no model can be judged here (04 §4). Two
things to carry forward: one expert entered most of the ratings, so expert-track metrics mostly
measure that expert's model; and on these labels the constant 3 has the lowest MAE of anything,
so MAE and accuracy must not be the headline (AUC, Spearman, balanced accuracy).

## 1. Setup

| | |
|---|---|
| Sessions | the six of 04 (one per activity, camera 01, groups rotating) |
| Config | the run's own (its cached tracks and features), new keys from `configs/deliverable3.json`; `engagement.labelers = "per_labeler"`, `labels.tau_att = 3` |
| Training | windows of 8 s centred on each rating entry; `labels.train_target = train_with_interp`, `labels.interp.mode = step`; one logistic-regression mixture (C 0.1) per labeler, per-experiment experts (configured) and a shared expert |
| Evaluation | leave-one-group-out; every held-out entry scored per label (`rating_metrics`), experts and self as separate tracks |
| Scripts | `tmp/e2e6/run_per_labeler.py` (training), `tmp/e2e6/plot_per_labeler.py`, `tmp/e2e6/infer_per_labeler.py`, `tmp/e2e6/viewer_snapshot.py`, `tmp/e2e6/tune_per_labeler.py`; output `tmp/e2e6_pl/` |

## 2. The data

4,769 rating entries had a linked track of their student in their window (4,960 rows: some
entries are seen by more than one track). By labeler:

| Labeler | Entries | Share |
|---|---|---|
| `labeler_01` | 2,817 | 59% |
| `labeler_03` | 916 | 19% |
| `self_labeling` | 467 | 10% |
| `labeler_02` | 270 | 6% |
| `labeler_04` | 181 | 4% |
| `labeler_05` | 118 | 2% |

**One expert made two thirds of the expert entries** (2,817 of 4,302). Per-label metrics pool the
entries, so the expert track mostly scores the `labeler_01` model, and that model gets 10–25 times
the training rows of the others. Whether this holds on the full dataset, and whether the expert
track should also be reported per labeler (macro over labelers), is a question for the full run.

## 3. Results without augmentation

Classified context (the gate's prediction), mean over the three folds:

| Track | Prediction | MAE | Spearman | AUC (≥ 3) | Balanced acc. | Macro F1 |
|---|---|---|---|---|---|---|
| experts | constant 3 | **0.605** | – | 0.500 | 0.500 | 0.122 |
| experts | training mean | 0.640 | – | 0.500 | 0.500 | 0.122 |
| experts | own labeler, per-experiment experts | 0.695 | 0.006 | 0.507 | 0.491 | **0.178** |
| experts | median, per-experiment experts | 0.677 | 0.001 | 0.497 | 0.484 | 0.135 |
| experts | majority, per-experiment experts | 0.666 | 0.019 | 0.505 | 0.490 | 0.125 |
| experts | own labeler, shared expert | 0.675 | 0.025 | **0.513** | 0.488 | 0.161 |
| experts | median, shared expert | 0.659 | 0.029 | 0.507 | 0.487 | 0.126 |
| experts | majority, shared expert | 0.658 | −0.023 | 0.485 | 0.492 | 0.126 |
| self | constant 3 | **0.654** | – | 0.500 | 0.500 | 0.118 |
| self | training mean | 0.689 | 0.024 | 0.502 | 0.500 | 0.118 |
| self | own labeler, per-experiment experts | 0.806 | 0.008 | 0.509 | 0.503 | 0.146 |
| self | median, per-experiment experts | 0.718 | 0.016 | 0.489 | 0.503 | 0.130 |
| self | majority, per-experiment experts | 0.712 | 0.045 | **0.523** | 0.509 | 0.124 |
| self | own labeler, shared expert | 0.751 | −0.044 | 0.489 | 0.494 | **0.165** |
| self | median, shared expert | 0.690 | 0.054 | 0.521 | **0.513** | 0.131 |
| self | majority, shared expert | 0.697 | −0.018 | 0.487 | 0.504 | 0.123 |

Readings:

- **Everything is at chance**: AUC 0.49–0.52, balanced accuracy 0.48–0.51, Spearman within ±0.06.
  Expected here: each held-out group's activities never appear in training (04 §4).
- **The constant 3 has the lowest MAE** in both tracks. The labels sit at 2 and 3 most of the time
  and a model's spread of predictions costs MAE when it carries no signal. MAE and accuracy reward
  predicting the middle; AUC, Spearman and balanced accuracy do not.
- **Own-labeler predictions have the highest macro F1** (0.15–0.18 against 0.12 for the
  constants): a labeler's own model predicts that labeler's habitual values, so some 2s are
  predicted as 2s. This is the per-labeler design doing what it is for, but at this data size it
  is habit, not signal (its AUC is at chance).
- **With the oracle context, the per-experiment rows equal the constant 3 exactly**: the held-out
  activity's expert has no training rows and predicts 0.5, read back as 3 (guide Chapter 13, O12).

## 4. Results with one augmented copy per session

One augmented copy of each session (`tmp/e2e6/extract_aug.py 1`: a frame pass per copy, about 11–12
minutes each), added to the training side of every fold (`run_per_labeler.py augmentation.copies=1`;
results `1d448bd5a00c`, against `2d4ef2c0164c` without copies). Held-out entries are the same 4,769
originals; the copies roughly double the training rows (6,550 / 5,150 / 7,438 per fold).
Own-labeler predictions, per-experiment experts, classified context:

| Track | Run | AUC | Spearman | Balanced acc. | Macro F1 | MAE |
|---|---|---|---|---|---|---|
| experts | without copies | 0.507 | 0.006 | 0.491 | 0.178 | 0.695 |
| experts | **with one copy** | **0.532** | **0.055** | **0.509** | **0.186** | 0.700 |
| self | without copies | 0.509 | 0.008 | 0.503 | 0.146 | 0.806 |
| self | with one copy | 0.509 | 0.024 | 0.521 | 0.167 | **0.770** |

Expert-track AUC per held-out group, without → with the copy: group 01 0.483 → 0.508, group 02
0.514 → 0.535, group 03 0.525 → 0.553.

- **The expert track improves in every fold** (AUC +0.022 to +0.028, Spearman 0.006 → 0.055). The
  direction is consistent, the size small, and everything is still close to chance; on six sessions
  it is a hint, not a result.
- **The self track does not change consistently** (AUC per fold +0.008, −0.011, +0.004).
- The shared expert gains less (expert AUC 0.513 → 0.520).
- Augmentation is cheap to keep on for the full experiment (`06_full_pipeline.md` uses one copy);
  whether it helps at scale is measured there.

## 5. The tools on the new path

- `plot_results`: per-session held-out labeler curves, the median, and every entry as a dot
  (`tmp/e2e6_pl/plots/2d4ef2c0164c/`). The plots make the labelers' disagreement visible: entries
  at 2, 3, 4 and 5 at the same moments.
- Inference in both prediction modes (`tmp/e2e6_pl/smoke_g1e1_{per_labeler,majority}/`): rating
  curves 2.5–3.0 on a training session; a labeler model with no rows for the activity is a flat 3.
- `output_viewer` with a per-labeler bundle: per-student predicted and labelled ratings.
- `tune_models --component expert`: the nested search runs; its oracle objective is 0.5 by
  construction here (O12).

## 6. What follows

- The full extraction is the real test (three groups per activity).
- Report the expert track also per labeler, given the imbalance of §2.
- Window length: 8 s here; the config is now 4 s (`docs/misc/targets.md` §7.2), so the full run uses 4 s.

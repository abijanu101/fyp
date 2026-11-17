# Attention targets: a taxonomy, the numbers, and a proposal

"The label" in this project and in the DIPSER literature is not one thing. Between a labeler's raw
change events and a number a model is scored against, three choices are made, usually silently:
**what one sample is** (the unit), **how the labelers are combined** (the aggregation), and
**whose ratings count** (the labelers). This document defines each choice, computes what every
combination looks like on the full dataset, places the published works and our pipeline in the
grid, and proposes a role for each combination. **The decisions are left open** (last column of
§6).

Numbers: all 483 labelled student-sessions, step labels (Chapter 13 of the guide, O25),
`python -m scripts.labels.label_baselines --grid`.

## 1. In short

> **Update 17 November 2026.** Experiment 06 Part 1 (`docs/experiments/06_full_pipeline_and_labels.md`)
> measures how well the labelers agree on these targets. For the per-label targets, a second human is
> a worse predictor than constant 3 on MAE (one other expert 0.75, the other experts' mean 0.67,
> constant 3 0.62), and the ranking ceiling is low (the other experts' mean: AUC 0.56, Spearman 0.15).
> The tables below should be read with that second reference line; the decisions in §7 stand. The
> student counts here (483) include experiments 07–09; the pipeline's six activities have 321.

- **The full grid** is 4 units × 4 aggregations × 3 labeler groups (experts, self, experts + self)
  = **48 combinations**. For self, **[unit] × {median, mean, majority, all} × self are redundant**:
  the median, mean and mode of one rating are that rating, and "all" keeps just that rating. Each
  unit's 4 self combinations are one target, so the 16 collapse to 4, leaving **36 distinct
  targets**.
- **The tamed grid** keeps two labeler groups, **experts** and **self**, kept apart rather than mixed,
  and drops per frame (identical to per second here, §4.1): 3 units × 4 aggregations for the
  experts + 3 for self = **15 targets**. Experts + self survives only where a published work needs
  it (§5).
- **The published works are per second (or per frame), not per label** (§5). Our pipeline is
  **per window × median × experts + self** (§5).
- **The aggregation changes the target more than the unit does.** The median and the majority sit
  exactly on 3 for a quarter to two thirds of all samples (the threshold knife-edge of O26); the
  mean of the experts does so for 8–24%.
- Proposal (§6): train and evaluate on **per window × mean × experts**; report against the
  literature on three per-second targets; keep self and per label for analysis.

## 2. The three dimensions

### 2.1 Unit: what one sample is

| Unit | One sample is | How the value is read | Samples (experts) |
|---|---|---|---|
| **per label** | a moment at which **any** included labeler entered a rating | every included labeler's value in force at that moment | 31,686 moments |
| **per second** | one second on a 1 Hz clock (the annotation rate: labelers saw 1 frame per second) | each labeler's value in force at that second | 140,198 s |
| **per frame** | one context-camera frame (5 fps, real timestamps, so camera dropouts are skipped) | each labeler's value in force at that frame | 650,530 frames |
| **per window** | one window of `window.length_s` seconds with `window.overlap` (the pipeline's windows; §3 uses 8 s every 4 s, the config until 16 November 2026, now 4 s) | each labeler's **time-weighted mean** over the window | 34,191 windows |

Per label weights every *entry* equally: a stretch of rapid changes contributes many samples, a long
hold during which nobody entered anything contributes one. Per second and per frame weight every
*moment* equally. Per window averages within each window first, so its values are smoother (lower
spread) and rarely integers.

**Which units need values between entries.** Labelers only enter a rating when it changes, so a
value exists *between* entries only through an interpolation; the step function (hold until the
next entry) is one, a zero-order hold, and Chapter 13 O25 lists the others. Each unit uses it
differently:

| Unit | Needs values between entries? | Where |
|---|---|---|
| per label × all, and self at any unit's entry moments | **no** | every sample is an entry as made |
| per label × median / mean / majority | **yes, for the other labelers** | at one labeler's entry, the others' values are read from their held ratings |
| per second | **yes** | 76.3% of seconds contain no entry from anyone; the value in force is read |
| per frame | **yes**, also within a second | as per second, at 5 fps (labels have 1-second resolution) |
| per window | **yes** | the time-weighted mean of the held ratings over the window |

**When per second would not need an interpolation.** Exactly when every second carries an
annotation: for "all", an entry from someone in every second; for a median, mean or majority, an
entry from **every combined labeler** in every second. DIPSER is far from either:

| Condition | Share of seconds that meet it |
|---|---|
| at least one entry from anyone | 24.0% (no student-session meets it for every second: 0 of 483) |
| an entry from every expert | 0.38% |
| one labeler's own entries | median 3.3% of seconds per labeler (most active: 37.7%) |

Restricting per second to actual entries would not give a distinct target. Only 23.7% of seconds
contain any entry (33,292 of 140,325, the shared session-start entry excluded); in 78.8% of those
only one labeler entered (two: 17.5%, three or more: 3.7%). Such a target is *per label* rounded
to the second, with median, mean and majority mostly taken over one labeler. The published
per-second targets hold values between entries: the dataset authors' majority is "held until the
majority changes", and Fragkiadakis et al. average every labeler's value in force at each second
(`benchmarks.md` §3.1, §3.3). Per second therefore stays defined with the hold, and every number in
§3 uses step labels; another interpolation would change the per-second, per-frame and per-window
rows, and the per label rows except "all".

A related unit appears in `benchmarks.md` §1 and the six-session analysis: **per change**, one sample
per run of an unchanged *aggregated* target. Per label counts the entries of *individual*
labelers; per change counts changes of the *combined* value. They differ when one labeler's entry
does not move the combined value.

### 2.2 Aggregation: how the labelers are combined

| Aggregation | Value | Notes |
|---|---|---|
| **median** | middle value of the labelers | with the 4 experts, the mean of the two middle ones, so x.5 values appear; with 5 (experts + self), one labeler's value |
| **mean** | average of the labelers | continuous; extreme raters pull it |
| **majority** (mode) | most frequent value | ties go to the value closest to 3, then the lower one; per window, the means are rounded first |
| **all** | no combining: every labeler's value is its own sample | the target is "what labeler *k* said"; 4 times the samples; what the per-labeler ensemble trains on |

### 2.3 Labelers

| Group | Who | Notes |
|---|---|---|
| **experts** | the student's 4 expert labelers (`labeler_01`–`03` always, plus `04` or `05`) | |
| **self** | the student's own rating | one labeler, so the aggregation is redundant: median = mean = majority = all = the student's rating; one target per unit, shown as "n/a" |
| **experts + self** | all 5, mixed | the window-grid paths' target (`labels.exclude_self = false`; the default before 16 November 2026) and the dataset authors' majority vote |

Keeping experts and self apart treats them as two different measurements: what observers saw, and
what the student reports. Mixing them makes the self-report one vote of five, which moves the
median the most (§4.3).

### 2.4 Choices outside the grid

These multiply the grid again but are documented elsewhere:

- **Interpolation** of the events (step, Gaussian, ...): Chapter 13, O25.
- **Thresholding:** a continuous value, or a class (attention ≥ `tau_att`), or a window class
  (share ≥ `tau_share`): Chapter 12, §12.1–12.2.
- **Pooling and weighting** when scoring: pooled or per subject, per window or per change:
  `benchmarks.md` §2.2, Chapter 12 §12.6.
- **Filtering:** dropping tied votes, mixed windows, frames without a face, labeler subsets
  (`benchmarks.md` §3.4).

## 3. Every target, with the constant prediction "3"

**Used** follows the decision of §7: *evaluation* = the per-label targets every metric is computed
on; *training* = the target the experts learn; *both*; – = not used; *undecided* = a candidate not yet
decided. Per
second is – (not used) for us; it stays in the table as the reference for the published numbers (§5).

Self rows appear once per unit with aggregation "n/a" (not applicable): its median, mean, majority and all are the
same target (§2.3). Mean and SD describe the target; *exactly 3* is the share of samples sitting on the value 3 (the
knife-edge of O26); RMSE and MAE are those of always predicting 3; *share ≥ 3* is the share that
would count as engaged at `tau_att = 3`.

**Accuracy and macro F1** need classes. Where the target is not a whole number (means, medians of
four, window means), it is rounded to the nearest rating, halves toward 3, and the value is marked
`*`. Rounding sends everything from 2.5 to 3.49 to 3, so a starred accuracy reads as "within half a
point of 3" and is much higher than the exact share: per second × median × experts is exactly 3 for
46.5% of seconds but rounds to 3 for 82.5%. **Starred and unstarred accuracies are not comparable.**

| Used | Aggregation | Labelers | Samples | Mean | SD | Exactly 3 | RMSE | MAE | Accuracy | Macro F1 | Share ≥ 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **per label** | | | | | | | | | | | |
| – | median | experts | 31,686 | 2.94 | 0.54 | 43.3% | 0.542 | 0.380 | 0.821* | 0.180* | 0.652 |
| – | mean | experts | 31,686 | 2.97 | 0.46 | 21.3% | 0.461 | 0.360 | 0.829* | 0.181* | 0.542 |
| – | majority | experts | 31,686 | 2.94 | 0.64 | 64.2% | 0.644 | 0.377 | 0.642 | 0.156 | 0.781 |
| evaluation | all | experts | 126,744 | 2.97 | 0.84 | 46.3% | 0.840 | 0.593 | 0.463 | 0.127 | 0.688 |
| evaluation | n/a | self | 4,444 | 3.00 | 0.98 | 39.7% | 0.977 | 0.720 | 0.397 | 0.114 | 0.649 |
| – | median | experts + self | 33,774 | 2.94 | 0.59 | 67.0% | 0.593 | 0.337 | 0.672* | 0.161* | 0.803 |
| – | mean | experts + self | 33,774 | 2.97 | 0.44 | 17.1% | 0.443 | 0.352 | 0.739* | 0.170* | 0.514 |
| – | majority | experts + self | 33,774 | 2.87 | 0.63 | 63.4% | 0.640 | 0.380 | 0.634 | 0.155 | 0.746 |
| – | all | experts + self | 168,723 | 2.97 | 0.85 | 45.9% | 0.849 | 0.601 | 0.459 | 0.126 | 0.685 |
| **per second** | | | | | | | | | | | |
| – | median | experts | 140,198 | 2.96 | 0.53 | 46.5% | 0.529 | 0.360 | 0.825* | 0.181* | 0.686 |
| – | mean | experts | 140,198 | 2.95 | 0.44 | 24.1% | 0.440 | 0.336 | 0.851* | 0.184* | 0.543 |
| – | majority | experts | 140,198 | 2.97 | 0.64 | 62.5% | 0.645 | 0.389 | 0.625 | 0.154 | 0.791 |
| – | all | experts | 560,792 | 2.95 | 0.81 | 45.9% | 0.811 | 0.580 | 0.459 | 0.126 | 0.686 |
| – | n/a | self | 87,026 | 3.00 | 0.87 | 45.0% | 0.871 | 0.620 | 0.450 | 0.124 | 0.696 |
| – | median | experts + self | 140,572 | 2.96 | 0.60 | 65.6% | 0.599 | 0.348 | 0.659* | 0.159* | 0.806 |
| – | mean | experts + self | 140,572 | 2.95 | 0.42 | 17.4% | 0.427 | 0.337 | 0.767* | 0.174* | 0.497 |
| – | majority | experts + self | 140,572 | 2.90 | 0.63 | 62.2% | 0.638 | 0.388 | 0.622 | 0.153 | 0.755 |
| – | all | experts + self | 702,265 | 2.95 | 0.82 | 45.7% | 0.819 | 0.586 | 0.457 | 0.125 | 0.684 |
| **per frame** | | | | | | | | | | | |
| – | median | experts | 650,530 | 2.96 | 0.53 | 46.3% | 0.531 | 0.362 | 0.824* | 0.181* | 0.684 |
| – | mean | experts | 650,530 | 2.95 | 0.44 | 24.1% | 0.442 | 0.338 | 0.849* | 0.184* | 0.545 |
| – | majority | experts | 650,530 | 2.97 | 0.65 | 62.3% | 0.647 | 0.391 | 0.623 | 0.154 | 0.789 |
| – | all | experts | 2,602,120 | 2.95 | 0.81 | 45.8% | 0.813 | 0.582 | 0.458 | 0.126 | 0.686 |
| – | n/a | self | 403,778 | 3.00 | 0.87 | 45.1% | 0.872 | 0.619 | 0.451 | 0.124 | 0.696 |
| – | median | experts + self | 652,157 | 2.96 | 0.60 | 65.6% | 0.599 | 0.348 | 0.659* | 0.159* | 0.804 |
| – | mean | experts + self | 652,157 | 2.95 | 0.43 | 17.4% | 0.428 | 0.337 | 0.766* | 0.173* | 0.497 |
| – | majority | experts + self | 652,157 | 2.90 | 0.63 | 62.2% | 0.638 | 0.388 | 0.622 | 0.153 | 0.754 |
| – | all | experts + self | 3,257,985 | 2.95 | 0.82 | 45.6% | 0.820 | 0.587 | 0.456 | 0.125 | 0.683 |
| **per window** | | | | | | | | | | | |
| – | median | experts | 34,191 | 2.96 | 0.47 | 24.2% | 0.470 | 0.333 | 0.778* | 0.175* | 0.523 |
| – | mean | experts | 34,191 | 2.95 | 0.41 | 8.2% | 0.408 | 0.319 | 0.808* | 0.179* | 0.452 |
| – | majority | experts | 34,191 | 2.99 | 0.59 | 67.1% | 0.591 | 0.335 | 0.671 | 0.161 | 0.829 |
| – | all | experts | 136,764 | 2.95 | 0.75 | 33.7% | 0.748 | 0.547 | 0.489* | 0.131* | 0.597 |
| – | n/a | self | 20,995 | 3.01 | 0.81 | 35.5% | 0.812 | 0.591 | 0.467* | 0.127* | 0.640 |
| – | median | experts + self | 34,285 | 2.96 | 0.52 | 48.7% | 0.526 | 0.325 | 0.692* | 0.164* | 0.674 |
| – | mean | experts + self | 34,285 | 2.95 | 0.40 | 5.1% | 0.401 | 0.322 | 0.796* | 0.177* | 0.425 |
| – | majority | experts + self | 34,285 | 2.92 | 0.59 | 66.6% | 0.590 | 0.339 | 0.666 | 0.160 | 0.793 |
| training | all | experts + self | 171,279 | 2.95 | 0.76 | 34.6% | 0.761 | 0.555 | 0.483* | 0.130* | 0.604 |

## 4. What the table shows

1. **Per frame = per second.** Every statistic agrees to within 0.003; per frame only has 4.6 times
   the samples. With step labels the two units cannot differ by more than camera dropouts, and
   those are short and spread evenly.
2. **The aggregation moves the knife-edge.** Majority puts 62–67% of samples exactly on 3 whatever
   the unit; the experts' median 24–47%; the experts' mean 8–24%. With a threshold at 3, targets
   with many exact 3s are decided by "≥" versus ">" for much of the data; the mean much less so.
3. **Adding the self-report changes the median most.** Median of experts: 46.5% exactly 3 per
   second; of experts + self: 65.6%. With five labelers the median *is* a labeler's value, and the
   self-report often supplies the middle 3; with the four experts alone, the median is often an
   x.5.
4. **Windows shrink the spread.** Per-window SDs are 0.41–0.59 for the combined expert targets
   against 0.44–0.64 per second: an 8-second (the config at the time) time-weighted mean already smooths short episodes
   away. This lowers the RMSE of a constant (0.41 for per window × mean × experts), so per-window
   numbers are not comparable with per-second ones.
5. **"All" is the noisiest expert target** (SD 0.75–0.84): it keeps every labeler's disagreement.
   Its constant RMSE (0.75–0.84) is the floor for predicting an individual expert.
6. **Self differs from the experts.** Its spread is the widest of all (SD 0.81–0.98) and its mean is
   exactly 3.0 against the experts' 2.94–2.99: students use the scale more fully than the experts
   do, and do not lean below 3.
7. **Per label favours change.** Per label × median × experts has fewer exact 3s than per second
   (43.3% vs 46.5%) and a slightly different mean (2.94 vs 2.96): moments with many entries are
   moments of change, and they get more weight.

## 5. Where the published works and our pipeline sit

| Work | Target | Plus | `benchmarks.md` |
|---|---|---|---|
| IEEE Access 2025 (dataset authors) | **per second × majority × experts + self** (one image every 2 s) | classes 1–5 | §3.1 |
| Fragkiadakis et al. 2026 | **per second × mean × experts** (target = last second of a 10 s window) | regression; frames without a face dropped | §3.3 |
| GitHub project | **per frame × majority × experts**, labelers 01–04 only | binary (1–2 / 3–5); tied frames and mixed 10-frame windows dropped | §3.4 |
| Liu et al. 2025 | unknown | | §3.2 |
| **This pipeline (training labels)** | **per window × median × experts + self** | thresholded at `tau_att`, then share ≥ `tau_share` | Chapter 13 |
| `scripts/labels/label_baselines.py` default output | per second × {mean × experts, majority × experts + self, majority × experts, all × experts + self} | | §1 |
| Krippendorff's α (`inspect_labels`) | per window midpoint × all × experts + self, and × experts (point values, not window means) | | Chapter 13, O22 |

No published work uses per label. To compare with them, the unit has to be per second (or per
frame, which is the same here). Only the dataset authors mix in the self-report; in the tamed grid
that target is kept for that one comparison.

## 6. Proposal: a role for each target

Four roles:

- **Literature**: report alongside our own numbers, to compare with a published work.
- **Train / evaluate**: candidates for the target the experts learn and the retention curve is
  scored against.
- **Analysis**: useful for studying the labels themselves (agreement, labeler habits, the
  self-report), not as a model target.
- **Drop**: redundant or ill-defined.

### 6.1 Experts

| Unit | Aggregation | Proposed role | Reasoning | Your decision |
|---|---|---|---|---|
| per window | mean | **Train / evaluate (proposed)** | continuous; only 8% exactly 3, so thresholds are stable; the same aggregation as Fragkiadakis, over our windows; supports a continuous target (04 §7, direction 3) | |
| per window | median | Train / evaluate (candidate) | the current target without the self-report; 24% exactly 3; robust to one extreme expert | |
| per window | all | Train / evaluate (labeler ensemble) | what `engagement.labelers = "ensemble"` trains on | |
| per window | majority | Drop | needs rounding of window means; 67% exactly 3 | |
| per second | mean | **Literature** (Fragkiadakis) | their regression target | |
| per second | majority | **Literature** (GitHub, nearest) | their classification target, before their filtering | |
| per second | median | Analysis | the per-second counterpart of the window median, to see what windowing changes | |
| per second | all | Analysis | the floor for predicting an individual expert; labeler habits | |
| per label | mean, median | Analysis, or a second evaluation view | change-weighted versions of the targets above (§6.4) | |
| per label | majority | Analysis | agreement at the moments labelers act | |
| per label | all | **Analysis** (the raw data) | every entry as made: distributions, habits, timing (O25) | |
| per frame | any | Drop | identical to per second (§4.1) | |

### 6.2 Self

One row per unit: the aggregation is redundant for a single labeler (§2.3).

| Unit | Proposed role | Reasoning | Your decision |
|---|---|---|---|
| per window | Analysis, or a separate target | if the student's own report should be predicted, this is its window form; with the widest spread (SD 0.81) it is the hardest target | |
| per second | Analysis | how students rate themselves against the experts | |
| per label | Analysis | when students chose to change their rating | |
| per frame | Drop | identical to per second | |

### 6.3 Experts + self

| Unit | Aggregation | Proposed role | Reasoning | Your decision |
|---|---|---|---|---|
| per second | majority | **Literature** (IEEE Access) | the dataset authors' target; the only reason to keep a mixed target | |
| per window | median | Drop after the switch | the window-grid paths' target (the default before 16 November 2026); replaced by an experts target if §6.1 is accepted (config `labels.exclude_self = true` covers the median; the mean would need a code change) | |
| all others | | Drop | mixing two different measurements; no published use | |

### 6.4 On per label

Per label answers "how well does the model do at the moments when a labeler acted?". That
emphasises change and is close to the *per change* evaluation of `benchmarks.md` §1, a useful
second view for the retention curve, whose purpose is change. But no published work uses it, so
numbers computed per label cannot be compared with theirs; and as a training target it gives most
weight to stretches where labelers were busy, which are not necessarily where students changed
most. Proposed: per label as an additional evaluation view, per second for comparison with the
literature, per window for training.

## 7. Decision (16 November 2026)

- **Training: per window centred on each entry, on the labeler's interpolated curve** (the
  interpolation chosen in the config: step, Gaussian, ...; Chapter 13 O25 and the label viewer show
  the candidates; details in §7.1 items 1 and 6).
- **Evaluation metrics: per label.** Every metric is computed at the moments labelers actually
  entered a rating, so no interpolation enters the score.

Per second is kept only as a reference for the published numbers (§5); it is not used for training
or evaluation.

### 7.1 Settled (16 November 2026)

1. **Training: per window, on interpolated labels.** A labeler's value for a window is the mean of
   that labeler's interpolated per-frame values over the window's frames (interpolation from the
   config). The model keeps its window features. The window length k is a config parameter
   (`window.length_s`, with `window.overlap`), not a property of the design; 8 s (the value until 16 November 2026)
   was one value of it, and too long for this target; it is now 4 s (§7.2).
2. **Evaluation: per label × all, for the experts and for self, equally weighted.** Each entry as a
   labeler made it (no aggregation, no interpolation), scored against the prediction for that
   student. Expert entries and self entries are two separate evaluation tracks.
3. **The shared session-start entry is excluded** from scoring.
4. **The training target is per window × all × experts + self**: every labeler's own window value,
   the four experts' and the student's, is a training label; no median, mean or majority is taken.
   Each labeler's labels train that labeler's own model (item 5); evaluation keeps experts and self
   as separate tracks.
5. **No averaging in training: one model per labeler, each on its own (labeler, window) rows.**
   Every training row is one labeler's value for one student-window, used once, by that labeler's
   model; no median, mean or pooling of labelers enters any training label. Models: one per expert
   labeler (`labeler_01`–`05`; `04` and `05` each rated only part of the students) and one for the
   self-report (trained on every student's own ratings). Combining happens only at display time
   (item 8).
6. **Training windows are centred on each label.** Every rating entry (session-start entry excluded)
   gets its own window of length k centred on the entry's time, even where such windows overlap;
   the training row is the rated student's track features in that window (the student needs a
   linked track at that moment; one row per camera that sees them). This replaces the fixed window
   grid for training only; inference keeps the regular grid, with features computed the same way.
   It also settles how evaluation matches an entry to a prediction: a window centred on the entry
   (§7.3). **The training value is configurable**, config `labels.train_target`:

   | Value | Training value of a label-centred window |
   |---|---|
   | `train_with_interp` (preferred) | that labeler's interpolated values over the window (option (b)) |
   | `train_vanilla` | the entry's own rating, as entered (option (a)); interpolation plays no part in training |

   `train_with_interp` is the preferred setting: the blend of the previous and the new value
   acts as a form of data augmentation, and the interpolation setting controls how much of the
   previous value bleeds in. **The target must stay soft**: with step labels and no other entry
   nearby the window's mean is the midpoint of the previous and the new rating, so cutting it at
   `tau_att = 3` would label every 2 → 3 entry "not engaged" (2.5). `train_vanilla` has no such
   issue: its value is a whole rating. The soft target is the normalised rating of item 7.
   Consequence to keep in mind: training sees only moments at which someone entered a rating (24% of seconds contain an entry, mostly at changes),
   while inference covers every window.

7. **The prediction is a normalised rating.** Each model is trained on the soft label
   (rating − 1) / 4 in [0, 1] (for `train_with_interp`, the window's interpolated mean, so a 2 → 3
   window is 0.375, not a hard class). Its output *p* is read back as the rating **1 + 4p**; metrics
   that treat the problem as classification **round** it to the nearest rating. **Clipping
   everywhere, even where it looks redundant:** targets to [0, 1], predictions to [0, 1], ratings
   read back to [1, 5]. Ratings outside 1–5 in the source files (at least one self-report of 7,
   `group_01/experiment_03/subject_18`, likely an emotion code) are already dropped when parsed, so
   the previous rating keeps holding (Chapter 13, §13.1); the clips are a second guard.
8. **The retention curve shows every labeler.** One curve per expert model (same colour), one for
   the self model (another colour), and the **median** of them as a thicker black curve; the curves
   read as "how these DIPSER labelers would have rated this class".

9. **Two prediction modes, one set of models** (config `output.prediction`):

   | Value | Final prediction |
   |---|---|
   | `per_labeler` | every labeler model's own prediction, shown as separate curves with their median (item 8) |
   | `majority` | per student and window, each labeler model's prediction quantised to `output.majority_resolution` (0.5: 1.0, 1.5, …, 5.0), then the most common value across the models (ties: the value closest to 3, then the lower one), averaged over the window's students; one final curve (the labeler curves are still drawn) |

   Both keep the labelers' disagreement visible instead of averaging it away in training. Evaluation
   scores both modes every run; the setting only chooses what inference outputs.

### 7.2 The window length (k): 4 s (decided 16 November 2026)

How much of a labeler's moment-to-moment variation survives averaging over a window (each
labeler's step curve at 5 fps, non-overlapping windows, all labelers of all 483 student-sessions):

| Window | Label variance kept by the window means |
|---|---|
| 1 s | 95.9% |
| 2 s | 87.7% |
| 3 s | 82.1% |
| 4 s (current) | 77.7% |
| 6 s | 70.7% |
| 8 s (previous config) | 65.6% |

Labels have 1-second resolution, so 1–2 s is the floor on the label side. On the feature side a
shorter window means fewer frames per feature (20 frames at 4 s, 10 at 2 s, at 5 fps), a track
needs `aggregation.min_coverage` (50%) of the window to be scored, and the number of windows grows
in proportion (overlap 50%). The feature pass reruns from cached tracks in about a second per
session, so trying several lengths is cheap.

### 7.3 Questions raised before implementing

- **Matching an entry to a prediction.** Settled by item 6: a window centred on the entry, with
  features computed for that window (the feature pass runs from cached tracks in about a second
  per session).
- **Scoring the retention curve.** The displayed curve is per labeler plus their median (item 8);
  per-label metrics score each labeler's model on that labeler's entries. `retention_r` has been
  removed from the code; whether a curve-shape metric comes back, and on which curve, is in
  `docs/misc/proposed_metrics.md`.

### 7.4 Implementation (16 November 2026)

Items 1–9 are implemented and are the default (`engagement.labelers = "per_labeler"`); the guide's
§7.9 describes the code. `plot_results`, `output_viewer` and `tune_models` support it (Chapter 8). Config keys: `labels.train_target`, `labels.interp.{mode,ramp_s,sigma_s}`,
`output.prediction`, `output.majority_resolution` (Chapter 9). The window length is 4 s (`window.length_s`, §7.2). Still open: a curve-shape metric (`docs/misc/proposed_metrics.md`).
A first run on the six-session caches (`scripts/analysis/e2e6/run_per_labeler.py`, 8 s windows, step labels,
`train_with_interp`) works end to end: 4,769 entries with a linked track, every prediction at chance
level, as expected from one session per activity (04 §4).


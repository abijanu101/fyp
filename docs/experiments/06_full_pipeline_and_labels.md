# The labels and the full pipeline

*Frozen 18 November 2026: no further changes; later results go in later experiments. Done
2026-11-17; Part 2 re-run 18 November with training-set, prediction-spread and per-labeler scores.
Config: [`06_full_pipeline_and_labels.json`](06_full_pipeline_and_labels.json); runs and logs:
`outputs/experiments/06_full_pipeline/`.*

*Part 1: attention is the student's internal state, so the self-report is the reference. Expert
observers recover it only weakly (AUC 0.61 at best, 0.63 combined), and no observer beats constant 3
on MAE. Part 2: the full pipeline barely ranks entries (expert-entry AUC 0.53) and loses to always
predicting 3 on MAE; it underfits, with every classical expert model. Part 1 comes first because it
decides how Part 2 is run and scored: per labeler, against the human ceiling.*

**In short:**

- **Part 1 — the labels.** The self-report is the only direct report of a student's attention,
  so it takes precedence (§1.6). Scored like the model, each expert predicts the student's own
  rating at AUC 0.53–0.61. Every expert's MAE is worse than constant 3's, and their median only ties
  it (0.663 against 0.670). Of 177 moments a student reports a 5, the experts' mean says 5 at none.
  The experts agree with each other even less: Krippendorff's α is between −0.06 and +0.04 on every
  unit and interpolation, and one expert predicts another at AUC 0.534 (Appendix A).
- **Part 2 — the model.** The per-labeler pipeline at full scale (52 session-cameras, one augmented
  copy, BoT-SORT + ReID + CMC, a single decoder per labeler) predicts 3 for 96% of expert entries:
  AUC 0.527, Spearman 0.07, MAE 0.674 against 0.619 for constant 3. On its own training groups it
  reaches only AUC 0.57: it underfits.
- **Reading.** On the experts' labels the model sits about halfway between chance and the experts'
  ceiling. On the self-report it is at chance (AUC 0.508), where the experts are not (0.61). "Beat constant 3
  on MAE" is not a goal the labels allow at the moment level. Interpolation decides whether labelers
  can be compared at all, and step (or a short ramp ending at the entry) is the reading that agrees
  best; a Gaussian that spreads a change around its entry agrees worse. FYP-2 needs a target humans
  agree on, not a stronger model of this one.

---

# Part 1 — the labels

## 1.1 Why this part

Part 2's model loses to "always 3" on MAE by 0.055 on a 1–5 scale and ranks entries barely better
than chance. That is a bad result only if the labels allow much better. A **ceiling** answers that:
how well can the labels be predicted by anything, measured the way the model is measured. The fairest
ceiling uses the labels themselves, other humans predicting a human, on the model's unit (each rating
entry), the model's entries (the 21,386 with a linked track), and the model's metrics
(`rating_metrics`: MAE, accuracy and macro F1 on the 1–5 rating, AUC and balanced accuracy at
rating ≥ 3, Spearman), summarised the same way (mean over the three groups).

Script: `python -m scripts.labels.label_agreement all --model-entries` (`scripts/labels/label_agreement.py`,
about 9 minutes; every table below is in `outputs/labels/agreement.json`). Scope: the six activities
of the pipeline (experiments 01–06), all three groups: **321 student-sessions** (each student once per
group and activity). Earlier documents quote 483: that counts all nine DIPSER experiments.

## 1.2 How DIPSER was labelled

Each student-session has four expert labelers (labeler_01–03 always, labeler_04 or labeler_05) and the
student's own self-report (`self_labeling`). A labeler's file is a list of **change points**: a rating
1–5 and the moment it was entered. The entries were made offline, frame by frame at one frame per
second: 83–91% of every labeler's entries fall in the first tenth of a second. Every labeler's first
entry is at the same instant, the session start; it is a starting judgement, not a moment the labeler
chose, and is excluded everywhere below (as in training).

| Labeler | Student-sessions | Entries | Share of expert entries | Mean | Median | SD | Shares 1 / 2 / 3 / 4 / 5 | Entries per min | Median run | Runs ≤ 2 s | Tracks that never change |
|---|---|---|---|---|---|---|---|---|---|---|---|
| labeler_01 | 321 | 14,258 | 61.5% | 2.73 | 3 | 0.73 | 0.00 / 0.41 / 0.47 / 0.10 / 0.02 | 8.9 | 3.0 s | 36% | 0% |
| labeler_02 | 321 | 1,263 | 5.4% | 2.88 | 3 | 0.74 | 0.00 / 0.33 / 0.47 / 0.19 / 0.01 | 0.8 | 13.0 s | 15% | 49% |
| labeler_03 | 321 | 5,702 | 24.6% | 2.85 | 3 | 0.98 | 0.00 / 0.46 / 0.32 / 0.11 / 0.10 | 3.6 | 3.1 s | 34% | 8% |
| labeler_04 | 178 | 1,370 | 5.9% | 2.89 | 3 | 0.85 | 0.00 / 0.37 / 0.42 / 0.16 / 0.05 | 1.5 | 9.0 s | 18% | 14% |
| labeler_05 | 143 | 586 | 2.5% | 3.03 | 3 | 0.88 | 0.00 / 0.32 / 0.39 / 0.24 / 0.05 | 0.8 | 24.0 s | 7% | 14% |
| **all experts** | 321 | 23,179 | 100% | 2.78 | 3 | 0.81 | 0.00 / 0.42 / 0.43 / 0.11 / 0.04 | | | | |
| self_labeling | 321 | 2,634 | – | 2.88 | 3 | 0.90 | 0.01 / 0.37 / 0.41 / 0.14 / 0.07 | 1.6 | 9.5 s | 14% | 11% |

Shares are of entries (session start excluded); runs are the time a value holds; SD is the
population SD of the entered ratings. What stands out:

- **The true distribution is narrow and centred just below 3.** Every labeler's median is 3, and
  the SD is 0.73–0.98. 85% of expert entries are a 2 or a 3. Two consequences follow:
  - **Constant 3 is the MAE-optimal constant for *every* labeler,** not just in the pool.
  - **The label SD (0.81 for the experts) is the scale for a model's prediction spread.** A model
    that predicted the labels perfectly would match it. The squared-error-optimal spread is r ×
    0.81, where r is the model's correlation with the labels, so with weak signal a good model
    predicts with a much smaller SD. Part 2 reports both.

- **One labeler dominates.** labeler_01 made 61.5% of the expert entries, at 8.9 a minute, with
  values that last a median 3 s. Pooled numbers are mostly about labeler_01.
- **Labelers have different styles.** labeler_02 never changes the rating for 49% of students;
  labeler_05 changes it every 24 s; labeler_03 uses 5 ten times as often as labeler_01.
- **They start from different levels.** The session-start rating is 3 for 89% of students by
  labeler_02 and 2 for 70% by labeler_03 (labeler_01: 58% 3, 32% 2).
- **Every entry is a change.** No entry repeats its labeler's previous value.
- **Ratings flip between neighbours.** 1s almost never occur; most entries move between 2 and 3.

By activity, expert entries per minute range from 1.8 (activity 01) to 5.4 (02), and the share of 4s
and 5s from 4% (03) to 21% (02 and 05).

## 1.3 What "agreement" can mean, and why interpolation is unavoidable

Two labelers can only be compared at moments where both have a value. But a labeler states a value
only at their entries, and **the entries of different labelers rarely coincide**:

| Expert entries with another expert's entry on the same student within | ±0.5 s | ±2 s | ±5 s | ±10 s |
|---|---|---|---|---|
| share of expert entries | 32% | 52% | 66% | 75% |

(Self entries: 49%, 78%, 91%, 96%, against any expert.) So for two thirds of the moments one expert
rated, no other expert said anything within a second. Any comparison has to decide what a labeler's
rating *is* between their entries: an **interpolation**. The literal record is the **step**: a value
holds from its entry until the next. Every agreement number depends on that choice, and on three
more:

| Choice | Options here | Why it matters |
|---|---|---|
| **Unit** (what is compared) | each entry moment; each second; each 4 s window's midpoint; each 4 s window's mean | per second and per window count a long-held value many times; per entry counts each judgement once, where labelers changed their mind |
| **Interpolation** | step, ramp, Gaussian (§1.4) | decides each labeler's value at the compared moments |
| **Labeler set** | experts; experts + self | the self-report is a different kind of judgement |
| **Measure** | Krippendorff's α (§1.5); predicting one labeler from others (§1.6); correlations of averages (§1.7) | α is a symmetric agreement index; prediction is the model's own yardstick |

## 1.4 Interpolation: what each mode assumes

`src/labels/interp.py`; the label viewer (`scripts/data_visualization/label_viewer.py`) draws each.

| Mode | Value between entries | What it assumes about the moment a change happened |
|---|---|---|
| **step** | the last entered value | the student changed exactly when the labeler typed it |
| **ramp w** | step, but each change is a straight ramp over the w seconds *before* the entry | the change happened during the preceding w seconds and the labeler entered it when it was complete |
| **gauss σ** | the step blurred by a Gaussian of width σ: each change an S-curve *centred* on the entry | the true moment is uncertain by about ±σ, as likely before the entry as after |

A ramp of width w and a Gaussian of σ = w/√12 spread a change equally (ramp 3 s ≈ σ 0.87 s). The
viewer's other modes (linear, monotone cubic, spline) and its decay and stale rules are covered in
guide 13 O25 (`outputs/labels/interp_compare_grid.md`, 8 s windows).

What the modes do to agreement (§1.5, §1.6):

- **On α they barely matter** (overall, experts, raw ratings): entry moments step 0.010, ramp 0.013,
  Gaussian σ 1 / 2 / 4 s −0.007 / −0.009 / −0.015; per second step −0.016, ramp −0.013, Gaussian
  −0.017 / −0.020 / −0.025.
- **On prediction they matter, in one direction**: one other expert predicting an entry gets AUC
  0.534 with step, 0.537 with ramp 3 s or 7 s, and 0.513 with any Gaussian; the mean of the other
  experts 0.563 (step), 0.568 (ramp), 0.525–0.526 (Gaussian). A Gaussian puts half of each change
  *before* its entry, so at the moment another labeler entered something, the curve is mid-change.
- **No reaction delay to correct.** Reading the other labeler's step curve a fixed lag later or
  earlier never helps: one other expert gets AUC 0.534 at lag 0, 0.529 / 0.522 / 0.512 / 0.507 at
  +1 / +2 / +5 / +10 s, and 0.493–0.503 at −1 to −5 s.

**So the step, or a short ramp that ends at the entry, is the reading under which labelers agree
best.** It is also the training default (Part 2 §2.4 measures the training side).

## 1.5 Krippendorff's α

**What it is.** Krippendorff's α measures how much raters agree *beyond what their rating habits
alone would produce*. It compares the disagreement actually observed between raters rating the same
items with the disagreement expected if the same ratings were shuffled at random across items:

  α = 1 − (observed disagreement) / (disagreement expected by chance)

α = 1 is perfect agreement; α = 0 means raters agree no more than chance given how often each rating
is used; α < 0 means they agree *less* than chance (systematically different). It handles any number
of raters, missing ratings, and (in the ordinal version used here) treats a 2-against-3 disagreement
as smaller than 2-against-5. The usual reading is: α ≥ 0.8 reliable, 0.67–0.8 tentative, below that
not a usable consensus. Two raters can agree on 75% of items and still have α = 0, if both say "3"
most of the time; guide 12 §12.3 works through such an example.
(`src/evaluation/agreement.py: krippendorff_alpha_ordinal`, tested against Krippendorff's 2011
example.)

**The 14 November computation** (`scripts/data_visualization/inspect_labels.py`,
`outputs/labels/report.json`): step values at the midpoints of 8 s windows, per activity, raw 1–5.
`label_agreement` reproduces it exactly (activity 01: with self 0.0098, experts only −0.080; activity
02: 0.130 / 0.120; …), so the two scripts measure the same thing.

**The grid** (4 s windows; overall over 321 student-sessions; items = units where ≥ 2 labelers have a
value):

| Unit | Interpolation | Experts, raw 1–5 | Experts, binary (≥ 3) | Experts + self, raw | Experts + self, binary | Items |
|---|---|---|---|---|---|---|
| entry moments | step | 0.010 | −0.029 | 0.041 | 0.000 | 20,640 |
| entry moments | Gaussian σ 2 s | −0.009 | −0.045 | 0.033 | −0.011 | 20,640 |
| per second | step | −0.016 | −0.053 | 0.033 | −0.009 | 95,963 |
| per second | Gaussian σ 2 s | −0.020 | −0.056 | 0.031 | −0.011 | 95,963 |
| window midpoint | step | −0.016 | −0.054 | 0.032 | −0.009 | 47,508 |
| window midpoint | Gaussian σ 2 s | −0.019 | −0.057 | 0.032 | −0.011 | 47,508 |
| window mean | step | −0.018 | −0.053 | 0.032 | −0.008 | 47,508 |
| window mean | Gaussian σ 2 s | −0.020 | −0.058 | 0.031 | −0.012 | 47,508 |

By activity (window midpoint, step, experts, raw): 01 −0.078, **02 +0.122**, 03 −0.062, 04 −0.029,
05 −0.079, 06 −0.037. By group: 01 +0.041, 02 −0.036, 03 −0.102.

- **α is zero, everywhere.** No unit, interpolation, labeler set or scale gets above 0.13 (activity
  02); overall it is between −0.06 and +0.04.
- **The binary view does not rescue it** (guide 13 O22 asked): at "≥ 3" the experts' α is −0.03 to
  −0.06, *below* the raw scale. Their disagreement is not small steps on the same side of a threshold;
  it straddles 2 and 3, exactly where the threshold sits.
- **Adding the self-report raises α slightly** (+0.03 to +0.05): the student agrees with the experts a
  little more than the experts agree with each other (§1.7 shows the same at the student level).

## 1.6 The human ceiling: how well do observers read a student's own state?

Attention is an internal state. The only direct report of it is the student's own; an expert
watching the footage infers it from behaviour. So when the two disagree, **the self-report takes
precedence**, and the question that sets a ceiling is: how well does a human observer, and therefore
any model trained on observers, recover what the student reports? DIPSER's self-reports have limits
too. They were entered offline with the same tool as the experts' (§1.2), so they are a
retrospective judgement, not an in-the-moment probe. They are still the closest the dataset gets to
the student's own state.

Every self-report entry (session start excluded) is predicted from the experts' curves at that
moment and scored with `rating_metrics`, exactly like the model. Leave one group out; mean over
groups; the model's 2,445 self entries (those whose student has a linked track):

| | Constant 3 | Model (held out) | One expert | Median of the experts | Mean of the experts | Learned combination | Student's own other ratings |
|---|---|---|---|---|---|---|---|
| MAE | 0.670 | 0.720 | 0.763 | **0.663** | 0.693 | 0.690 | 0.733 |
| RMSE | 0.910 | 0.923 | 1.097 | 0.917 | 0.888 | **0.881** | 0.923 |
| AUC\* | 0.500 | 0.508 | 0.556 | 0.605 | 0.613 | **0.628** | 0.564 |
| Spearman | – | 0.007 | 0.118 | 0.206 | 0.232 | **0.235** | 0.178 |
| Macro F1\*\* | 0.116 | 0.120 | **0.257** | 0.180 | 0.176 | 0.195 | 0.210 |
| Accuracy\*\* | 0.411 | 0.413 | 0.426 | 0.439 | 0.432 | **0.453** | 0.403 |

\* threshold (AUC: ratings split at ≥ 3); \*\* discretisation (prediction rounded to a whole rating).
"One expert" pools every (entry, expert) pair. The mean of the experts is continuous like the model's
output, so it is the fair comparison for AUC and Spearman. "Learned combination" is a ridge
regression on every expert's value, leaving one group out: the best a linear reader of the experts
does. "Student's own other ratings" is the student's time-weighted mean over their other runs in the
session.

**Each expert separately** (all 2,634 self entries for the students that expert rated; step; pooled
over groups):

| Expert | AUC\* | Spearman | MAE | Constant 3's MAE, same entries | Self entries |
|---|---|---|---|---|---|
| labeler_01 | **0.609** | **0.214** | 0.690 | 0.666 | 2,634 |
| labeler_02 | 0.527 | 0.091 | 0.686 | 0.666 | 2,634 |
| labeler_03 | 0.570 | 0.154 | 0.751 | 0.666 | 2,634 |
| labeler_04 | 0.535 | 0.043 | 0.971 | 0.688 | 1,298 |
| labeler_05 | 0.536 | 0.098 | 0.861 | 0.645 | 1,336 |

**What a student reports against what the experts' mean says** (rounded, all self entries; rows:
the student's rating, columns: the experts' mean rounded):

| student \ experts | 1 | 2 | 3 | 4 | 5 | entries |
|---|---|---|---|---|---|---|
| **1** | 0 | 3 | 24 | 5 | 0 | 32 |
| **2** | 0 | 104 | 849 | 16 | 0 | 969 |
| **3** | 0 | 47 | 1,009 | 32 | 0 | 1,088 |
| **4** | 0 | 7 | 328 | 33 | 0 | 368 |
| **5** | 0 | 5 | 146 | 26 | 0 | 177 |

What this shows:

- **No expert beats constant 3 at reading the student.** Each expert's MAE against the self-report
  is above constant 3's on the same entries, by 0.02 (labeler_02) to 0.28 (labeler_04). The best
  combination of all four, their median, ties it (0.663 against 0.670).
- **The best observer ranks moments only weakly.** labeler_01 reaches AUC 0.61 and Spearman 0.21;
  two experts are near chance (AUC 0.53). Pooling all four and fitting their weights reaches AUC 0.63.
- **The experts miss the student's extremes.** Of 177 moments a student reports a 5, the experts'
  mean says 5 at none and 4 at 26. Of 969 self-reported 2s, it says 2 at 104 (11%). When a student
  reports a change in their own attention, the observers mostly still see a 3.
- **The model is at chance on the self-report** (AUC 0.508, Spearman 0.007), while the experts are
  not (0.61). The experts see something of the student's state that our features do not.
- **The student's own other ratings predict an entry less well than the experts do** (AUC 0.564
  against 0.613). Each self-report entry is a change, so the student's average over the rest of the
  session is a poor guide to it (§1.8 measures this for every labeler).
- **The experts agree with the student more than with each other.** One expert predicting another
  expert reaches AUC 0.534 and Spearman 0.08 on the model's expert entries, against 0.556 and 0.118
  for predicting the student. So the experts do not share one observer bias: if they did,
  they would agree with each other more than with the student. The expert-against-expert numbers (the ceiling for the
  experts' own labels) are in Appendix A.

This is the ceiling for the target that matters: observers recover the student's own report at AUC
about 0.6, and no observer, alone or combined, beats predicting 3 on MAE.

## 1.7 Coarser units: do labelers agree on who is engaged?

If moment-to-moment ratings are noise, averages might still agree: which students are generally more
engaged, per activity. Each labeler's time-weighted mean rating per student-session (step), compared
across labelers:

| Unit | Units | Experts, mean pairwise Spearman | Experts, mean pairwise MAE | α on rounded means | Self vs median of experts, Spearman |
|---|---|---|---|---|---|
| student × activity | 321 | 0.112 | 0.643 | −0.067 | **0.332** |
| student × 2 minutes | 642 | 0.100 | – | −0.051 | 0.292 |
| student × 1 minute | 1,284 | 0.092 | – | −0.046 | 0.277 |

By activity (student × activity, experts' pairwise Spearman): 01 0.092, 02 0.159, 03 0.167, 04 0.031,
05 −0.002, 06 0.063.

- **Averaging does not create agreement.** Experts barely agree on which students are more engaged in
  a whole session (Spearman 0.11), and α on their rounded session means is negative: each labeler's
  level (§1.2: labeler_03 starts at 2, labeler_02 at 3) differs more than students do.
- **The student agrees with the experts' consensus more than experts agree with each other**
  (Spearman 0.33 at the session level). The median of four experts averages out each one's habits;
  what remains is a weak shared signal about the student.
- **Most of the rating variance is within a student, for the frequent labelers**: the share of a
  labeler's per-second rating variance that lies *between* students is 0.33 (labeler_01), 0.32
  (labeler_03), but 0.59–0.75 for labeler_02, 04 and 05 and 0.56 for the self-report. The labelers
  who change their rating often are mostly describing fluctuations within a student, the ones others
  cannot reproduce.

## 1.8 Perspectivism: the case, our use, and its premise tested

**The case.** Machine learning usually aggregates annotators into one "gold" label (a majority, a
median) on the assumption that disagreement is noise around a true value. *Perspectivism* (data
perspectivism) argues that disagreement is often signal: annotators legitimately see the same item
differently, and aggregation erases those views. It proposes training and evaluating on each
annotator's labels instead (Cabitza, Campagner & Basile, *Toward a Perspectivist Turn in Ground
Truthing for Predictive Computing*, AAAI 2023; Basile et al., *We Need to Consider Disagreement in
Evaluation*, BPPF @ ACL 2021; Plank, *The "Problem" of Human Label Variation*, EMNLP 2022; the survey
of methods: Uma et al., *Learning from Disagreement: A Survey*, JAIR 72, 2021). Engagement is a
natural case: it is an inner state judged from outside, and the self-report is a different perspective
by construction, not a noisy expert.

**Our use of it.** The per-labeler design (`docs/misc/targets.md` §7) is perspectivist: one model per
labeler, every (labeler, window) pair its own training row, nothing averaged in training; evaluation
per entry against each labeler's own ratings, experts and self as separate tracks; the median curve is
a summary for display, not a gold label; at inference `output.prediction` gives either every labeler's
view (`per_labeler`) or their majority (`majority`).

**Its premise, tested.** Modelling each annotator separately pays off when each annotator is
*consistent with themselves* and different from the others: a stable perspective. If a labeler sees
a student in their own way, that labeler's other ratings of the student should predict their next
rating better than other people's ratings do. This is the most important test in this part, so it is
set out in full.

### 1.8.1 How self-consistency was measured

Nobody rated the same moment twice, so a labeler's agreement with *themselves* at a moment cannot be
measured. What can be measured is whether a labeler's view of a *student* carries over from one part
of the session to another.

**The predictor.** For every rating entry of an expert A on student s (session start excluded):

1. Split A's ratings of s into runs: each value held from its entry to A's next entry (the last one
   to the session's last frame), from the session start on.
2. Remove the run that this entry starts, so the entry's own value is not used to predict itself.
3. Average the remaining runs' values, each weighted by how long it held: A's **mean rating of s over
   the rest of the session** (`scripts/labels/label_agreement.py`, `ceiling_rows`, column `own_loro`).

Example: A rates s 3 from 0 to 20 s, 2 from 20 to 50 s, 4 from 50 to 80 s and 3 from 80 to 120 s. For
the entry at 20 s (the 2), the predictor is the mean of the other runs, (3 × 20 + 4 × 30 + 3 × 40) /
90 = 3.33. The entry is scored as a 2 predicted by 3.33.

**The comparison.** The same entries predicted by the other experts' curves at that moment (their
step values at the entry's time, averaged; self-reports and A excluded): what the other people saw at
that moment. Both are continuous, so ranking metrics treat them alike.

**The scoring.** Exactly as the model and the rest of §1.6: `rating_metrics` on each group's entries,
averaged over the three groups. AUC asks whether the predictor ranks entries of 3 or more above entries
of 1–2; Spearman whether it orders all entries by rating. "Pooled" puts every expert's entries
together (61.5% are labeler_01's); "per labeler" scores each expert's entries separately and averages
the five, so each labeler counts equally. All 23,179 expert entries.

### 1.8.2 Result

| Expert entries | Pooled: MAE | RMSE | AUC | Spearman | Per labeler, averaged: MAE | RMSE | AUC | Spearman |
|---|---|---|---|---|---|---|---|---|
| Constant 3 | 0.619 | 0.839 | 0.500 | – | 0.640 | 0.849 | 0.500 | – |
| A's own mean over the rest of the session | 0.664 | 0.858 | **0.573** | 0.146 | 0.689 | 0.864 | 0.562 | 0.161 |
| The other experts' mean at the moment | 0.668 | 0.860 | 0.563 | 0.143 | 0.669 | 0.863 | **0.624** | **0.243** |
| A's previous value | 1.216 | 1.315 | 0.202 | −0.474 | | | | |

(`outputs/labels/agreement.json`: `ceiling.subsets["entry/experts"]`, predictors "self-consistency: A's
mean over its other runs" and "mean of other experts, step", their `macro_over_labelers`; on the
model's 21,386 entries the pooled values are within 0.002.)

### 1.8.3 What it shows, and what it does not

- **Pooled, a labeler's own view and the others' view predict about equally** (AUC 0.573 against
  0.563). The pooled numbers are mostly labeler_01's.
- **Per labeler, the others predict better than the labeler's own view** (AUC 0.624 against 0.562,
  Spearman 0.243 against 0.161). For the four less frequent experts, what other people saw at that
  moment is a better guide to their rating than their own ratings of the same student elsewhere in the
  session.
- **The two predictors measure different things.** A's session mean is almost constant across A's
  entries on one student: it can only say whether A rates this student high or low overall. The others'
  mean is a value at the moment. That they score alike means *A's overall impression of a student is
  worth about as much as another person's view of the moment*. It does not show that labelers are
  "equally consistent with themselves and with others" in a stricter sense, which these labels cannot
  measure.
- **The own-mean predictor is pulled against the entry.** Every entry is a change, and changes swing
  back (A's previous value predicts the entry *inversely*, AUC 0.20): the runs next to an entry often
  hold the opposite value and sit inside A's session mean. This understates self-consistency somewhat.
  Leaving out a wider neighbourhood around the entry would remove it; that was not run.
- **A's mean in the student's other activities** (subject ids name the same student across a group's
  activities) would test a view that lasts beyond one session; that was not run either.

**What we can claim.** No expert shows a view of a student that predicts their own ratings better than
the other experts' view of the moment does; for most experts (averaged per labeler) the others predict
them better. The perspectivist premise, that each labeler holds a stable view worth modelling
separately, finds no support at the moment level in these labels. Whether a view lasts across
sessions, and how much the swing-back effect hides, are open.

### 1.8.4 Other evidence on the premise

- **What differs between labelers is style, not view**: rating level (labeler_03 starts at 2,
  labeler_02 at 3), how often they change (0.8 to 8.9 times a minute), how much of the scale they use.
  A per-labeler model can learn a level; that is the labeler prior, which on its own scores AUC 0.50.
- **The previous value predicts the opposite of the next entry** (AUC 0.20): a change from a 2 goes to
  a 3 and back. Moment-level entries are oscillations between neighbouring values.
- **The pooled arm tests the same premise on the model side** (Part 2 §2.3): one model trained on every
  labeler's rows, scored per labeler, against one model per labeler. It ranks entries the same (AUC
  0.526 against 0.527, Spearman 0.074 against 0.072) and is *better* on MAE (0.653 against 0.674) and
  RMSE (0.818 against 0.838): pooling gives each prediction more data and pulls it toward the shared
  mean. Separate models per labeler buy nothing here.

**So perspectivism splits in two.** As an *evaluation* stance it is necessary on DIPSER: with α ≈ 0
there is no consensus label to recover, a median or majority "gold" is an artefact of the
aggregation rule, and each labeler's ratings (with the self-report as its own track) are the only
defensible targets. As a *modelling* choice it is not supported: no labeler holds a stable view beyond
what others share (§1.8.3), and a pooled model does as well as per-labeler ones. Perspectivism tells us
*what to score against*; it does not make the moment-level target learnable.

---

# Part 2 — the model: the full pipeline

## 2.1 Setup and arms

The first full-scale training of the per-labeler pipeline (`docs/misc/targets.md` §7): one model per
labeler, trained on windows centred on that labeler's rating entries, evaluated per label against the
entries as made. Each labeler's model is a **single decoder**: one expert for every activity
(`mixture.expert_scope = "shared"`). Every arm shares the frame pass, the features, the
label-centred windows, the targets, the folds (leave one group out) and the seed, so arms are
compared entry for entry.

| Arm | Setting (on top of the config) | What it varies |
|---|---|---|
| **Per-labeler** (the config) | – | one logistic-regression model per labeler |
| **Pooled** | `engagement.labelers=pooled` | one model trained on every labeler's rows together, answering for all of them (the aggregated counterpart of the per-labeler design; Part 1 §1.8) |
| **Expert model** | `expert.name=random_forest / knn` | the classical model inside each labeler's decoder (§2.3) |
| **Training target** | `labels.interp.*`, `labels.train_target`, `window.length_s` | how a labeler's ratings become a training target (§2.4) |
| **Augmentation** | `augmentation.copies=0` | without the one augmented copy of every recording that the other arms train on (§2.3) |

The windows are 4 s (`window.length_s`); here each is scored on its own, later they are the unit a
temporal model consumes (FYP-2). The activity gate is still trained (it shades the output graph) but
has no effect on engagement with a single decoder. Whether activity-specific experts (a mixture of
experts) help is left to a separate experiment.

**Data.** 52 session-cameras plus one augmented copy each (104 frame passes, features
`outputs/cache/features/2685b0745fde/`), with D3's detector threshold (0.1098) and T2's tracker
(BoT-SORT with ReID and CMC), resolved from `configs/derived.py` and frozen in the config file.
23,831 rating entries (session-start entries excluded), of which 21,386 expert and 2,445 self
entries have the rated student linked to a track; 45,166 training rows. Per fold, 11,685–17,724
rows are held out.

## 2.2 Commands

```
# features: frame pass + feature pass for every session-camera, then the augmented copy
python -m scripts.run_pipeline extract --config docs/experiments/06_full_pipeline_and_labels.json

# the arms (each: leave-one-group-out evaluation + a final bundle)
python -m scripts.train_models --config docs/experiments/06_full_pipeline_and_labels.json                     # c5341cb6385e
python -m scripts.train_models --config docs/experiments/06_full_pipeline_and_labels.json --set engagement.labelers=pooled
python -m scripts.train_models --config docs/experiments/06_full_pipeline_and_labels.json \
    --set expert.name=random_forest 'expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20}'
python -m scripts.train_models --config docs/experiments/06_full_pipeline_and_labels.json \
    --set expert.name=knn 'expert.params={"n_neighbors": 25}'

# compare two results (hash prefixes; --plot saves confusion heatmaps to outputs/plots/)
python -m scripts.reporting.compare_results c5341cb6385e <other hash> --names per_labeler other --plot
```

The training-target arms (§2.4) are the same command with the `--set` shown in their table. Every
arm's results hash and training log are listed in `outputs/experiments/06_full_pipeline/`. Training
one arm takes about 4 minutes on the cached features (the random forest and k-NN about 20).

## 2.3 Result

Every result has two tables, one row per model or arm:
- **Profile:** the mean, median and SD of the predictions.
- **Metrics:** MAE, RMSE, AUC\*, Spearman, macro F1\*\* and accuracy\*\*.

Every data cell is **train / test**: the same fold's model scored on the original entries of its own
training groups, then on its held-out group, each the mean over the three folds. Rows in italics are
references, with one value per cell on all entries of the track (06 Part 1, `outputs/labels/agreement.json`):
- in the profile, the self-report ratings and the median of the expert ratings;
- in the metrics, the other experts' mean and median predicting the same labels (the human ceiling).

\* threshold (AUC: labels split at rating ≥ 3); \*\* discretisation (prediction rounded to a whole
rating, halves toward 3); unmarked: the raw prediction. Built with
`python -m scripts.reporting.results_table`.

### 2.3.1 Expert entries

The 21,386 expert entries whose student has a linked track:

**Profile** (train / test)

| | Mean | Median | SD |
|---|---|---|---|
| Logistic regression (`c5341cb6385e`) | 2.809 / 2.822 | 2.787 / 2.784 | 0.156 / 0.194 |
| Random forest (`5df1987449dd`) | 2.811 / 2.821 | 2.792 / 2.792 | 0.128 / 0.140 |
| k-NN (`75a55ca34835`) | 2.824 / 2.859 | 2.902 / 2.884 | 0.844 / 0.742 |
| Pooled labelers (`fdb140e9477e`) | 2.814 / 2.817 | 2.801 / 2.805 | 0.101 / 0.097 |
| No augmented copy (`f6dcb84aaa6b`) | 2.817 / 2.830 | 2.793 / 2.789 | 0.164 / 0.205 |
| Constant 3 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 |
| Training mean | 2.815 / 2.815 | 2.815 / 2.815 | 0.000 / 0.000 |
| *Self-report labels* | 2.885 | 3.000 | 0.895 |
| *Median of expert labels* | 2.895 | 3.000 | 0.510 |

**Metrics** (train / test)

| | MAE | RMSE | AUC* | Spearman | Macro F1** | Accuracy** |
|---|---|---|---|---|---|---|
| Logistic regression (`c5341cb6385e`) | 0.637 / 0.674 | 0.790 / 0.838 | 0.572 / 0.527 | 0.193 / 0.072 | 0.128 / 0.134 | 0.428 / 0.417 |
| Random forest (`5df1987449dd`) | 0.632 / 0.668 | 0.786 / 0.833 | 0.589 / 0.523 | 0.232 / 0.062 | 0.122 / 0.127 | 0.427 / 0.421 |
| k-NN (`75a55ca34835`) | 0.821 / 0.848 | 1.005 / 1.042 | 0.582 / 0.532 | 0.239 / 0.094 | 0.225 / 0.195 | 0.353 / 0.346 |
| Pooled labelers (`fdb140e9477e`) | 0.641 / 0.653 | 0.802 / 0.818 | 0.553 / 0.526 | 0.147 / 0.074 | 0.120 / 0.119 | 0.427 / 0.423 |
| No augmented copy (`f6dcb84aaa6b`) | 0.635 / 0.675 | 0.788 / 0.841 | 0.577 / 0.525 | 0.204 / 0.065 | 0.129 / 0.133 | 0.428 / 0.416 |
| Constant 3 | 0.618 / 0.619 | 0.840 / 0.839 | 0.500 / 0.500 | - / - | 0.120 / 0.119 | 0.427 / 0.423 |
| Training mean | 0.647 / 0.656 | 0.812 / 0.821 | 0.498 / 0.501 | - / - | 0.120 / 0.119 | 0.427 / 0.423 |
| *Mean of experts* | 0.668 | 0.860 | 0.563 | 0.143 | 0.208 | 0.447 |
| *Median of experts* | 0.638 | 0.913 | 0.543 | 0.120 | 0.222 | 0.451 |

### 2.3.2 Self-report entries

The 2,445 self-report entries whose student has a linked track; the reference Part 1 §1.6 argues
for:

**Profile** (train / test)

| | Mean | Median | SD |
|---|---|---|---|
| Logistic regression (`c5341cb6385e`) | 2.876 / 2.866 | 2.870 / 2.863 | 0.122 / 0.126 |
| Random forest (`5df1987449dd`) | 2.868 / 2.870 | 2.879 / 2.880 | 0.093 / 0.079 |
| k-NN (`75a55ca34835`) | 2.933 / 2.947 | 2.920 / 2.973 | 0.607 / 0.513 |
| Pooled labelers (`fdb140e9477e`) | 2.817 / 2.817 | 2.807 / 2.806 | 0.094 / 0.091 |
| No augmented copy (`f6dcb84aaa6b`) | 2.877 / 2.860 | 2.867 / 2.851 | 0.138 / 0.152 |
| Constant 3 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 |
| Training mean | 2.873 / 2.873 | 2.873 / 2.873 | 0.000 / 0.000 |
| *Self-report labels* | 2.885 | 3.000 | 0.895 |
| *Median of expert labels* | 2.876 | 3.000 | 0.428 |

**Metrics** (train / test)

| | MAE | RMSE | AUC* | Spearman | Macro F1** | Accuracy** |
|---|---|---|---|---|---|---|
| Logistic regression (`c5341cb6385e`) | 0.691 / 0.720 | 0.885 / 0.923 | 0.572 / 0.508 | 0.174 / 0.007 | 0.119 / 0.120 | 0.413 / 0.413 |
| Random forest (`5df1987449dd`) | 0.684 / 0.711 | 0.875 / 0.916 | 0.614 / 0.511 | 0.294 / 0.030 | 0.117 / 0.116 | 0.412 / 0.411 |
| k-NN (`75a55ca34835`) | 0.771 / 0.834 | 0.958 / 1.031 | 0.584 / 0.518 | 0.233 / 0.038 | 0.233 / 0.172 | 0.388 / 0.355 |
| Pooled labelers (`fdb140e9477e`) | 0.706 / 0.718 | 0.898 / 0.913 | 0.547 / 0.513 | 0.109 / 0.032 | 0.117 / 0.116 | 0.412 / 0.411 |
| No augmented copy (`f6dcb84aaa6b`) | 0.691 / 0.731 | 0.880 / 0.933 | 0.578 / 0.510 | 0.192 / 0.014 | 0.119 / 0.121 | 0.413 / 0.409 |
| Constant 3 | 0.668 / 0.670 | 0.910 / 0.910 | 0.500 / 0.500 | - / - | 0.117 / 0.116 | 0.412 / 0.411 |
| Training mean | 0.696 / 0.706 | 0.900 / 0.912 | 0.500 / 0.504 | - / - | 0.117 / 0.116 | 0.412 / 0.411 |
| *Mean of experts* | 0.689 | 0.884 | 0.619 | 0.241 | 0.179 | 0.435 |
| *Median of experts* | 0.661 | 0.914 | 0.609 | 0.213 | 0.179 | 0.439 |

### 2.3.3 What the tables show

Logistic regression and the random forest learn the soft rating target and output an expected
rating. k-NN cannot use soft labels here (`SklearnExpert.SOFT`): it is trained on rating ≥ 3 and
outputs P(rating ≥ 3), read back as 1 + 4p. For k-NN only the rank metrics (AUC, Spearman) are
comparable; its profile, MAE, RMSE and discretised metrics measure a different quantity.

- **The predictions are a near-constant at the label mean.** Logistic regression predicts with an SD
  of 0.19, against the labels' 0.81, around a mean of 2.82 (labels 2.80). Its median sits at 2.78,
  below the labels' median of 3. Rounded, almost every prediction is a 3 (§2.5).
- **It underfits.** On its training groups it reaches only Spearman 0.19 and AUC 0.57; held out, 0.07
  and 0.53. The random forest fits its training groups a little better (0.23 / 0.59) and does no
  better held out.
- **On the self-report every model is at chance** (logistic regression: AUC 0.508, Spearman 0.007).
  The random forest reaches Spearman 0.29 on its training groups' self entries. Nothing that tracks
  the student's report carries across groups.
- **MAE and RMSE reward shrinking.**
  - The pooled model (one model for all labelers) predicts with the smallest SD (0.10). It has the
    lowest MAE (0.653) and RMSE (0.818) of the models, and ranks no better (AUC 0.526).
  - Constant 3 still has the lowest MAE of all (0.619), and the training mean the lowest RMSE
    (0.821).
  - The model's RMSE (0.838) is at that floor.
- **Augmentation changes nothing.** Without the one augmented copy, every number moves by about 0.01 at
  most (paragraph below).

### 2.3.4 Each labeler separately

Logistic regression, expert entries. Each column is that labeler's own
entries; "macro" is the mean over the five labelers, "pooled" all entries together:

**Profile** (train / test)

| | Mean (predicted) | Mean (labels) | Median (predicted) | Median (labels) | SD (predicted) | SD (labels) |
|---|---|---|---|---|---|---|
| labeler_01 | 2.795 / 2.787 | 2.729 / 2.738 | 2.783 / 2.779 | 3.000 / 3.000 | 0.122 / 0.118 | 0.728 / 0.724 |
| labeler_02 | 2.894 / 2.932 | 2.885 / 2.948 | 2.886 / 2.914 | 3.000 / 3.000 | 0.229 / 0.228 | 0.735 / 0.722 |
| labeler_03 | 2.777 / 2.775 | 2.851 / 2.843 | 2.765 / 2.761 | 3.000 / 3.000 | 0.107 / 0.107 | 0.974 / 0.954 |
| labeler_04 | 2.978 / 3.159 | 2.978 / 3.030 | 2.948 / 3.159 | 3.000 / 3.000 | 0.222 / 0.259 | 0.858 / 0.868 |
| labeler_05 | 3.017 / 3.106 | 3.007 / 2.985 | 3.004 / 3.113 | 3.000 / 3.000 | 0.248 / 0.283 | 0.831 / 0.809 |
| **macro** | 2.892 / 2.952 | 2.890 / 2.909 | 2.877 / 2.945 | 3.000 / 3.000 | 0.186 / 0.199 | 0.825 / 0.815 |
| **pooled** | 2.809 / 2.822 | 2.786 / 2.799 | 2.787 / 2.784 | 3.000 / 3.000 | 0.156 / 0.194 | 0.812 / 0.810 |

**Metrics** (train / test)

| | MAE | RMSE | AUC* | Spearman | Macro F1** | Accuracy** |
|---|---|---|---|---|---|---|
| labeler_01 | 0.582 / 0.604 | 0.715 / 0.746 | 0.567 / 0.530 | 0.182 / 0.077 | 0.129 / 0.128 | 0.465 / 0.461 |
| labeler_02 | 0.561 / 0.581 | 0.676 / 0.717 | 0.665 / 0.629 | 0.392 / 0.268 | 0.157 / 0.156 | 0.469 / 0.473 |
| labeler_03 | 0.787 / 0.785 | 0.968 / 0.954 | 0.516 / 0.509 | 0.091 / 0.052 | 0.100 / 0.096 | 0.329 / 0.316 |
| labeler_04 | 0.650 / 0.906 | 0.796 / 1.090 | 0.648 / 0.485 | 0.348 / 0.038 | 0.154 / 0.124 | 0.426 / 0.314 |
| labeler_05 | 0.627 / 0.843 | 0.752 / 1.033 | 0.697 / 0.445 | 0.443 / -0.152 | 0.156 / 0.142 | 0.414 / 0.352 |
| **macro** | 0.641 / 0.744 | 0.781 / 0.908 | 0.618 / 0.519 | 0.291 / 0.057 | 0.139 / 0.129 | 0.421 / 0.383 |
| **pooled** | 0.637 / 0.674 | 0.790 / 0.838 | 0.572 / 0.527 | 0.193 / 0.072 | 0.128 / 0.134 | 0.428 / 0.417 |

Constant 3 on the same entries:

**Profile** (train / test)

| | Mean (predicted) | Mean (labels) | Median (predicted) | Median (labels) | SD (predicted) | SD (labels) |
|---|---|---|---|---|---|---|
| labeler_01 | 3.000 / 3.000 | 2.729 / 2.738 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 | 0.728 / 0.724 |
| labeler_02 | 3.000 / 3.000 | 2.885 / 2.948 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 | 0.735 / 0.722 |
| labeler_03 | 3.000 / 3.000 | 2.851 / 2.843 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 | 0.974 / 0.954 |
| labeler_04 | 3.000 / 3.000 | 2.978 / 3.030 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 | 0.858 / 0.868 |
| labeler_05 | 3.000 / 3.000 | 3.007 / 2.985 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 | 0.831 / 0.809 |
| **macro** | 3.000 / 3.000 | 2.890 / 2.909 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 | 0.825 / 0.815 |
| **pooled** | 3.000 / 3.000 | 2.786 / 2.799 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 | 0.812 / 0.810 |

**Metrics** (train / test)

| | MAE | RMSE | AUC* | Spearman | Macro F1** | Accuracy** |
|---|---|---|---|---|---|---|
| labeler_01 | 0.559 / 0.561 | 0.779 / 0.780 | 0.500 / 0.500 | - / - | 0.127 / 0.127 | 0.465 / 0.463 |
| labeler_02 | 0.543 / 0.542 | 0.751 / 0.747 | 0.500 / 0.500 | - / - | 0.127 / 0.127 | 0.467 / 0.466 |
| labeler_03 | 0.771 / 0.770 | 0.985 / 0.967 | 0.500 / 0.500 | - / - | 0.099 / 0.096 | 0.329 / 0.316 |
| labeler_04 | 0.668 / 0.689 | 0.887 / 0.906 | 0.500 / 0.500 | - / - | 0.113 / 0.110 | 0.395 / 0.381 |
| labeler_05 | 0.640 / 0.628 | 0.848 / 0.833 | 0.500 / 0.500 | - / - | 0.115 / 0.116 | 0.404 / 0.411 |
| **macro** | 0.636 / 0.638 | 0.850 / 0.847 | 0.500 / 0.500 | - / - | 0.116 / 0.115 | 0.412 / 0.408 |
| **pooled** | 0.618 / 0.619 | 0.840 / 0.839 | 0.500 / 0.500 | - / - | 0.120 / 0.119 | 0.427 / 0.423 |

- **One labeler is predictable: labeler_02.** Held out, labeler_02's model reaches AUC 0.629 and
  Spearman 0.268, the only clear signal in the experiment. labeler_02 responds to the hand-raising
  scores and to head lowering (experiment 08, Appendix A), which the features measure directly.
- **The dominant labeler is barely predicted.**
  - labeler_01 (61.5% of entries): AUC 0.530, Spearman 0.077.
  - labeler_03: 0.509 / 0.052.
  - The pooled numbers are mostly labeler_01's.
- **The two minority labelers overfit.**
  - labeler_04's and labeler_05's models reach Spearman 0.35 and 0.44 on their training groups, but
    only 0.04 and −0.15 held out; they have fewer than 1,400 entries each.
  - Their held-out prediction means (3.16 and 3.11) are also off their labels' (3.03 and 2.99). That
    costs MAE: 0.906 and 0.843, against constant 3's 0.689 and 0.628.
- **On MAE, constant 3 beats the model for every labeler,** by 0.04 (labeler_01 and labeler_02) to
  0.22 (labeler_04). Macro over labelers: 0.744 against 0.638.

### 2.3.5 Dropped and unchanged arms

**SVM dropped (18 November).** An RBF SVM was tried as a fourth expert model and dropped as
impractically slow: one arm took about 1.5 hours, against about 4 minutes for logistic regression.
- **The cost is in prediction.** An RBF SVM scores each row against every support vector, and on
  labels this close to chance nearly every training row probably becomes one.
- **Each fold therefore scores about 72k rows** (held out plus the training-set scores), for every
  labeler's model, on one core (`SVC` has no `n_jobs`).

That run's results (`13785d8fe677`) are kept but not reported. Logistic regression takes its place as
the default expert model in later experiments.

**Augmentation.** Each augmented copy is a full frame pass of a recording with its frames degraded:
colour, noise, a flip and cut-outs (`src/ingestion/augment.py`, guide 08 §8.4).
- **Copies add training rows only;** evaluation is always on the originals.
- **One copy per recording doubles the training rows and changes nothing:** the "No augmented copy"
  column (`f6dcb84aaa6b`) is within about 0.01 everywhere, with the same confusion matrix.
- **The six-session gain of +0.02–0.03 (05) does not hold at full scale.**
- **Heavier augmentation is not pursued.** It guards against overfitting to the training footage,
  and this model underfits. Four copies (about 8.5 hours of extraction) were planned and dropped.

## 2.4 Interpolation in training

How a labeler's curve is read between entries shapes the training target (`labels.interp`, Part 1
§1.4). The evaluation never interpolates (it scores raw entries), so differences here are not
artefacts of the scoring.

`window_mean` is symmetric around the entry. For an isolated change, step already targets the
midpoint of the jump, and a Gaussian gives the same value; ramp and Gaussian differ from step only
where entries are dense. The informative contrasts are therefore:
- interpolated against not (`train_vanilla`: the entry's own rating);
- the window length.

Settings:
- `labels.interp.mode=ramp labels.interp.ramp_s=3` (or 7);
- `labels.interp.mode=gauss labels.interp.sigma_s=1` (or 2);
- `labels.train_target=train_vanilla`;
- `window.length_s=8`.

Expert entries, logistic regression:

**Profile** (train / test)

| | Mean (predicted) | Mean (labels) | Median (predicted) | Median (labels) | SD (predicted) | SD (labels) |
|---|---|---|---|---|---|---|
| step (default) (`c5341cb6385e`) | 2.809 / 2.822 | 2.786 / 2.799 | 2.787 / 2.784 | 3.000 / 3.000 | 0.156 / 0.194 | 0.812 / 0.810 |
| ramp 3 s (`9f5c78e85899`) | 2.794 / 2.811 | 2.786 / 2.799 | 2.767 / 2.763 | 3.000 / 3.000 | 0.171 / 0.222 | 0.812 / 0.810 |
| ramp 7 s (`5c38aa124d4e`) | 2.785 / 2.804 | 2.786 / 2.799 | 2.758 / 2.752 | 3.000 / 3.000 | 0.181 / 0.235 | 0.812 / 0.810 |
| Gaussian 1 s (`83551175c8ce`) | 2.813 / 2.825 | 2.786 / 2.799 | 2.791 / 2.789 | 3.000 / 3.000 | 0.159 / 0.196 | 0.812 / 0.810 |
| Gaussian 2 s (`59fe8967ec34`) | 2.821 / 2.833 | 2.786 / 2.799 | 2.805 / 2.805 | 3.000 / 3.000 | 0.168 / 0.203 | 0.812 / 0.810 |
| no interpolation (`07cc8c676819`) | 2.782 / 2.802 | 2.786 / 2.799 | 2.751 / 2.744 | 3.000 / 3.000 | 0.198 / 0.253 | 0.812 / 0.810 |
| step, 8 s windows (`0159e99cadc2`) | 2.831 / 2.846 | 2.786 / 2.797 | 2.822 / 2.823 | 3.000 / 3.000 | 0.184 / 0.217 | 0.811 / 0.810 |
| Constant 3 | 3.000 / 3.000 | 2.786 / 2.799 | 3.000 / 3.000 | 3.000 / 3.000 | 0.000 / 0.000 | 0.812 / 0.810 |
| Training mean | 2.815 / 2.815 | 2.786 / 2.799 | 2.815 / 2.815 | 3.000 / 3.000 | 0.000 / 0.000 | 0.812 / 0.810 |

**Metrics** (train / test)

| | MAE | RMSE | AUC* | Spearman | Macro F1** | Accuracy** |
|---|---|---|---|---|---|---|
| step (default) (`c5341cb6385e`) | 0.637 / 0.674 | 0.790 / 0.838 | 0.572 / 0.527 | 0.193 / 0.072 | 0.128 / 0.134 | 0.428 / 0.417 |
| ramp 3 s (`9f5c78e85899`) | 0.634 / 0.679 | 0.783 / 0.842 | 0.585 / 0.532 | 0.225 / 0.092 | 0.135 / 0.137 | 0.431 / 0.416 |
| ramp 7 s (`5c38aa124d4e`) | 0.636 / 0.683 | 0.781 / 0.844 | 0.580 / 0.529 | 0.224 / 0.096 | 0.138 / 0.139 | 0.431 / 0.415 |
| Gaussian 1 s (`83551175c8ce`) | 0.636 / 0.673 | 0.792 / 0.840 | 0.574 / 0.530 | 0.186 / 0.068 | 0.128 / 0.135 | 0.428 / 0.417 |
| Gaussian 2 s (`59fe8967ec34`) | 0.635 / 0.672 | 0.797 / 0.845 | 0.576 / 0.535 | 0.168 / 0.057 | 0.130 / 0.137 | 0.429 / 0.418 |
| no interpolation (`07cc8c676819`) | 0.635 / 0.685 | 0.780 / 0.846 | 0.584 / 0.530 | 0.229 / 0.097 | 0.147 / 0.143 | 0.434 / 0.414 |
| step, 8 s windows (`0159e99cadc2`) | 0.636 / 0.674 | 0.801 / 0.851 | 0.575 / 0.534 | 0.160 / 0.049 | 0.135 / 0.142 | 0.430 / 0.421 |
| Constant 3 | 0.618 / 0.619 | 0.840 / 0.839 | 0.500 / 0.500 | - / - | 0.120 / 0.119 | 0.427 / 0.423 |
| Training mean | 0.647 / 0.656 | 0.812 / 0.821 | 0.498 / 0.501 | - / - | 0.120 / 0.119 | 0.427 / 0.423 |
| *Mean of experts* | 0.668 | 0.860 | 0.563 | 0.143 | 0.208 | 0.447 |
| *Median of experts* | 0.638 | 0.913 | 0.543 | 0.120 | 0.222 | 0.451 |

A ramp of width w and a Gaussian of σ = w/√12 have the same spread: ramp 3 s ≈ σ 0.87 s, ramp 7 s ≈
σ 2 s.

- **No target changes the result.** Held-out AUC stays at 0.527–0.535 and MAE at 0.672–0.685.
  Spearman varies more (0.049–0.097), but stays small.
- **Training on the entry's own rating spreads the predictions most** (SD 0.25, against 0.19 for
  step) and ranks a little better (Spearman 0.097, AUC 0.530). It costs MAE (0.685), as any extra
  spread does when the signal is this weak.
- **8 s windows do not help** (AUC 0.534, Spearman 0.049).

**Decision (18 November).** No interpolation in training from now on: every later experiment trains
on the entry's own rating (`labels.train_target = train_vanilla`, now the default in
`configs/deliverable3.json`). None of the interpolated targets changed the result, so the simpler
target wins. Interpolation stays where it is needed: comparing labelers (Part 1).

## 2.5 Confusion matrices

`compare_results` sums them over the held-out folds (true rating in rows, the prediction rounded to a
whole rating in columns, halves toward 3); `--plot` saves them as heatmaps.

**Single decoder, expert entries** (count / share of the row):

| true \ predicted | 1 | 2 | 3 | 4 | 5 | entries | recall |
|---|---|---|---|---|---|---|---|
| **1** | 0 | 0 | 3 / 1.00 | 0 | 0 | 3 | 0.00 |
| **2** | 0 | 69 / 0.01 | 8614 / 0.97 | 240 / 0.03 | 0 | 8923 | 0.01 |
| **3** | 0 | 80 / 0.01 | 8810 / 0.96 | 253 / 0.03 | 0 | 9143 | 0.96 |
| **4** | 0 | 31 / 0.01 | 2237 / 0.95 | 96 / 0.04 | 0 | 2364 | 0.04 |
| **5** | 0 | 4 / 0.00 | 931 / 0.98 | 18 / 0.02 | 0 | 953 | 0.00 |
| precision | – | 0.38 | 0.43 | 0.16 | – | | |
| predicted | 0 | 184 | 20595 | 607 | 0 | 21,386 | |

1. **The model is constant 3 with a few exceptions**: 96% of expert entries are predicted 3 (20,595 of
   21,386); self entries 99.4%. Of the 607 predicted 4s, 16% are 4s, near the 11% base rate.
2. **The expert ratings are mostly 2s and 3s in equal measure** (42% and 43%; 4s 11%, 5s 4.5%,
   1s three entries), so the task is first of all telling a 2 from a 3, and the model predicts rows 2
   and 3 alike. At `tau_att` 3 the binary view is "2 against 3 or more" and the model calls 99%
   engaged (balanced accuracy 0.50).

For comparison, the median of the other experts predicting the same entries (Appendix A) spreads its
predictions: 4,725 predicted 2, 16,655 predicted 3, 1,685 predicted 4. It gets 2,435 of the 8,923
true 2s right, but also predicts 2 for 1,748 true 3s: a human disagrees with a human in both
directions, where the model simply stays at 3.

---

# Reading

**(a) The FYP-1 result.** A subject-independent pipeline (leave one group out: no student, room or
session seen in training) reaches about half of the human ceiling on the labels it was given: AUC
0.527 against 0.564 for the other experts' mean, Spearman 0.072 against 0.145 (the best expert model,
k-NN, 0.532 / 0.094). On the self-report, the reference for the student's own state (§1.6), it is at
chance (AUC 0.508), while the experts reach 0.61. On MAE neither the model nor any human combination
beats constant 3, on either target. The model underfits (training AUC 0.55–0.59 for every expert model), so the limit is the features and the
target, not the decoder.

**FYP-1's two claims, as this experiment frames them:**

1. **Simple, cheap features can do well enough** — *to be shown*. "Well enough" has to be judged
   against the human ceiling, not against constant 3: on these labels constant 3 is unbeatable on MAE
   even for humans, and on RMSE the training mean (0.821) is the real floor. The current features
   (box kinematics, pooled residual flow, SCB behaviour classes, scene statistics) get about half-way
   to the ceiling. The next features (the clustering-based ones) are tested by how much of the
   remaining gap to 0.564 / 0.145 they close.
2. **A single number cannot score attention** — supported here, developed in FYP-2. Labelers do not
   agree on a moment's rating (α ≈ 0); their disagreement is not noise around one value but differences
   in level, how often they change, and how much of the scale they use (§1.2, §1.8); and the strongest
   shared signal is at the level of a student's session and between the student and the experts'
   consensus (§1.7). One rating per moment hides all of that. FYP-2 replaces it with a suite of
   statistics: per-labeler distributions, level and variability over time, and agreement-aware summaries.

**Perspectivism** (§1.8) is the short-term consequence: evaluate against each labeler, never against an
aggregated gold. **Leaving DIPSER** is the long-term one: its moment-level labels have a ceiling no
model can show much under.

**(b) Limits of the claim.**
- The ceiling is measured with *other* labelers. A labeler's consistency with themselves can only be
  approximated, by their mean over the rest of the session (§1.8.1), which measures their overall view
  of a student, not a repeat judgement of the same moment, and is pulled against the entry by the
  swing-back of changes.
- labeler_01 is 61.5% of the expert entries and the least predictable, so pooled numbers are mostly
  about labeler_01. Averaged per labeler instead (macro), the experts' mean reaches AUC 0.62 while
  the model reaches 0.519 (§2.3.4): per labeler, the model is further from the ceiling than the
  pooled "half-way" comparison suggests. Its one clear per-labeler signal is labeler_02 (AUC 0.629).
- The entries were made retrospectively at 1 fps; no lag improves agreement, but a labeler's reaction
  time could vary from entry to entry, which no fixed lag captures (the ±k s upper bounds bound it).
- A single labeler's 1–5 value ties heavily, which lowers its AUC; the continuous mean of the others is
  the fair row.

**(c) For FYP-2.**
- **Change the target, not the model.** No unit or interpolation of DIPSER's moment ratings has
  usable agreement (α ≤ 0.13). The strongest shared signal found is at the session level between the
  student and the experts' consensus (Spearman 0.33), and the experts' mean predicting the
  self-report (AUC 0.61–0.63). Candidates: a suite of statistics instead of one rating (above); the
  self-report as a target, with the experts' consensus as an input or auxiliary; observable behaviours
  that can be annotated reliably (looking at the board, writing, phone use); or a dataset with reported
  agreement, checked with `scripts/labels/label_agreement.py` before any training.
- **Moving off DIPSER costs less than it sounds.** Detection, tracking, the feature pass, per-label
  evaluation and the agreement analysis are dataset-agnostic; the DIPSER-specific parts are the loader
  (`src/datasets/dipser.py`), the label parsing (`src/labels/attention.py`) and the detector and tracker
  tuning (experiment 02).
- **Report the ceiling next to every result**, as here, so a score has a scale.
- **Keep step (or a ramp ending at the entry) as the interpolation**; Gaussian blurring lowers
  agreement.

**(d) What this means for `docs/misc/targets.md` and `docs/misc/benchmarks.md`.**
- *targets.md.* Its decisions in §7 stand: per label, per labeler, experts and self as separate tracks,
  session start excluded, step by default. Its table of "every target against constant 3" (§3) needs a
  second reference line, the human ceiling, because constant 3 is not the only floor: for the
  per-label targets, a human scores MAE 0.75 against constant 3's 0.62. Its per-second and per-frame
  units count long holds many times and give the most flattering numbers (§2); this part shows they do
  not change agreement (α is the same per second and per entry), only the apparent accuracy. Its
  student counts (483) include experiments 07–09; the pipeline's scope is 321.
- *benchmarks.md.* Its trivial-baselines table (§1) gains a human-ceiling row: on the per-label targets
  the other experts' mean gets MAE 0.668, AUC 0.564. The published DIPSER results it lists pool folds
  or units (per frame, per second); §6 of experiment 04 showed pooling turns AUC 0.51 into 0.85, so
  those numbers are not comparable with ours or with the ceiling. The dataset paper reports no
  inter-rater agreement and the follow-up compares each labeler with a majority it is part of; this
  is, as far as we know, the first agreement measurement on DIPSER's ratings, and it says moment-level
  engagement in DIPSER is not a target humans agree on.

---

## Reproduction

```
python -m scripts.labels.label_agreement all --model-entries        # Part 1, ~9 min -> outputs/labels/agreement.json
python -m scripts.data_visualization.inspect_labels          # the 14 Nov alpha (outputs/labels/report.json)
# Part 2: the commands of §2.2 (results and logs per arm: outputs/experiments/06_full_pipeline/)
```

## Appendix A: experts against each other (aggregates)

The ceiling for the experts' own labels: every expert entry predicted from the *other* experts, on
the model's entries. These are the numbers earlier versions of this document led with.

Every entry of a labeler A (session start excluded) is predicted from the **other** labelers' curves
at that moment, and scored with `rating_metrics` exactly like the model. Predictors:

- **one other expert**: each other expert's curve separately, every (entry, other expert) pair a row;
- **median / mean of the other experts** (the target labeler and self excluded); the mean is
  continuous like the model's output, so it is the fair comparison for AUC and Spearman (a single
  labeler's 1–5 value ties heavily);
- **learned combination**: ridge regression of A's rating on every other labeler's value (missing →
  a flag) and A's identity, leave one group out — the best a linear reader of the other humans does;
- **self-consistency**: A's own *other* ratings of the same student (time-weighted mean over A's other
  runs), and A's previous value;
- **labeler prior**: A's mean rating in the other groups.

**On exactly the model's entries** (21,386 expert entries with a linked track; mean over groups):

| Predictor (step) | MAE | RMSE | Accuracy | Macro F1 | AUC | Spearman |
|---|---|---|---|---|---|---|
| Constant 3 | **0.619** | 0.839 | 0.423 | 0.119 | 0.500 | – |
| **Model** (per-labeler, held out) | 0.674 | 0.838 | 0.417 | 0.134 | 0.527 | 0.072 |
| One other expert | 0.747 | 1.057 | 0.416 | 0.239 | 0.534 | 0.083 |
| Median of the other experts | 0.638 | 0.913 | 0.451 | 0.222 | 0.542 | 0.119 |
| Mean of the other experts | 0.668 | 0.861 | 0.447 | 0.208 | **0.564** | **0.145** |
| Learned combination of the others | 0.658 | **0.811** | 0.432 | 0.138 | 0.563 | 0.146 |
| Self-consistency: A's mean over its other runs | 0.666 | 0.858 | 0.443 | 0.231 | 0.571 | 0.142 |
| Self-consistency: A's previous value | 1.215 | 1.314 | 0.000 | 0.000 | 0.202 | −0.473 |
| Labeler prior | 0.675 | 0.828 | – | 0.119 | 0.495 | 0.045 |

**MAE and RMSE have different floors.** MAE is smallest when a constant prediction is the *median*
rating, which is exactly 3; RMSE is smallest at the *mean*, about 2.8. So on RMSE constant 3 (0.839) is
not the bar: the training mean (0.821) and the labeler prior (0.828) both beat it without looking at
a student. Against that floor, the model (0.838; random forest 0.833) is at it, and only the learned
combination of the other humans (0.811) is below it. "Beating constant 3" on RMSE says nothing.

On all 23,179 expert entries the numbers are within 0.002 of these. The learned combination fits its
training groups only a little better (AUC 0.586, Spearman 0.201) than it predicts the held-out one.

**Robustness of the ceiling:**

| Variant (expert entries) | One other expert: MAE / AUC / Spearman | Mean of other experts: MAE / AUC / Spearman |
|---|---|---|
| step (above) | 0.747 / 0.534 / 0.081 | 0.668 / 0.563 / 0.143 |
| ramp 3 s | 0.740 / 0.537 / 0.087 | 0.664 / 0.568 / 0.152 |
| Gaussian σ 2 s | 0.766 / 0.513 / 0.046 | 0.688 / 0.526 / 0.086 |
| step, other read 1 s later | 0.754 / 0.529 / 0.073 | 0.673 / 0.553 / 0.129 |
| step, other read 2 s later | 0.761 / 0.522 / 0.062 | 0.679 / 0.540 / 0.109 |
| **run midpoints** instead of entries (24,470) | 0.737 / 0.532 / 0.079 | 0.658 / 0.558 / 0.135 |
| **macro over target labelers** (step) | 0.735 / 0.573 / 0.150 | 0.669 / 0.624 / 0.243 |
| *upper bound*: best match within ±2 s | 0.649 / 0.600 / 0.221 | |
| *upper bound*: best match within ±5 s | 0.595 / 0.635 / 0.296 | |
| *upper bound*: best match within ±10 s | 0.548 / 0.666 / 0.361 | |

- **Run midpoints** (the middle of each of A's runs, where A's value is most surely in force rather
  than at the instant A changed it) give the same ceiling: the per-entry unit is not what makes it low.
- **Macro over labelers** is higher (AUC 0.62, Spearman 0.24): labeler_01 (61.5% of entries) is the
  hardest to predict (0.51–0.56 AUC from any one other labeler), and dominates the pooled numbers. The
  model's results are not broken down per labeler yet, so the model has no macro row.
- **The upper bounds** let each prediction pick, within ±k s, the other expert's value *closest to the
  answer*: they select by the target and are not achievable. Even with ±2 s of cherry-picking, MAE is
  0.649; only at ±5 s does it fall below constant 3.

**Per pair** (A ← B, step, one other labeler, pooled; AUC / Spearman / MAE, MAE of constant 3 on the
same entries in brackets): the minority experts are predicted best *by labeler_01*: labeler_04 ←
labeler_01 0.693 / 0.337 / 0.535 (0.628), labeler_05 ← labeler_01 0.663 / 0.382 / 0.580 (0.666),
labeler_02 ← labeler_01 0.666 / 0.318 / 0.614 (0.542). In the other direction labeler_01 is predicted
at 0.514–0.564. The full matrix is in Appendix B.

**Per activity** (mean of the other experts, step, AUC / Spearman): 01 0.562 / 0.116, **02 0.598 /
0.214**, 03 0.579 / 0.149, 04 0.527 / 0.075, 05 0.540 / 0.089, 06 0.544 / 0.096. Activity 02 is also
the one with α above zero.

**Self entries** (2,445 on the model's entries; now §1.6): the mean of the experts predicts the student's own
rating at AUC 0.613, Spearman 0.232, MAE 0.693 (constant 3: 0.670); the learned combination 0.628 /
0.235 / 0.690. The model: 0.508 / 0.007 / 0.720. The experts' ceiling is higher for the self-report
than for each other.

**Confusion of a human predicting a human** (median of the other experts, all expert entries; rows
true 1–5, columns predicted 1–5):

| true \ predicted | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| **1** | 0 | 1 | 2 | 0 | 0 |
| **2** | 0 | 2,435 | 6,620 | 550 | 50 |
| **3** | 0 | 1,748 | 7,594 | 520 | 23 |
| **4** | 0 | 352 | 1,735 | 498 | 21 |
| **5** | 0 | 189 | 704 | 117 | 20 |

The humans move off 3 far more often than the model (28% of predictions against 4%), but rows 2 and 3
are predicted almost alike: a true 2 is predicted 2 in 25% of cases, a true 3 in 18%.

## Appendix B: one labeler predicting another

A ← B: B's step curve at A's entries (session start excluded), pooled over groups. AUC / Spearman /
MAE, with constant 3's MAE on the same entries.

| A ← B | AUC | Spearman | MAE | MAE const 3 | Entries |
|---|---|---|---|---|---|
| labeler_01 ← labeler_02 | 0.514 | 0.046 | 0.611 | 0.558 | 14,258 |
| labeler_01 ← labeler_03 | 0.529 | 0.055 | 0.682 | 0.558 | 14,258 |
| labeler_01 ← labeler_04 | 0.518 | 0.043 | 0.856 | 0.542 | 7,578 |
| labeler_01 ← labeler_05 | 0.564 | 0.171 | 0.814 | 0.576 | 6,680 |
| labeler_01 ← self | 0.532 | 0.087 | 0.731 | 0.558 | 14,258 |
| labeler_02 ← labeler_01 | 0.666 | 0.318 | 0.614 | 0.542 | 1,263 |
| labeler_02 ← labeler_03 | 0.593 | 0.197 | 0.700 | 0.542 | 1,263 |
| labeler_02 ← labeler_04 | 0.588 | 0.217 | 0.839 | 0.539 | 900 |
| labeler_02 ← labeler_05 | 0.589 | 0.033 | 0.895 | 0.551 | 363 |
| labeler_02 ← self | 0.547 | 0.112 | 0.765 | 0.542 | 1,263 |
| labeler_03 ← labeler_01 | 0.574 | 0.150 | 0.777 | 0.778 | 5,702 |
| labeler_03 ← labeler_02 | 0.507 | 0.025 | 0.803 | 0.778 | 5,702 |
| labeler_03 ← labeler_04 | 0.529 | 0.041 | 1.035 | 0.794 | 3,692 |
| labeler_03 ← labeler_05 | 0.497 | 0.009 | 0.922 | 0.748 | 2,010 |
| labeler_03 ← self | 0.511 | 0.057 | 0.883 | 0.778 | 5,702 |
| labeler_04 ← labeler_01 | 0.693 | 0.337 | 0.535 | 0.628 | 1,370 |
| labeler_04 ← labeler_02 | 0.550 | 0.144 | 0.612 | 0.628 | 1,370 |
| labeler_04 ← labeler_03 | 0.614 | 0.140 | 0.695 | 0.628 | 1,370 |
| labeler_04 ← self | 0.535 | 0.085 | 0.812 | 0.628 | 1,370 |
| labeler_05 ← labeler_01 | 0.663 | 0.382 | 0.580 | 0.666 | 586 |
| labeler_05 ← labeler_02 | 0.541 | 0.181 | 0.630 | 0.666 | 586 |
| labeler_05 ← labeler_03 | 0.602 | 0.225 | 0.819 | 0.666 | 586 |
| labeler_05 ← self | 0.620 | 0.293 | 0.676 | 0.666 | 586 |
| self ← labeler_01 | 0.609 | 0.214 | 0.690 | 0.666 | 2,634 |
| self ← labeler_02 | 0.527 | 0.091 | 0.686 | 0.666 | 2,634 |
| self ← labeler_03 | 0.570 | 0.154 | 0.751 | 0.666 | 2,634 |
| self ← labeler_04 | 0.535 | 0.043 | 0.971 | 0.688 | 1,298 |
| self ← labeler_05 | 0.536 | 0.098 | 0.861 | 0.645 | 1,336 |

labeler_04 and labeler_05 never rate the same student, so they have no pair.

Each row is scored on A's entries for the students B also rated, and "MAE const 3" is constant 3 on
those same entries. It is therefore the same within a block wherever B rated every student
(labeler_01–03 and self: 321 student-sessions each). It differs where B is labeler_04 or labeler_05,
who rated only some students (178 and 143), so the entry counts differ too.

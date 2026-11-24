# Chapter 13 — From ratings to reported numbers: the implementation in detail

Chapter 12 explains the *concepts* behind the results. This chapter documents exactly what the
code *does*, step by step, from a labeler's raw rating to the numbers in a results file and the
curve at inference. It is meant for checking the implementation against the intended design, so
it records behaviour as it is, cites the code for every claim, and marks each place where the
code makes a choice that the design may or may not intend.

**Scope:** this chapter documents the window-grid paths (`engagement.labelers` = `consensus` or
`ensemble`). Since 16 November 2026 the default is `per_labeler` (Chapter 7, §7.9;
`docs/misc/targets.md` §7), which replaces the window labels, consensus and window-level
evaluation described here; observations O3–O7, O10–O17 and O22 concern the window-grid paths.

**Nothing here has been changed in the code.** The observations (O1, O2, …) are neutral: they say
what happens and what it implies, not what it should be. §13.17 collects them in one table.

Measured figures come from the six-session run (`docs/experiments/04_six_session_run.md`, cached
in `tmp/e2e6/`) with its settings: `tau_att = 3`, `tau_share = 0.6`, late aggregation, Vec as
context, logistic-regression experts (`C = 0.1`), random-forest gate.

The path, in one picture:

```
labeler JSON events ──► AttentionTrack (step function)                     §13.1
        │
        ▼  per student, per window: each labeler's time-weighted mean       §13.3
   consensus = median over labelers ──► student label: consensus ≥ tau_att  §13.4
        │                                       │
        ▼  over the camera's students            │                           §13.2
   labelled share ──► window label: share ≥ tau_share (if ≥ 3 students)    §13.5
                                                 │
tracks ──► SID–TID association ──► each track row gets its student's label  §13.6
        │
        ▼  rows: one per (window, track): trk.* (+ Vec)                     §13.7
   experts (per activity or shared) ──► P(engaged) per row                  §13.8
   gate ──► activity probabilities ──► mixture weights the experts          §13.9
        │
        ▼  window score = mean of its rows' P                               §13.10
   cut at tau_share ──► predicted engaged                                   §13.11
        │
        ▼  leave-one-group-out, per fold                                    §13.12
   AUC, F1, balanced accuracy vs window label                               §13.13
```

## 13.1 A labeler's ratings become a step function

**Code:** `src/labels/attention.py`, `parse_events` (line 47), `AttentionTrack` (line 22).

Each labeler file in a subject ZIP (`labels/<labeler>.json`) is a list of events with a time and,
sometimes, an `attention` value. `parse_events` keeps the events with an attention value in 1–5,
sorts them by time, and shifts them by `labels.offset_s` (0 in the config). Values outside 1–5 are
dropped, and the previous value keeps holding (DIPSER has at least one attention "7").

The result is a **step function**: value `v[i]` holds from `t[i]` until the next event; the last
value holds forever. Two ways to read it:

- `at(t)` (line 28): the value in force at time `t`; NaN before the first event.
- `mean_over(t0, t1)` (line 33): the **time-weighted mean** over `[t0, t1)`. If a rating changes
  from 3 to 2 halfway through a window, the mean is 2.5. It returns **NaN if the window starts
  before the labeler's first event**, even if the first event falls inside the window.

> **O1.** A labeler whose first rating comes 1 second into a window is ignored for that whole
> window (NaN), not counted for the remaining 7 seconds. At session starts this can remove labelers
> from the first windows.

> **O2.** The last rating holds "indefinitely": a labeler who stopped rating before the end of the
> session is treated as still giving their last value until the end.

> **O25.** The step function reads each event as "from now on, the value is v". Whether that is
> what a rating means is open: a rating may instead summarise what the labeler *has just seen*
> (retrospective), and every rating is entered after a reaction delay. What the files show
> (all 483 labelled student-sessions, `scripts/labels/label_baselines.py` data):
>
> | | Expert labelers | Self-report |
> |---|---|---|
> | Rating tracks | 1,932 | 481 |
> | Changes per minute (median) | 2.2 | 1.2 |
> | Duration of a value before it changes (median; 10th–90th percentile) | 3.1 s (1.0–23 s) | 8.1 s (1.6–59 s) |
> | Values that last ≤ 2 s | 33% | 14% |
> | Tracks that never change | 14% (73% of them at 3) | 12% |
>
> Every labeler of a student, self included, has their first event at the same instant (the
> session start), and first values vary (experts: 2 34%, 3 52%, 4 12%), so the first event is a
> judgement, not a tool default. The many 1–2 s values suggest frame-by-frame annotation of
> the 1-fps video. Whether the self-report was entered live or afterwards is not stated in the
> sources read. **The event times say the ratings were entered offline, frame by frame:** 88% of all
> 41,694 rating events after the first (experts and self-reports alike) fall in the first tenth of
> a second, where live key presses would spread evenly (10% per tenth). This matches the dataset
> paper's "annotators saw one frame per second" (`benchmarks.md` §4). Under that protocol a long
> gap means the labeler stepped through the frames and recorded no change; whether they looked
> at every frame is not known.
>
> **Alternatives, compared but not implemented in the pipeline.** Each labeler's events turned into a
> curve in six ways, then the pipeline's student consensus (median of 8-second means) and label
> (consensus ≥ 3), over all 34,285 student-windows (`scripts/analysis/label_interp_compare.py`):
>
> | Mode | How events become a curve | Consensus vs step: mean diff / r | Student labels that flip | Engaged rate (step: 0.680) |
> |---|---|---|---|---|
> | step | value holds until the next event (current) | – | – | 0.680 |
> | ramp (3 s) | step, with a 3 s linear ramp ending at each event | 0.045 / 0.98 | 3.5% | 0.655 |
> | linear | straight line between events | 0.306 / 0.64 | 29.0% | 0.465 |
> | pchip | monotone cubic between events | 0.319 / 0.61 | 29.2% | 0.463 |
> | cubic | natural cubic spline through events | 0.741 / 0.20 | 30.9% | 0.570 |
> | backward | value applies to the time before its event | 0.484 / 0.29 | 35.8% | 0.632 |
> | gauss σ = 1 s | the step blurred by a Gaussian (each change an S-curve centred on its event) | 0.009 / 0.999 | 3.6% (11.4%) | 0.645 (0.567) |
> | gauss σ = 2 s | same | 0.022 / 0.996 | 6.9% (16.6%) | 0.615 (0.517) |
> | gauss σ = 4 s | same | 0.053 / 0.983 | 11.9% (22.2%) | 0.574 (0.468) |
> | gauss σ = 2 s, 2 s earlier | centred 2 s before each event (a rating as a look back) | 0.057 / 0.976 | 8.0% (17.5%) | 0.616 (0.519) |
>
> | decay, half-life 10 s | step, relaxing toward 3 with time since the labeler's last event | 0.253 / 0.59 | 1.1% | 0.679 |
> | decay, half-life 30 s | same | 0.181 / 0.79 | 0.5% | 0.678 |
> | decay, half-life 60 s | same | 0.126 / 0.89 | 0.3% | 0.678 |
>
> | stale, 4 s | step; a labeler silent for more than 4 s (half a window) counts as 3 | 0.307 / 0.25 | 26.6% | 0.940 |
> | stale, 8 s | same, after 8 s | 0.290 / 0.33 | 24.6% | 0.921 |
> | gauss σ = 2 s + stale 4 s | both | 0.308 / 0.27 | 25.9% | 0.904 |
>
> The stale rule turns 89–92% of all student-windows into a consensus of exactly 3 (step: 49.5%):
> each labeler is silent for more than 4 s most of the time, so the median of five is neutral almost
> everywhere, and at `tau_att = 3` nearly every student becomes "engaged".
>
> Decay toward 3 never carries a value across 3, so at `tau_att = 3` it changes almost no labels,
> while it reshapes the continuous consensus (r 0.59–0.89 with step) and removes most high values:
> at `tau_att = 4` the engaged rate falls from 9.4% to 0.1–0.3%, at 3.5 from 14.5% to 1.6–7.7%.
>
> Gaussian rows: the first figure rounds the consensus to 0.01 before thresholding, the one in
> brackets does not. The Gaussian never quite reaches zero, so a consensus of exactly 3 becomes
> 2.9999 whenever a later change is downward, and flips; see O26.
>
> The choice matters: interpolating between events flips almost a third of the student labels and
> moves the engaged rate from 68% to 47%, because a value that held for a long stretch is replaced
> by a line toward the next value across that whole stretch. A plain cubic spline spends 30% of the
> time outside 1–5. A Gaussian changes the labels in proportion to σ, and turns short episodes
> into shallow dips (a 1-second 2 between 3s reaches only 2.8 at σ = 2 s). The label viewer shows
> step, ramp, linear, pchip, cubic and gauss (`scripts/data_visualization/label_viewer.py --interp
> <mode>`, key `i`; `--sigma` and keys `;` `'` for the Gaussian's width), with `--decay` (key `d`) and
> `--stale` (key `s`) on top of any mode; backward and the additive variant were dropped from it.

## 13.2 Which students count for a session-camera

**Code:** `src/labels/session.py`, `_students_and_labeler` (line 21).

A session-camera's label pool is built from the subject ZIPs of its group and experiment. With
`labels.students = "camera"` (the config value), only subjects that have a manual box in **this
camera's** annotation file (`abi_labels`, `index.annotations(key)`) are kept; `"all"` keeps every
labelled subject of the session.

> **O3.** "Visible in this camera" comes from **one static box per subject per camera** (the
> manual annotation). A student annotated for a camera counts in every window of that camera,
> whether or not they are actually detected or tracked in that window. The window label therefore
> describes the annotated students, while the model's score (§13.10) is built from whoever is
> tracked. On the six sessions the two agree closely: 10.9 labelled students vs 11.3 tracked rows
> per window on average, and the labelled share differs from the share among *linked tracked*
> students by 0.02 on average (correlation 0.99).

## 13.3 A student's consensus attention in a window

**Code:** `src/labels/window_labels.py`, `WindowLabeler.consensus` (line 27).

For one student and one window:

1. each of the student's labelers gives their time-weighted mean over the window (§13.1);
2. labelers with NaN (window starts before their first rating) are dropped;
3. if `labels.exclude_self` is true, the self-report (`self_labeling`) is dropped; **the config
   has it false, so the student's own rating is included**;
4. the **median** of what remains is the consensus. With no labeler left, NaN.

Each DIPSER student has 4 expert labelers plus the self-report (Chapter 12, `benchmarks.md` §4), so
the median is usually over 5 values: the middle one.

> **O4.** The consensus is a median of *time-weighted means*, so it can be a non-integer (e.g. the
> 2.5 above), and the median of five values is one labeler's mean, not an average. With
> `tau_att = 3`, a student is engaged only if at least three of five labelers average ≥ 3 over the
> window.

> **O5.** The self-report counts as one of five votes (`exclude_self = false`). In the taxonomy of
> `docs/misc/targets.md`, the pipeline's target is *per window × median × experts + self*. The published works
> differ: the dataset authors' majority includes it, Fragkiadakis et al. exclude it
> (`benchmarks.md` §3.1, §3.3).

> **O26.** **49.5% of all student-window consensus values are exactly 3.0** (34,285 student-windows,
> step labels, 8-second windows): the median lands on a labeler whose window mean is exactly 3,
> because 3 is the most common rating and many values hold for whole windows. With
> `tau_att = 3` and the rule "≥", half of all student labels sit exactly on the threshold and are
> "engaged" only because of the equality. Any change that nudges such a value (smoothing,
> a different interpolation, floating-point error, a labeler's 1-second 2) flips it. The design's
> `tau_att = 4` has the opposite problem (almost nothing is ≥ 4, O6).

## 13.4 A student's label

**Code:** `WindowLabeler.student_labels` (line 34); `student_labels_session` in
`src/labels/session.py` (line 38).

`1.0` if consensus ≥ `tau_att`, `0.0` if below, NaN if no consensus. The same rule is used to count
engaged students for the window label (§13.5), so the student labels and the window label are
consistent by construction. Student labels ignore `min_labeled_students`.

With the per-labeler ensemble (`engagement.labelers = "ensemble"`, off in the config),
`labeler_labels_session` (line 46) gives each labeler's *own* call instead: their time-weighted
mean ≥ `tau_att`, no median.

## 13.5 The window's labelled share and label

**Code:** `WindowLabeler.label` (line 43).

1. consensus for every student in the pool (§13.2), NaN ones dropped;
2. **share engaged** = share of those with consensus ≥ `tau_att`; NaN if none;
3. **window label** = 1 if share ≥ `tau_share`, else 0, **but only if at least
   `min_labeled_students` (3) students have a consensus**; otherwise the label is `None` (the
   window is not used for training or classification metrics). The share is still computed.

`scripts/train_models.py`, `load_table` (line 51) attaches these to the window table as `y`,
`share` and `n_labeled`.

> **O6.** The window label is a *threshold of a threshold*: attention → student (at `tau_att`) →
> share → window (at `tau_share`). Information is lost twice: a student at consensus 2.9 and one at
> 1.0 are both "not engaged", and a window at share 0.59 and one at 0.0 are both "not engaged".
> 04 Appendix B.1 shows the result: the window label changes 0–18 times in 73 windows per session.

> **O7.** Windows with fewer than 3 labelled students have no label but keep their share, so they
> are excluded from AUC/F1. (The removed `retention_r` still counted them; `docs/misc/proposed_metrics.md`.) On the six
> sessions this is one window.

## 13.6 Linking tracks to students (training only)

**Code:** `src/tracking/association.py`, `associate_sids` (line 112), `frame_matches`, `_mean_box_sids`.

Late aggregation trains each track's row on *its student's* label, so each track id (TID) must be
linked to a subject id (SID). Because the manual annotation is one **static box per subject per
camera**:

1. in every frame, the tracks' boxes and the subjects' static boxes are matched one-to-one by the
   Hungarian algorithm on IoU (intersection over union); pairs with IoU < `association.min_iou`
   (0.3) are discarded;
2. each track's matched SIDs are counted over all its frames; the most frequent wins (ties: the
   alphabetically first);
3. the winner is accepted only if it matched in at least `association.min_match_share` (0.5) of
   **all** the track's frames; otherwise the track stays unlinked.

The static boxes are not right in every frame (experiment 01 §2): students lean, turn and stand,
and TAs walk through the room, often behind seated students, with no box of their own. A TA crossing
a student's box can match it for a few frames; the 50% rule above is what keeps such a track from
being linked to the student.

**Alternative: `association.method` = `"mean_box"`** (the default, `"per_frame"`, is the procedure
above and is what every experiment so far used; a config without the key gets it, so older hashes
are unchanged). Each track is reduced to its **mean box** over all its frames, and the mean boxes are
matched to the subject boxes by the Hungarian algorithm on IoU, once per track instead of once per
frame. A fragmented student has several tracks, so the matching runs in rounds: tracks left over
may join a subject that already has tracks, but only if they share no frame with those tracks (two
boxes at the same time are two people). Both guards still apply: the mean box needs IoU ≥
`min_iou`, and the track must overlap its subject's box in at least `min_match_share` of its frames.
On the 52 original recordings (base config) it gives the same links as `per_frame` in 33; in the
rest it mostly links one to three fewer tracks, and swaps a track in three.

Unlinked tracks (people not annotated, such as teachers and TAs, or students who moved away from
their box) get a NaN label and are left out of training.

> **O8.** A student who changes seat or leans far out of their annotated box can fail the 0.5
> share and lose their label; a person standing in front of a seated student for most of a track
> can take that student's label. On the six sessions 92% of track rows are linked (8% unlinked).

## 13.7 The rows the experts see

**Code:** `src/experts/levels.py`, `early_rows` (line 54), `late_rows` (line 59);
`scripts/train_models.py`, `engagement_rows` (line 110).

- **Early** (`engagement.aggregate = "early"`): one row per window = the window's Vec (58 class-level
  features, Chapter 6, §6.4). Label: the window label `y` (§13.5).
- **Late** (the config): one row per (window, visible track) = that track's own `trk.*` features
  (22 numbers), plus the window's Vec appended **if `engagement.use_class_context` is true** (it is).
  Label: the linked student's label (§13.4), NaN if unlinked. The row also carries its window
  index and its window's activity (`exp`).

> **O9.** In late mode with context, every student in a window carries the same 58 Vec numbers next
> to their own 22. Vec differs mainly *between sessions* (room, camera, activity), so the expert can
> learn session-level offsets from it. 04 Appendix B.4 measured that turning context off improved held-out
> AUC and `retention_r`.

## 13.8 The experts

**Code:** `src/experts/expert.py`; config `expert = {"name": "logreg", "params": {"C": 0.1}}`.

**Learned experts** (`SklearnExpert`, line 25): logistic regression, random forest, SVM or k-NN on
the row features, inside a pipeline that fills missing values with the training median and standardises every feature (`src/core/classifiers.py`). The hand-weighted expert instead treats a missing feature as average (z = 0).

- **Class balancing (line 42):** logistic regression, random forest and SVM are built with
  `class_weight = "balanced"` unless the config overrides it: during training, each class's
  errors are weighted inversely to its frequency, as if the classes were 50/50. k-NN is balanced
  by undersampling the larger class instead.
- **Degenerate training sets (line 52):** if the training rows contain only one class, or none at
  all, the expert predicts a **constant**: the training label rate, or **0.5 when it has no rows**.

> **O10.** Balanced class weights change what the predicted probability means. A balanced model's
> probabilities are calibrated to a 50/50 world, not to the real engaged rate. Measured on the six
> sessions (shared expert, fitted on all rows, in-sample): 64.1% of labelled student rows are
> engaged, but the mean predicted probability is **0.524**. This matters for §13.10, where the
> mean probability is read as an estimate of the share, and for §13.11, where it is cut at 0.6.

**Hand-weighted expert** (`HandWeightedExpert`, line 95): never sees labels.
`fit` only learns each feature's mean and standard deviation on the training rows; prediction is
the sigmoid of a fixed weighted sum of standardised features (line 115):

  P = 1 / (1 + exp(−Σ wᵢ · (xᵢ − μᵢ) / σᵢ))

Weights from `hand_weighted.weights`; in late mode the window-level keys are mapped to the
student's own feature (`student_key`, `vec.kg.stillness.mean` → `trk.kg.stillness`). The phone-use
weight's sign is set per activity (`phone_sign`).

> **O11.** Standardised features average 0 on the training rows, so the hand-weighted score
> averages about 0.5 by construction, whatever the engaged rate. Cut at 0.6 (§13.11), most
> windows are predicted "not engaged" (04 §1: F1 0.09).

> **O12.** An expert whose activity has no training rows learns μ = 0, σ = 1 (hand-weighted: raw
> feature values go into the sigmoid) or predicts the constant 0.5 (learned). Under
> leave-one-group-out with one session per activity (04), every held-out activity is in this
> situation.

## 13.9 The gate and the mixture

**Code:** `src/spu/gate.py`, `src/spu/context.py`, `src/experts/mixture.py`.

**Gate.** A random forest (`gate = random_forest`, 300 trees, `min_samples_leaf = 5`) trained on
the scene features (`scene.*`; plus Vec if `gate_uses_vec`, false) of **all** training windows,
labelled or not, to predict the activity (`01`–`06`). `predict_proba` (line 30) returns six
probabilities in class order; **classes absent from training get probability 0**.

**Mixture** (`ExpertMixture`, line 17):

- `expert_scope = "per_experiment"` (config): six experts, each trained only on rows of its own
  activity; `"shared"`: one expert on all rows, used for all six.
- `mode = "soft"` (config): score = Σ over activities of gate probability × that activity's expert
  probability (line 47). `"hard"`: only the most probable activity's expert.

**Context source.** *Classified*: the gate's probabilities. *Oracle*: a one-hot vector on the true
activity (`context_probs`). Every row of a window uses its window's probabilities.

> **O13.** Combining O12 with this: in 04, the held-out group's activities are absent from training,
> so the gate gives them probability 0 and the *classified* score is made entirely by **other**
> activities' experts; the *oracle* score uses only the held-out activity's expert, which predicts
> the constant 0.5 for every row. That is why the oracle rows in 04 §1 show AUC 0.50 and F1 0
> (0.5 is below the 0.6 cut-off): they are the constant, not a measurement.

## 13.10 The window score

**Code:** `predict_windows` (line 102), `window_scores` (line 78); `fit_rows` (line 85) for `fill`.

The window score is the **plain mean of its rows' probabilities**: in late mode, the mean over every
tracked person in the window, **linked or not**. A window with no rows (no visible track) gets
`fill`, the training rows' label rate (in late mode the student-row engaged rate, 0.64 on the six
sessions; no window was empty there).

The module docstring of `levels.py` states the intent: "for late aggregation that estimates the
share of engaged students (which is how the window label itself is defined, §8.1)".

> **O14.** For the mean to estimate the share, each row's probability must be a calibrated
> probability of *that student* being engaged. Two things in the current implementation affect
> this: balanced class weights (O10: the mean sits near 0.5 instead of the real rate) and
> unlinked people (teachers, TAs, 8% of rows) being averaged in at inference and evaluation even
> though the share counts only annotated students.

> **O15.** Every tracked person counts equally, regardless of how many frames of the window they
> are visible in (as long as they pass the 50% coverage rule, Chapter 6, §6.4).

## 13.11 The cut-off

**Code:** `decision_threshold` in `src/experts/levels.py` (line 26); used in `train_models.py`
line 194 and saved in the model bundle as `tau`.

- Late: the cut-off is `labels.tau_share` (0.6): the window is predicted engaged if its score ≥ 0.6,
  mirroring the label rule "share ≥ `tau_share`".
- Early: `mixture.tau` (0.5), since the score is P(window engaged).

The `majority` baseline is cut at **0.5**, not at the variant's cut-off (`train_models.py`
line 226): its constant score is the training windows' engaged rate, so it predicts "engaged"
whenever more than half the training windows were engaged.

> **O16.** The 0.6 cut-off assumes the score is on the scale of the share (O14). With scores
> centred near 0.5 (O10, O11), the cut-off at 0.6 pushes predictions toward "not engaged", which
> moves F1 and balanced accuracy but not AUC.

> **O17.** The models and the `majority` baseline are cut at different thresholds (0.6 vs 0.5) in
> late mode, so their F1 values are not computed under the same rule.

## 13.12 The evaluation loop

**Code:** `scripts/train_models.py`, `main` (line 396), `heldout_splits` (line 102),
`src/spu/splits.py` (`logo_splits`).

For each group `g` (three folds, groups in sorted order):

1. **Split.** Training: every window of the other two groups (plus their augmented copies, if any);
   test: the original windows of `g` (augmented copies are never scored).
2. **Gate.** Fit on the training windows' scene features and activities; predict the test windows.
   Gate accuracy and macro-F1 are recorded.
3. **Labelled test windows** = those with a window label (§13.5). If none, the fold is skipped.
4. **`majority`**: constant = mean window label of the training windows, scored at 0.5 (O17).
5. **Each variant** (`configured`, `hard_top1`, `shared_expert`, `hand_weighted`): fit on the
   training rows that have a label (`fit_rows`: rows of training windows with a finite `y`); for
   each context (classified, oracle) predict every test window (§13.10), then:
   - `binary_metrics(window label, score, cut-off)` on labelled test windows (§13.13);
   - the same binary metrics per activity (`per_experiment`).
6. **Summary**: for each metric, the mean and standard deviation over folds, **ignoring NaN**
   (`summarize`, line 181; `np.nanstd`, population standard deviation, ddof = 0).

After the folds, the final models are fitted on all windows and saved for inference (they are not
evaluated).

> **O18.** The "±" in results is the population standard deviation over three folds (ddof = 0),
> which is smaller than the sample standard deviation (ddof = 1) by a factor of √(2/3) ≈ 0.82.

> **O19.** A fold where the test windows have one class gets AUC and balanced accuracy NaN, and the
> summary averages the remaining folds without saying how many were used.

## 13.13 How each metric is computed

**Code:** `src/evaluation/metrics.py`, `binary_metrics` (line 10).

| Metric | Exactly |
|---|---|
| `f1` | sklearn `f1_score(y, score ≥ cut-off)` for the **engaged** class (positive = 1); 0 when undefined |
| `auc` | sklearn `roc_auc_score(y, score)`; NaN if the fold's test labels have one class |
| `balanced_acc` | sklearn `balanced_accuracy_score(y, score ≥ cut-off)`; NaN if one class |
| `n`, `pos_rate` | labelled test windows, and their engaged share |

All are computed **per window, pooled within the fold** (both test sessions of a group together),
then averaged over folds (§13.12).

> **O20.** *Resolved by removal (16 November 2026).* The removed `retention_check` returned NaN for a
> whole session if any window's share was NaN; recorded with the metric in `docs/misc/proposed_metrics.md`.

> **O21.** The binary metrics pool the two held-out sessions of a fold, which mixes their levels
> (Chapter 12, §12.6). No reported metric is computed inside each session.

## 13.14 Label agreement (Krippendorff's α)

> **Superseded by experiment 06 Part 1** (`docs/experiments/06_full_pipeline_and_labels.md` §1.5–1.7,
> `scripts/labels/label_agreement.py`): α on entry moments, per second, window midpoints and window means,
> under step, ramp and Gaussian interpolation, raw and binary, with sample sizes, plus the human
> ceiling. It reproduces the numbers below exactly, answers O22 (binary α is lower than raw: −0.03 to
> −0.06 for the experts) and finds α between −0.06 and +0.04 overall in every setting.

**Code:** `src/evaluation/agreement.py`, `krippendorff_alpha_ordinal`;
`scripts/data_visualization/inspect_labels.py` (units from line 61).

The α function is the standard ordinal Krippendorff's α: it builds the coincidence matrix of all
pairs of values given to the same item, weights disagreements by the ordinal distance (which
depends on how many ratings fall between the two values), and returns 1 − observed / expected
disagreement. Items with fewer than two ratings are ignored.

How `inspect_labels` builds the items:

- once per (group, experiment), using **every** subject of the session (not filtered by camera);
- an item is (student, window midpoint); each labeler contributes **the value in force at the
  midpoint** (`at(mid)`), not the window's time-weighted mean;
- items are pooled across the three groups for each activity: one α per activity, with and
  without the self-report;
- the ratings are the **raw 1–5 values**.

> **O22.** The agreement is measured on raw point ratings at window midpoints, while the target is
> a *thresholded median of time-weighted means* (§13.3–13.5). Agreement on the binary student label
> (each labeler's own call at `tau_att`) or on the window label is not computed anywhere. The two
> can differ: two labelers who say 2 and 1 disagree on the 1–5 scale but agree on "not engaged".

> **O23.** The consensus distribution printed by `inspect_labels` uses every labeler, ignoring
> `labels.exclude_self`; with the config value (false) this makes no difference.

## 13.15 At inference

**Code:** `src/core/pipeline.py`, `Pipeline.predict` (line 160; the window-grid branch from line 175).

1. The gate predicts activity probabilities per window from the scene features.
2. Rows are built as in training (late: every visible track, Vec appended if the trained model
   used it, columns fixed to the trained ones).
3. The window score is the mean row probability (§13.10); `engaged` = score ≥ the bundle's `tau`.
4. `retention_graph.png` additionally draws a smoothed curve (mean over `output.smooth_windows` = 5
   windows); the JSON keeps the raw scores.

There is no association at inference (no annotations exist): every tracked person, student or not,
is averaged in. With the labeler ensemble, every labeler's model is averaged for every row, since
nothing says which labelers would have rated whom.

> **O24.** In training, unlinked people are excluded from the *training* rows but included in the
> *evaluated* window score (§13.10); at inference they are included in the score. The trained
> experts have never been trained on a non-student, but score them.

## 13.16 Trace: what produced each row of 04 §1

Putting the above together for the six-session run (one session per activity, each activity in
one group):

| Row of 04 §1 | What actually happened |
|---|---|
| `majority` | constant = training windows' engaged rate, cut at 0.5. F1 varies by fold with the held-out group's balance (12%, 69%, 94% engaged). |
| `configured/classified` | the held-out group's two activities have no training rows → their experts are constant 0.5 (O12), but the gate gives them probability 0 (absent classes), so each window is scored by the *four training activities'* experts, weighted by how much the room looks like theirs. |
| `configured/oracle` | weight 1 on the held-out activity's expert → constant 0.5 → AUC 0.5, F1 0 at cut-off 0.6 (O13). |
| `shared_expert/*` | one expert trained on the four training sessions; the gate is irrelevant (every column is the same expert), so classified = oracle. |
| `hand_weighted/classified` | per-activity standardisation from the training activities' rows, mixed by the gate; scores centred near 0.5, cut at 0.6 → few "engaged" predictions (O11). |
| `hand_weighted/oracle` | the held-out activity's hand-weighted expert has μ = 0, σ = 1 (no rows), so raw feature values go into the sigmoid. |
| Gate accuracy 0.00 | the held-out activities never appear in training, and absent classes get probability 0, so the gate can never predict them. |

## 13.17 All observations

| # | Where | What happens | Bears on |
|---|---|---|---|
| O1 | `attention.py:33` | a labeler is NaN for a whole window that starts before their first rating | label coverage at session start |
| O2 | `attention.py:22` | the last rating holds to the end of the session | labels after a labeler stopped |
| O3 | `session.py:21` | the label pool is the camera's annotated students, whether tracked or not | share vs score denominators |
| O4 | `window_labels.py:27` | consensus = median of time-weighted means; with 5 labelers, 3 must average ≥ `tau_att` | target definition |
| O5 | config `exclude_self` | the self-report is one of the five votes | target definition |
| O6 | `window_labels.py:43` | window label = threshold of a threshold | target granularity |
| O7 | `window_labels.py:43` | windows with < 3 labelled students: no label, but a share | metric populations differ |
| O8 | `association.py:66` | static boxes, IoU ≥ 0.3, ≥ 50% of frames; 8% of rows unlinked | who gets a training label |
| O9 | config `use_class_context` | every student's row carries the same class Vec | session-level offsets |
| O10 | `expert.py:42` | experts trained class-balanced; mean P 0.52 vs 64% engaged | score scale |
| O11 | `expert.py:115` | hand-weighted score centred near 0.5 by construction | score scale |
| O12 | `expert.py:52`, `expert.py:109` | an expert with no rows: constant 0.5 (learned) or raw features (hand-weighted) | held-out activities |
| O13 | `gate.py:30`, `mixture.py:44` | absent classes get gate probability 0; oracle uses the empty expert | 05 oracle and configured rows |
| O14 | `levels.py:78` | window score = mean P over all tracked people, read as the share | score vs target |
| O15 | `levels.py:78` | every tracked person counts equally | score vs target |
| O16 | `levels.py:26` | late cut-off = `tau_share` on a score centred near 0.5 | F1, balanced accuracy |
| O17 | `train_models.py:435` | `majority` cut at 0.5, models at the variant's cut-off | baseline comparison |
| O18 | `train_models.py:181` | "±" is the population SD over 3 folds | reported spread |
| O19 | `train_models.py:181` | NaN folds silently dropped from means | reported means |
| O20 | (removed) | one NaN share made a session's `retention_r` NaN; resolved by removing the metric | – |
| O21 | `metrics.py:10` | binary metrics pool the two held-out sessions | level vs shape |
| O22 | `inspect_labels.py:61` | α on raw midpoint ratings, not on the thresholded target | label-quality evidence |
| O23 | `inspect_labels.py` | its consensus ignores `exclude_self` | label report |
| O24 | `pipeline.py:160` | non-students scored and averaged in at evaluation and inference | score vs target |
| O25 | `attention.py:22` | a rating is read as "from now on", with no reaction delay or retrospective span | label timing |
| O26 | `window_labels.py:38` | half of student consensus values are exactly 3.0, on the `tau_att` threshold | label stability |

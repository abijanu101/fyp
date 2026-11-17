# Chapter 12 — Reading the results: a primer

The experiment documents (`docs/experiments/`), especially `04_six_session_run.md`, use a
vocabulary of thresholds, metrics and evaluation choices that the earlier chapters define only in
passing. This chapter builds it from scratch, in the order you need it: what is being predicted,
how the labels are made (and why they are shaky), how a model is tested, what each number
measures, and how the *way* a number is computed changes it. Every example uses numbers from the
six-session run, so you can check each idea against the document.

If you only need a one-line definition, Chapter 9 (§9.3–9.4) has them. This chapter is for
understanding why a number is high or low. Chapter 13 then documents exactly how the code
computes each of these, step by step, with every implementation choice listed.

## 12.1 What is being predicted

The pipeline's output is a **retention curve**: one value per **window** of a session. By default
(per-labeler training, Chapter 7 §7.9) that value is a predicted 1–5 rating, with one curve per
labeler model and their median. The window-grid paths, used in 04 and throughout this chapter,
output an engagement score instead. A window's length and overlap are config settings (`window.length_s`, `window.overlap`). The
six-session run, used throughout this chapter, had 8 seconds starting every 4 (50% overlap), so a
5-minute session has 73 windows (Chapter 6, §6.2); the config has used 4 seconds since 16 November
2026.

To train and test a model we need a "true" value per window to compare with. DIPSER gives
attention ratings per *student*; Chapter 7 (§7.6) turns them into window values in four steps.
Here they are on an example window with five labelled students:

| Step | What happens | Example |
|---|---|---|
| 1. Each labeler's mean | each labeler's 1–5 rating, averaged over the 8 seconds (time-weighted, because ratings are change points that hold until changed) | student A: labeler 1 → 3.0, labeler 2 → 2.5, labeler 3 → 3.0, labeler 4 → 4.0, self → 3.0 |
| 2. Consensus | the **median** over that student's labelers | A: median(3.0, 2.5, 3.0, 4.0, 3.0) = **3.0** |
| 3. Student engaged? | consensus ≥ **`tau_att`** | with `tau_att = 3`: A is engaged (3.0 ≥ 3) |
| 4. Window | **labelled share** = engaged students ÷ labelled students; the window is engaged if share ≥ **`tau_share`** | 3 of 5 engaged → share 0.6; with `tau_share = 0.6` the window is engaged |

So each window has two targets:

- the **labelled share**, a number between 0 and 1 (in 04 it is the grey line in the plots; the
  model's score should follow it);
- the **window label**, engaged (1) or not (0): the share cut at `tau_share`. Classification
  metrics (§12.4) compare against this.

In the window-grid paths with **late aggregation** (the 04 setup) each *student* also gets their
own label (step 3), and the experts are trained on those student labels. The window score is then the mean of the students'
predicted probabilities, an estimate of the share.

## 12.2 The `tau`s

"tau" (τ) is just the conventional name for a threshold. Three of them appear in the results:

| Name | Cuts | Value in 04 | Meaning |
|---|---|---|---|
| `labels.tau_att` | a student's consensus attention (1–5) | 3 (design: 4) | "a student counts as engaged if their labelers' median rating is at least this" |
| `labels.tau_share` | a window's labelled share (0–1) | 0.6 (design: 0.5) | "a window counts as engaged if at least this share of its students are engaged". In late mode, also the **cut-off** for the model's score: score ≥ 0.6 → predicted engaged |
| `mixture.tau` | the model's score in **early** mode | 0.5 | the cut-off when the model predicts the window label directly; unused in late mode |

Why 3 and 0.6 instead of the design's 4 and 0.5: labelers almost never give 4 or 5, and the median
removes most of the few they do give, so with `tau_att = 4` only 0.5% of windows are engaged. The
values used in 04 are **provisional**, chosen to give the pipeline both classes to learn from,
not because they are right (Chapter 11, §11.5).

The cut-off only matters for metrics that turn a score into a yes/no (accuracy, F1). Metrics that
use the score itself (AUC, correlation) do not depend on it, which is one reason 05 prefers them.

## 12.3 How good are the labels? Agreement and Krippendorff's α

A model can only be judged against labels that mean something. If the labelers do not agree with
each other, there is no single truth to predict, and even a perfect model would look mediocre.

**Raw agreement is misleading.** Suppose two labelers rate four students:

| Student | Labeler 1 | Labeler 2 |
|---|---|---|
| A | 3 | 3 |
| B | 3 | 3 |
| C | 3 | 3 |
| D | 3 | 2 |

They agree on 75% of students. But labeler 1 says 3 to everyone. Two labelers who *both* mostly say
3 will agree most of the time by chance, whether or not they see the same thing.

**Krippendorff's α** corrects for that:

  α = 1 − (observed disagreement) / (disagreement expected by chance)

- *Observed disagreement*: how often (and, for ordinal data, how far) labelers differ on the same
  item.
- *Expected disagreement*: how often two ratings drawn at random from **all** the ratings given
  (here: seven 3s and one 2) would differ. If almost everything is a 3, chance disagreement is low.

In the table, observed and expected disagreement are both 0.25 (one mismatched pair out of the
possible ones in each case), so **α = 1 − 0.25/0.25 = 0**: the 75% agreement is exactly what chance
predicts. Change student C to (2, 2) and the labelers now agree on a 2 as well, which chance makes
less likely: α rises to about 0.5.

Reading α:

| α | Meaning |
|---|---|
| 1 | perfect agreement |
| ≥ 0.8 | reliable (a common standard); 0.67–0.8 tentatively usable |
| 0 | agreement no better than chance |
| < 0 | systematic disagreement |

*Ordinal* α (what this project uses) also counts **how far apart** two ratings are: 2 vs 3 is a
smaller disagreement than 1 vs 5. Our items are (student, window midpoint) pairs, each labeler
contributing the rating in force at that moment (`scripts/data_visualization/inspect_labels.py`).

**On DIPSER, ordinal α is between −0.08 and 0.13 per activity** (Chapter 11, §11.5). The labelers
use the scale differently (one sits at 3 for 88% of the time, another at 2 for 73%), so their
agreement is close to chance. Consequences for reading 05:

- every number in 04 is measured against noisy labels; a modest score may be close to the best
  achievable;
- a model can look good by learning a *labeler's habits* instead of student behaviour;
- part of a student's label changing over time is labeler noise (04 §6.2).

The published works report agreement differently (each labeler against the majority vote, F1
0.41–0.61; `docs/misc/benchmarks.md` §3.1), but it tells the same story.

## 12.4 Testing a model: folds, baselines, variants

### Leave-one-group-out

DIPSER has three **groups** of students (different people, rooms and days). To test whether a
model works on people it has never seen, training uses two groups and testing the third: the
**held-out** group. Doing this three times, once per group, gives three **folds**. A number like
"AUC 0.51 ± 0.14" is the mean over the three folds ± their standard deviation. With only three
folds, the ± is itself very uncertain.

In 05 each group has two sessions, so each fold trains on four sessions and tests on two.

### Trivial baselines

A **trivial baseline** predicts something without looking at the input, to show what a number
means when nothing has been learned:

- **`majority` / train mean**: always predict the training folds' engaged rate (or mean share).
- **Constant 3** (in `benchmarks.md`): always predict attention 3.

A model is only useful where it beats the baseline **on the same metric computed the same way**.
Some metrics flatter baselines (§12.5); that is the main reason to always show one.

### The variants in 04

| Variant | What it is |
|---|---|
| `configured` | the mixture as configured: six per-activity experts, weighted by the gate |
| `shared_expert` | one expert for all activities |
| `hand_weighted` | fixed hand-written weights (no learning); a sanity baseline |
| `majority` | the trivial baseline |
| *late, no context* | `engagement.late_context = false`: the experts see each student's own features but not the class summary Vec |
| *minus flow* | no model: the class's negative mean residual flow used directly as the score |

**Gate, classified, oracle.** The *gate* predicts which of six activities is happening; the
*experts* score engagement per activity. *Classified* results weight the experts by the gate's
predictions (what a real deployment gets); *oracle* results use the true activity, as if the gate
were perfect. In 05 every activity appears in only one group, so the held-out group's activities
are never seen in training: the gate scores 0% and the oracle rows are uninterpretable by
construction (04 §4). Only `shared_expert` and `hand_weighted` are meaningful there.

## 12.5 The metrics

### From a score to a yes/no: the confusion matrix

Cut each score at the cut-off and compare with the label:

| | Labelled engaged | Labelled not engaged |
|---|---|---|
| **Predicted engaged** | true positive (TP) | false positive (FP) |
| **Predicted not engaged** | false negative (FN) | true negative (TN) |

From those four counts:

- **Accuracy** = (TP + TN) / all. Easy to read and easy to fool: if 79% of windows are engaged,
  "always engaged" scores 79%.
- **Precision** (for "engaged") = TP / (TP + FP): of the windows called engaged, how many were.
- **Recall** = TP / (TP + FN): of the engaged windows, how many were found.
- **F1** = 2 · precision · recall / (precision + recall), the harmonic mean: high only if both are.
  F1 is computed **for one class**. "F1 engaged" and "F1 not engaged" are different numbers.
- **Macro F1**: the plain average of the per-class F1s; every class counts equally.
- **Weighted F1**: the average weighted by class frequency, so the common class dominates.
- **Balanced accuracy**: the average of the recalls of both classes. Chance = 0.5 whatever the
  class balance.

**How a constant games them.** With 36% of student-windows not engaged (04 §3.1):

| Constant | Accuracy | F1 not engaged | Balanced accuracy | AUC |
|---|---|---|---|---|
| always "engaged" | 0.64 | 0 | 0.50 | 0.50 |
| always "not engaged" | 0.36 | 2·0.36/(1+0.36) = **0.53** | 0.50 | 0.50 |

An F1 for the minority class rewards predicting that class everywhere (recall 1, precision = its
share, so F1 = 2s/(1+s)). That is why 05 notes that the shared expert's not-engaged F1 of 0.50 is
*below* a constant. Balanced accuracy and AUC cannot be gamed this way.

### AUC: ranking, no cut-off

**AUC** (area under the ROC curve) asks: pick one engaged and one not-engaged window at random;
how often does the engaged one get the higher score? Example with four windows:

| Window | Label | Score |
|---|---|---|
| w1 | engaged | 0.7 |
| w2 | engaged | 0.4 |
| w3 | not | 0.5 |
| w4 | not | 0.2 |

The four engaged/not pairs: (w1, w3) right, (w1, w4) right, (w2, w3) **wrong**, (w2, w4) right →
AUC = 3/4 = 0.75.

- 1 = perfect ranking, **0.5 = chance**, 0 = perfectly reversed.
- An AUC **below 0.5** means higher scores go with the *other* class. In 04 §6.3 a single feature
  with AUC 0.36 is useful: *low* values go with engaged, it is just signed the other way.
- AUC ignores the cut-off and the scale of the scores, so it compares models whose scores are not
  calibrated.

### Continuous targets: MAE, RMSE

When the target is a number (the labelled share, or a 1–5 rating):

- **MAE** (mean absolute error): the average distance between prediction and truth. MAE 0.155 on
  the share means the predicted share is off by 0.155 (15.5 percentage points) on average.
- **RMSE** (root mean squared error): square the errors, average, take the root. Large misses count
  more, so RMSE ≥ MAE always; a big gap between them means a few big misses.

Both only make sense when the score *is* an estimate of the target (a share predicted as a share).
The hand-weighted score and "minus flow" are on arbitrary scales, so 05 leaves their MAE blank.

### Correlation: does it move with the target?

- **Pearson r**: how well a straight line relates two series, from −1 through 0 (no linear
  relation) to +1. It ignores offset and scale: a curve that is always 0.3 too high but rises and
  falls exactly with the labels has r = 1. A constant has no r (undefined, treated as 0).
- **Spearman ρ**: Pearson on the *ranks* instead of the values, so it only asks "when one goes up,
  does the other?", and is not thrown off by outliers or curved relations. 04 §6.3 uses it for
  features: ρ = −0.3 between class motion and the labelled share means "more motion, fewer students
  rated attentive", consistently but weakly.

### `retention_r`: the metric for the curve (removed, in 04's results)

**`retention_r`** was this project's own metric: for each held-out session, the Pearson r between
the model's score curve and the labelled-share curve over the session's windows; then the mean
over sessions. It was **removed from the code on 16 November 2026** and is documented as a proposal
for review in `docs/misc/proposed_metrics.md`; it is explained here because 04 reports it.

It answers the question a retention graph exists for: *when the class's engagement rises and falls
during a session, does the curve rise and fall with it?* Because Pearson ignores offset, it does
not care whether the curve sits at the right height, only whether it has the right **shape**.
Because it is computed **inside each session**, it cannot be inflated by getting the difference
between sessions right (§12.6). 0 = no relation (chance); in 04 the default models reach 0.01–0.12, the
shared expert without context 0.24, and the single feature "minus flow" 0.30.

Sessions where the score or the share never changes are skipped (no r exists).

## 12.6 The same metric, computed differently

A metric's name does not fix its value. Three choices change it, and papers rarely state them.

### Within versus between: where the variance is

The labelled share varies for two reasons: some sessions are more engaged than others (**between**
sessions), and engagement rises and falls during a session (**within**). The **between-session
share of variance** says how much of the target is the first kind. In 05 it is 69%: most of what
there is to predict is each session's overall level, which depends on group and activity, exactly
what leave-one-group-out holds out. The within-session movement is about ±0.1.

The same split exists for students (04 §6.2: 44% of student-label variance is *between* students,
some are generally rated attentive; 56% *within* a student over time).

This matters because a model can score well by getting the **level** right and still have no idea
**when** engagement changes. Per-session metrics (such as the removed `retention_r`) measure the "when"; pooled
metrics mix in the "level".

### Pooled, per session, per fold

| Choice | How | What dominates |
|---|---|---|
| **Per fold** | score each held-out group, average the three (what `train_models` reports) | each fold's own ranking |
| **Pooled** | put all held-out predictions from all folds in one pile, score once | differences *between* groups and sessions |
| **Per session** | score each session, average over sessions | movement *within* sessions |

04 §3.1 shows the size of the effect on the same predictions: the shared expert's window AUC is
0.51 per fold, **0.85 pooled** and 0.56 per session. Pooled, it is rewarded for ranking the three
held-out groups' overall levels in the right order (its mean scores 0.35, 0.53, 0.88 against
engaged rates of 12%, 69%, 94%), three points, not a skill. The same pooling makes the train-mean
baseline look worse than chance (AUC 0.13): holding out a highly engaged group *lowers* the
training mean, so the constant is anti-correlated with the held-out level by construction. Rule:
**do not pool predictions across leave-one-group-out folds**.

### Per window (or per second) versus per change

Labels are change points: a value holds until a labeler changes it. Counting every window (or
every second) gives a stretch that held for ten minutes 150 times the weight of one that held for
four seconds. Counting **per change** gives each run of an unchanged label weight 1 in total.

- Per window favours what is common and long-lasting, which constants are good at.
- Per change gives every episode, including short dips in attention, equal say.

On the DIPSER labels the constant "predict 3" scores accuracy 0.62 per second but 0.45 per change
(`benchmarks.md` §1); in 04 §3.1 the shared expert's student-level AUC falls from 0.61 to 0.53.
Neither is "right"; they answer different questions, and a result should say which it used.

## 12.7 Reading 05, section by section

| 05 section | What to look at | Concepts |
|---|---|---|
| §1 Setup | `tau_att = 3`, `tau_share = 0.6` are provisional | §12.1–12.2 |
| §2 Pipeline | ms per frame, kept tracklets, windows | Chapters 5–6 |
| §3 Results table | AUC and `retention_r` first; F1 against `majority`; ± is over three folds; ignore oracle and gate rows | §12.4–12.5 |
| §3.1 Benchmark metrics | baseline row first; per session over pooled; per window vs per change | §12.5–12.6 |
| §4 Why the numbers say little | one session per activity breaks gate and oracle; base rates differ by activity | §12.4 |
| §6.1 Target | 69% of variance between sessions | §12.6 |
| §6.2 Student labels | between vs within students; labeler noise | §12.3, §12.6 |
| §6.3 Features | within-session ρ; single-feature AUC below 0.5 = negative sign | §12.5 |
| §6.4 Models vs one feature | `retention_r` of models vs "minus flow" | §12.5 |

A quick checklist for any results table in this project:

1. **What is the target?** Window label, labelled share, or student label; which `tau`s.
2. **What is the baseline on this metric?** If none is shown, compute it (`scripts/labels/label_baselines.py`
   for label-level constants).
3. **Does the metric depend on the cut-off?** If yes, is the cut-off principled or tuned?
4. **How was it aggregated?** Per fold, pooled or per session; per window or per change.
5. **Level or shape?** Would getting each session's average right be enough to score well?
6. **How noisy are the labels?** With α near 0, small differences between models are not
   meaningful.

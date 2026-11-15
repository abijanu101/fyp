# Chapter 7 — Models: gate, experts, labels

After the feature pass, every window is a row of numbers. This chapter covers what is learned
from those rows: the context gate that predicts the activity, the per-activity engagement
experts, the mixture that combines them, the labels they are trained on, and the evaluation
protocol that keeps the numbers honest.

## 7.1 One recipe for every classical model (`src/core/classifiers.py`)

The gate and the experts are both scikit-learn classifiers, and they share one construction
recipe:

```python
CLASSIFIERS = {
    "random_forest": lambda seed, **p: RandomForestClassifier(random_state=seed, **p),
    "svm":           lambda seed, **p: SVC(probability=True, random_state=seed, **p),
    "logreg":        lambda seed, **p: LogisticRegression(max_iter=2000, random_state=seed, **p),
    "knn":           lambda seed, **p: KNeighborsClassifier(**p),
}

def build_classifier(name, seed=0, **params) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", CLASSIFIERS[name](seed, **params)),
    ])
```

Every model is therefore three steps:

1. **Impute**: replace NaN with the column's median *from the training data*. This is where
   "no student visible in this window" (NaN) becomes a number.
2. **Scale**: shift and rescale every column to mean 0, standard deviation 1, again using
   training statistics. Logistic regression, SVM and k-NN are sensitive to scale; without this, a
   column measured in hundreds would dominate one measured in fractions.
3. **Classify.**

Because the imputer and scaler are fitted inside the scikit-learn `Pipeline`, they only ever see
training data. Fitting them on all data first would leak test-set statistics into training.

## 7.2 The context gate (`src/spu/`)

### What it predicts

The gate outputs, for each window, probabilities over the six activities:
`g = [p(01), p(02), ..., p(06)]`. At training time the true activity is free: it is the
experiment folder (`SessionKey.exp_id`).

### The class

```python
class ContextGate:
    def __init__(self, model, classes, seed=0, **params):
        self.classes = list(classes)
        self.model = build_classifier(model, seed, **params)

    def fit(self, X, y): ...

    def predict_proba(self, X):
        """(n, len(classes)) in self.classes order; classes absent from training get 0."""
```

`predict_proba` re-orders scikit-learn's columns into our fixed class order and adds zero columns
for any class the training data lacked. Downstream code can then always assume column k means
activity `classes[k]`.

### Its input

By default the gate reads only the scene features (`scene.*` columns), so it does not depend on
tracking at all. Setting `gate_uses_vec: true` also gives it Vec: that is the §7 variant testing
whether track-derived features help context prediction.

### Oracle versus classified context (`src/spu/context.py`)

```python
def context_probs(mode, classes, true_exp=None, gate=None, X_gate=None):
    if mode == "oracle":        # one-hot on the true experiment: perfect context
        ...
    if mode == "classified":    # the gate's prediction: what a deployed system gets
        return gate.predict_proba(X_gate)
```

Running the experts on oracle context measures how good the experts are when context is known
exactly. Running them on the classified context measures the real system. The design asks for
both, with **classified as the headline number**.

## 7.3 The engagement experts (`src/experts/expert.py`)

An expert maps Vec to the probability that a window is engaged. There is one expert per activity,
because what "engaged" looks like differs: phone use is on-task in the online quiz and off-task in
a lecture (§8.2).

```python
class EngagementExpert(ABC):
    @abstractmethod
    def fit(self, X, y): ...
    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:   # (n,) probability of "engaged"
```

Two implementations (three registry names):

- **`SklearnExpert`** (registered as `logreg` and `random_forest`). It always uses
  `class_weight="balanced"`, so the minority class is not ignored. If a training fold contains
  only one class, it cannot fit a classifier, so it predicts that class's rate as a constant
  instead of crashing.
- **`HandWeightedExpert`** (`hand_weighted`). The baseline every learned expert must beat (§8.2).
  It standardises a few chosen columns and applies fixed weights from the config
  (`hand_weighted.weights`), with the phone weight's sign flipped per activity
  (`phone_sign`). `fit` only learns the means and standard deviations; it never looks at labels.
  Weights for columns that are not in the input are skipped silently. The configured SCB model
  has no phone class, so `phone_key` (`vec.scb.using_phone.mean`) is currently absent and the phone
  term does nothing.

## 7.4 The mixture (`src/experts/mixture.py`)

```
p_engaged = sum_k  g_k * p_k(Vec)          (soft: weight each expert by the gate)
p_engaged = p_argmax(g)(Vec)               (hard: use only the gate's top activity)
```

```python
mix = ExpertMixture(classes, make_expert, scope="per_experiment", mode="soft")
mix.fit(X, y, exp)              # each expert trains only on windows of its own experiment
p = mix.predict_proba(X, g)     # g: (n, 6) context probabilities, oracle or classified
```

`make_expert` is a function that builds a fresh, untrained expert, so the mixture can create six
independent copies. With `scope="shared"`, all six slots hold *one* expert trained on everything.
That is the §8.3 comparison asking whether per-activity experts earn their data cost.

With early aggregation, the window is called engaged if `p_engaged >= mixture.tau` (0.5). Late
aggregation uses a different cut-off (§7.5).

## 7.5 Early or late aggregation (`src/experts/levels.py`)

This section answers one question: **what is one training example for an expert?** There are two
reasonable answers, and the config key `engagement.aggregate` picks between them. The default is
`late`; `early` is the design as written and is kept for comparison.

### A worked example

Take one 8-second window in which three students are visible. To keep the numbers small, pretend
the only feature is stillness (the share of frames a student sits still), and that the labels say:

| Student (track) | Stillness | Their label (consensus attention ≥ `tau_att`?) |
|---|---|---|
| A (TID 4) | 0.9 | engaged (1) |
| B (TID 9) | 0.2 | not engaged (0) |
| C (TID 11) | 0.8 | engaged (1) |

Two of the three labelled students are engaged, so the window's *share engaged* is 0.67. With
`labels.tau_share = 0.6`, the window label is "engaged", because 0.67 ≥ 0.6 (§7.6 has the full
rule).

**Early aggregation: the example is the window.** First squash the three students into Vec:
`stillness.mean = 0.63`, `stillness.std = 0.31`, and so on. The expert gets **one example**:

```
input  [0.63, 0.31, ...]   (Vec)          target  1   (the window label)
```

At prediction time it outputs one number, say 0.7, read as "probability this window is engaged",
and the window counts as engaged if that is at least `mixture.tau` (0.5).

**Late aggregation: the example is a student in a window.** No squashing first. The expert gets
**three examples**, each one student's own features, plus Vec as shared context when
`engagement.late_context` is true:

```
input  [0.9 | 0.63, 0.31, ...]   (A's own features | Vec)   target  1   (A's own label)
input  [0.2 | 0.63, 0.31, ...]   (B's own features | Vec)   target  0
input  [0.8 | 0.63, 0.31, ...]   (C's own features | Vec)   target  1
```

At prediction time it outputs one number per student, say 0.8, 0.3 and 0.7. The window still
needs a single score, so they are averaged: (0.8 + 0.3 + 0.7) / 3 = 0.6. That average is an
estimate of the share engaged, the same quantity the window label is built from, so it is compared
with the same threshold, `tau_share`.

The averaging happens **after** the model in late aggregation and **before** it in early
aggregation. That is all the names mean.

| | `early` | `late` (default) |
|---|---|---|
| One example per | window | visible student in a window |
| Its features | the window's Vec | that student's own features (`trk.*`), plus Vec if `late_context` |
| Its target | the window label | that student's own label |
| Window score | the one prediction | the mean of the students' predictions |
| Engaged if score ≥ | `mixture.tau` | `labels.tau_share` |

### Why "rows", and what a row must remember

In code an example is a **row** of a matrix `X`. Early aggregation has one row per window; late
aggregation has one row per student per window, so a 73-window session with about ten visible
students has about 730 rows.

Everything downstream works per **window**: cross-validation holds out whole groups of windows,
the gate predicts the activity per window, and the metrics and the retention graph are per window.
A late row on its own does not say which window it belongs to, so each row carries that as a tag.
With the tag, three operations work identically in both modes:

1. **Splitting**: a row goes to the training or test side with its window.
2. **Context**: a row gets the gate's activity probabilities of its window, which pick the experts
   that score it.
3. **Averaging back**: the window's score is the mean over the rows tagged with it. In early mode
   each window has one row, so the "mean" is that row's prediction unchanged.

Early aggregation is then simply the special case of one row per window, and one code path serves
both. The rows and their tags are bundled in one small dataclass:

```python
@dataclass
class EngagementRows:
    X: np.ndarray        # (R, F) expert inputs, one row each
    cols: list[str]      # names of X's columns
    window: np.ndarray   # (R,) which window (row of the window table) each row belongs to
    exp: np.ndarray      # (R,) the row's experiment, so per-experiment experts train on their own
    y: np.ndarray        # (R,) training label: 1 / 0, NaN = unlabelled (not trained on)
```

For three windows in which 2, 0 and 3 students are visible:

```
early:  row  window  X              late:  row  window  tid  X
         0     0     Vec(w0)                0     0      4   trk.*(tid 4, w0) + Vec(w0)
         1     1     Vec(w1)                1     0      9   trk.*(tid 9, w0) + Vec(w0)
         2     2     Vec(w2)                2     2      4   trk.*(tid 4, w2) + Vec(w2)
                                            3     2      9   ...
                                            4     2     11   ...
```

Window 1 has no late rows at all, because nobody was visible. It still needs a score, so it gets
`fill` (below).

### Four functions do the rest

(`levels.py` also has `fit_predict`, a wrapper that calls `fit_rows` then `predict_windows`.)

All of them are in `src/experts/levels.py`:

- **`early_rows(windows, vec_cols)`**: the window table's `vec.*` columns, window index `0..n-1`.
- **`late_rows(windows, tracks, context, cols=None)`**: the per-student table's `trk.*` columns,
  with the window's `vec.*` columns appended to each row when `context` is on. Pass `cols` to force
  a column list (at inference, the trained model's columns), so a session that happens to lack a
  column still gives a matrix of the right shape, with NaN in the missing column.
- **`fit_rows(make_mixture, rows, train_windows)`**: trains a fresh mixture on the rows whose window
  is in the training split *and* whose label is not NaN. It also returns `fill`, the training label
  rate, used below.
- **`predict_windows(mix, rows, windows, g, fill)`**: gives every row its own window's gate
  probabilities, predicts each row, then averages rows back to one probability per window
  (`window_scores`, a `np.bincount` mean). A window with no rows gets `fill`, because there is
  nobody to score.

Cross-validation splits **windows** (by group), never rows. Rows follow their window into the train
or test side, so students from a held-out group never appear in training.

Because early aggregation has one row per window, `predict_windows` returns that row's
probability unchanged: the early mode gives exactly what the old window-level code gave
(`test_early_rows_reproduce_the_window_level_mixture`). The gate, the metrics and the outputs are
window-level in both modes, so the two compare directly.

### Where the per-student rows come from

```
feature pass       WindowAggregator.aggregate_with_tracks(window, ...) -> (Vec, per-student rows)
                   SessionFeatures.track_rows: [{"window": i, "tid": 4, "kg.speed": ..., ...}, ...]
       │
       ▼  run_pipeline extract
disk               <session>.csv          one row per window   (vec.*, scene.*)
                   <session>.tracks.csv   one row per (window, tid)   (window, tid, trk.*)
       │
       ▼  train_models: engagement_rows(cfg, df, vec_cols)
training           early: early_rows(...), y = window label
                   late:  late_rows(...),  y = that student's label in that window
       │
       ▼  model bundle: mixture, "aggregate", "row_cols", "fill", "tau"
inference          Pipeline.predict rebuilds the same rows from SessionFeatures in memory
                   (to_track_frame) and calls predict_windows
```

The per-student rows are not new computation: the aggregator already computed every student's
features to build Vec, and used to throw them away after taking the mean and standard deviation.
It now also returns them. That is why switching modes needs no new frame pass, only
`run_pipeline extract --reuse` to write the `.tracks.csv` files from cached tracks.

### Labels for late rows

A late row needs the label of *that student* in *that window*, but the tracker only knows a TID.
At training time only, `train_models` links TIDs to subjects with `associate_sids`
(`src/tracking/association.py`): each track is matched, frame by frame, to our manual student
boxes by IoU (at least `association.min_iou`), and gets a subject only if that subject covers at
least `association.min_match_share` (50%) of the track's frames. It then asks
`student_labels_session` for each subject's label per window. That is the same median-of-labelers
and `tau_att` rule the window label counts with, minus the `tau_share` step, which only applies to
a whole class. A row stays unlabelled
(NaN, skipped in training) when its track matched no subject, for example a TA walking past, or
when its subject has no labels in that window. `data.row_label_coverage` in the results reports
the labelled share.

At inference nothing needs an identity: rows are predicted and averaged, and per-student
probabilities exist only inside `predict_windows`. Nothing per student is written to the graph.

### Two settings that depend on the mode

**The engaged / not-engaged cut-off.** The two modes produce different kinds of window score:

- early: the score is the probability that the window is engaged, so it is cut at `mixture.tau`
  (0.5);
- late: the score is an estimate of the share of engaged students. The window label is
  "share ≥ `labels.tau_share`", so the score is cut at that same `tau_share`.

`decision_threshold(aggregate, mixture_tau, tau_share)` picks the cut-off. `train_models` uses it
for F1 and balanced accuracy and saves it in the model bundle as `tau`;
`Pipeline.predict` reads it from there for the graph's `engaged` column. AUC and `retention_r` do
not use a cut-off, so they are unaffected.

One caveat: `SklearnExpert` trains with `class_weight="balanced"`, which pushes each student's
probability toward 0.5 when classes are imbalanced. The mean is then not an exact share, only a
score that rises with it, so the `tau_share` cut is the principled default, not a tuned optimum.
Compare modes on AUC and `retention_r` first.

**The hand-weighted baseline.** Its weights name window columns (`vec.kg.stillness.mean`, ...).
In late mode, `train_models.expert_factory` maps each one to the student's own feature with
`student_key` (`vec.kg.stillness.mean` → `trk.kg.stillness`), so the baseline scores each student
from that student's features, just as the learned experts do, and still works with
`late_context=false`.

### Consensus labels or a per-labeler ensemble (`engagement.labelers`)

By default (`"consensus"`) a late row's label is the student's consensus: the median over labelers
of their time-weighted mean rating, thresholded at `tau_att`. With labeler agreement near zero
(Chapter 11), that median mostly reflects the middle rater's habits and discards the rest.

`"ensemble"` keeps the labelers apart instead:

- **Training.** One mixture per expert labeler (`fit_ensemble`), each trained only on the students
  that labeler rated, on that labeler's own call (their time-weighted mean ≥ `tau_att`, the same
  rule as the consensus). Labelers in `engagement.ensemble_exclude` (by default the students'
  self-ratings, a different construct) and labelers who rated no training student are skipped.
- **Prediction.** A `LabelerEnsemble` averages its labelers' probabilities for each row, **over the
  labelers who rated that row's student**: a labeler who never rated a student has no vote on
  that student. Rows with no rating labeler (unlinked tracks such as teachers), and every row at
  inference, where nothing is known about labelers, average all of them.
- **Everything else is unchanged:** the gate, the mixture inside each labeler model (per-activity
  or shared experts), the window averaging, the cut-off and the metrics.

The per-labeler labels come from `labeler_labels_session` (`src/labels/session.py`) and ride on
`EngagementRows` as `y_by_labeler` and `rated_by`; `engagement_rows` builds them whenever the
ensemble is configured. Ensembles need late aggregation (per-student rows).

Why the "who rated" rule matters on DIPSER: the fourth expert slot is `labeler_04` in group 1,
split between `labeler_04` and `labeler_05` in group 2, and `labeler_05` in group 3
(`docs/experiments/01_dipser.md`). Averaging only over the student's own labelers keeps a
group-3 student from being scored by a labeler-04 model that never saw group 3's ratings.

### Why it might or might not help

Why it might help: the window label is itself a share of engaged students, so predicting students
and averaging matches its definition; there are about ten times more training rows; and a class
that is half engaged, half not stays visible instead of being averaged into Vec. Why it might not:
per-student labels are noisier than window labels, and association errors become label errors.

## 7.6 Labels: from labeler events to window labels (`src/labels/`)

### The raw format

Each student's ZIP contains one JSON file per labeler (`labeler_01.json` ... and
`self_labeling.json`, the student's own rating). Each file is a list of change events:

```json
[{"datetime": "10:40:43:047895", "attention": "3", "emotion": "9"},
 {"datetime": "10:45:25:006335", "emotion": "1"}]
```

Only entries with an `attention` key change attention; the value holds until the next one.

### `AttentionTrack`: a step function

`parse_events` (`src/labels/attention.py`) turns one labeler's list into an `AttentionTrack`:
sorted change times and values. It drops values outside 1–5 (DIPSER contains at least one
attention value of `7`).

```python
tr.at(t)               # value in force at time t; NaN before the first entry
tr.mean_over(t0, t1)   # time-weighted mean over [t0, t1); NaN if the window starts before
                       # the first entry (the student is "unlabelled" there)
```

`mean_over` weights each value by how long it held inside the window. If a labeler said 2 for 3
seconds and 4 for 5 seconds of an 8-second window, the mean is (2×3 + 4×5)/8 = 3.25.

### `WindowLabeler`: from students to a window label

`src/labels/window_labels.py` follows §8.1:

1. per student, per labeler: time-weighted mean attention over the window;
2. per student: the **median across labelers** (optionally excluding the self-report);
3. a student is engaged if that median `>= tau_att`;
4. the window is engaged if at least `tau_share` of its labelled students are engaged;
5. windows with fewer than `min_labeled_students` labelled students get no label (dropped).

It returns a `WindowLabel(engaged, share_engaged, n_labeled)`. `share_engaged` is kept as a soft
target for the retention-curve check.

`label_session` (`src/labels/session.py`) applies this to one session-camera, and
`student_labels_session` returns each student's own label per window (for late aggregation). With
`labels.students: "camera"`, it uses only the students annotated as visible in that camera.

### The label problem, in brief

At the design's defaults (`tau_att=4`, `tau_share=0.5`) almost no window is engaged (0.5%), and
the labelers barely agree (Krippendorff's α ≈ 0). Chapter 11 has the numbers. The plan is to use
provisional thresholds (`tau_att=3`, `tau_share=0.6`) to get the pipeline running end to end, then
return to label design as a separate phase.

## 7.7 The evaluation protocol: leave-one-group-out

Windows from the same session are near-duplicates (they overlap by half and show the same room
and students). If test windows came from sessions also in training, the model could score well by
recognising the room. So the evaluation always holds out **a whole group**: all its experiments
and cameras, therefore a different set of students.

```python
for held_out, train_idx, test_idx in logo_splits(groups):   # src/spu/splits.py
    ...  # fit on two groups, evaluate on the third
```

With three groups this is three folds. Every result is reported as mean ± standard deviation
across the three folds; with so few folds, the spread is as informative as the mean.

### What `train_models` reports

For each fold, `scripts/train_models.py` fits the gate and several mixture variants, and scores:

| Name in results | What it is |
|---|---|
| `configured/classified` | the configured experts with the gate's context: **the headline** |
| `configured/oracle` | the same experts with perfect context |
| `hard_top1/...` | only the gate's top activity's expert |
| `shared_expert/...` | one expert for all activities |
| `hand_weighted/...` | the fixed-weight baseline |
| `majority` | always predict the training fold's majority class |

Metrics come from `src/evaluation/metrics.py`: F1, AUC and balanced accuracy for engagement;
accuracy, macro-F1 and per-class precision/recall for the gate. AUC and balanced accuracy are NaN
on a fold whose test set has only one class, because they are undefined there.

The `retention_r` metric is the retention-curve check from §11: per session, the correlation
between the predicted engagement curve and the share of labelled students engaged.

## 7.8 The output graph (`src/output/graph.py`)

`build_graph` only accepts window-level arrays (context probabilities, engagement probabilities,
track counts), so per-student data cannot reach the output even by mistake. Each record:

```json
{"t_start": 0.0, "t_end": 8.0,
 "predicted_context": {"label": "03", "confidence": 0.81},
 "layout_family": "grid", "engaged": "Y", "engagement_prob": 0.67, "n_visible_tracks": 14}
```

`write_graph_png` smooths `engagement_prob` with a 5-window moving average and shades the
background by predicted activity.

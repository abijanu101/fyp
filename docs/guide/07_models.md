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

Three implementations:

- **`SklearnExpert`** (registered as `logreg` and `random_forest`). It always uses
  `class_weight="balanced"`, so the minority class is not ignored. If a training fold contains
  only one class, it cannot fit a classifier, so it predicts that class's rate as a constant
  instead of crashing.
- **`HandWeightedExpert`** (`hand_weighted`). The baseline every learned expert must beat (§8.2).
  It standardises a few chosen columns and applies fixed weights from the config
  (`hand_weighted.weights`), with the phone weight's sign flipped per activity
  (`phone_sign`). `fit` only learns the means and standard deviations; it never looks at labels.

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

The window is called engaged if `p_engaged >= mixture.tau` (0.5).

## 7.5 Labels: from labeler events to window labels (`src/labels/`)

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

`label_session` (`src/labels/session.py`) applies this to one session-camera. With
`labels.students: "camera"`, it uses only the students annotated as visible in that camera.

### The label problem, in brief

At the design's defaults (`tau_att=4`, `tau_share=0.5`) almost no window is engaged (0.5%), and
the labelers barely agree (Krippendorff's α ≈ 0). Chapter 10 has the numbers. The plan is to use
provisional thresholds (`tau_att=3`, `tau_share=0.6`) to get the pipeline running end to end, then
return to label design as a separate phase.

## 7.6 The evaluation protocol: leave-one-group-out

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

## 7.7 The output graph (`src/output/graph.py`)

`build_graph` only accepts window-level arrays (context probabilities, engagement probabilities,
track counts), so per-student data cannot reach the output even by mistake. Each record:

```json
{"t_start": 0.0, "t_end": 8.0,
 "predicted_context": {"label": "03", "confidence": 0.81},
 "layout_family": "grid", "engaged": "Y", "engagement_prob": 0.67, "n_visible_tracks": 14}
```

`write_graph_png` smooths `engagement_prob` with a 5-window moving average and shades the
background by predicted activity.

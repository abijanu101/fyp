# Chapter 9 — Running experiments

Chapters 5–8 explained what the code does. This chapter is about using it to answer questions:
"does ReID help?", "is late aggregation better?", "does SCB earn its cost?". It has two parts:

- **9.1–9.4: vocabulary.** The words that appear in results files and in these docs (oracle,
  fold, held-out, nested, the variant names, the metrics) and what each one means here.
- **9.5–9.7: experimenting through the config.** How a single JSON file drives every run, how to
  change it safely, and what every key of `configs/deliverable3.json` does.

## 9.1 What an experiment is in this codebase

An experiment is **a configuration plus the data, producing a results file named by the
configuration's hash**:

```
configs/deliverable3.json  +  --set overrides   ──►  config (one dict)  ──►  config hash  a1b2c3...
                                                                │
           run_pipeline extract  (features, cached by tracking and feature hash)
           train_models          ──►  outputs/results/a1b2c3....json   every metric, + the full config
                                      outputs/models/a1b2c3....joblib  the trained models
           plot_results          ──►  outputs/plots/a1b2c3.../         held-out curves per session
```

Two runs with the same config produce the same hash and overwrite the same file; any change,
however small, gives a new hash and a new file. A results file always contains the complete
config it came from, so a number can always be traced back to exactly what produced it. Quote
the hash with every number you report (Chapter 4, §4.4).

## 9.2 Data and evaluation terms

**Session.** One camera of one experiment of one group, e.g. `group_02/experiment_05/context_cam_01`.
52 exist.

**Group.** One set of students in one room (`group_01`–`group_03`). Each group did all six
activities. Everything about a group is shared across its sessions: the room, the lighting, the
camera angles, the people and their clothes.

**Window.** An 8-second slice of a session, with 50% overlap. The unit every engagement metric is
computed on.

**Labelled window.** A window with a label: at least `labels.min_labeled_students` (3) students had
a labeler's rating in it. Unlabelled windows are still predicted but not scored.

**Leave-one-group-out (LOGO).** The evaluation protocol. With three groups there are three
**folds**: in each, the models train on two groups and are scored on the third, the
**held-out group**. Every number is therefore about people and rooms the model has never seen.
Splitting windows randomly instead would put neighbouring, half-overlapping windows of the same
session on both sides, and the scores would mostly measure memorisation of the room.

**Fold scores and summaries.** Each metric is computed per fold; results files report every fold
and a summary of `mean` and `std` over the three folds. With three folds, `std` is a rough sense of
stability, not a confidence interval. Differences smaller than it should be treated as ties.

**Out-of-sample.** Predicted by a model that never saw that data. Everything in the results files
and in `plot_results` is out-of-sample; the final model saved for inference is trained on all three
groups and has no held-out score of its own.

**Nested LOGO.** Used by the tuning scripts (Chapter 8, §8.5). Choosing hyperparameters by their
score on the held-out group and then reporting that same score is optimistic, because the choice
already looked at the test data. Nested LOGO runs the search *inside* each fold, using only the two
training groups, and scores the winner once on the held-out group. `tune_models` reports both:
the **nested** estimate is the honest one; the **selected** parameters are what goes into the
config.

## 9.3 Context and model terms

**Context.** Which of the six activities is running (`01`–`06`, the experiment folder). Known
exactly in DIPSER, predicted in deployment.

**Gate.** The model that predicts context from scene features (§7.2). Scored by **accuracy** and
**macro-F1** (the F1 of each activity, averaged with equal weight, so a rare activity counts as
much as a common one).

**Classified versus oracle context.** Every engagement result appears twice:

- **classified**: the experts are weighted by the gate's *predicted* activity probabilities. This
  is what a deployed system gets, and it is the **headline number**.
- **oracle**: the experts are weighted by the *true* activity (a one-hot vector from the folder
  name), as if the gate were perfect. An "oracle" in evaluation is any component replaced by the
  ground truth, to measure how good the rest of the system could be.

The gap between them is the cost of gate errors. Oracle well above classified means improving the
gate is worth it; the two close together means the experts, not the gate, are the bottleneck.

**Expert, mixture, scope, mode.** An expert scores engagement for one activity; the mixture
combines the six (§7.4):

- `mixture.mode = "soft"`: weight every expert by the gate's probability for its activity;
  `"hard"`: use only the expert of the most likely activity.
- `mixture.expert_scope = "per_experiment"`: six experts, each trained only on its own activity;
  `"shared"`: one expert trained on everything, used for all six.

**Aggregation, rows, cut-off.** Late (default) or early aggregation, explained in §7.5: whether the
experts score each student and average per window, or score the class summary Vec. The **cut-off**
is the window score above which a window counts as engaged: `labels.tau_share` in late mode,
`mixture.tau` in early mode.

### The variants in a results file

`train_models` evaluates five engagement predictors, each with classified and oracle context,
under the names `<variant>/<context>`:

| Variant | What it is | The question it answers |
|---|---|---|
| `configured` | the mixture exactly as the config says | the result |
| `hard_top1` | same experts, `mode = "hard"` | does soft mixing help? |
| `shared_expert` | one expert for all activities | do per-activity experts earn their data cost? (§8.3) |
| `hand_weighted` | fixed weights, never sees labels | does learning beat a hand-written rule? (§8.2) |
| `majority` | always predicts the training set's engaged rate | the floor: anything at or below it has learned nothing |

`majority` has no context and appears once.

## 9.4 Metric and label terms

### Engagement metrics (per window, on labelled windows of the held-out group)

| Metric | Meaning | Uses the cut-off? | Chance level |
|---|---|---|---|
| `auc` | ROC AUC: the probability that a random engaged window scores above a random not-engaged one | no | 0.5 |
| `f1` | F1 of the "engaged" class: balances missed and false engaged windows | yes | depends on the class balance |
| `balanced_acc` | mean of the hit rates on engaged and on not-engaged windows | yes | 0.5 |
| `retention_r` | per session, the Pearson correlation between the score curve and the labelled share engaged, averaged over sessions | no | 0 |
| `n`, `pos_rate` | labelled windows scored, and the share of them labelled engaged | | |

AUC and `retention_r` are the most informative here: they do not depend on where the cut-off
sits, and `retention_r` asks directly whether the curve goes up and down with the labels, which
is what a retention graph is for. With a skewed class balance F1 can look good for a model that
always says "engaged"; compare it with `majority`.

`per_experiment` repeats the metrics for each activity separately, inside each fold.

### The `data` block

| Key | Meaning |
|---|---|
| `n_sessions`, `n_windows` | how much data was found |
| `label_coverage` | share of windows that have a window label |
| `pos_rate`, `pos_rate_by_exp` | share of labelled windows that are engaged, overall and per activity (`null` = no labelled window of that activity) |
| `aggregate` | `late` or `early` |
| `n_rows` | expert rows: windows (early) or student-windows (late) |
| `row_label_coverage` | late: share of student rows linked to a labelled student; low means association, not the model, is the limit |
| `row_pos_rate` | late: share of labelled student rows that are engaged |

### Label terms (§7.6)

**Consensus.** A student's attention in a window: the median over labelers of each labeler's
time-weighted mean rating (1–5). **`tau_att`**: a student is engaged if consensus ≥ `tau_att`.
**Share engaged**: engaged students ÷ labelled students in a window. **`tau_share`**: the window is
engaged if share ≥ `tau_share`. **Krippendorff's α**: agreement between labelers, from 1 (perfect)
through 0 (no better than chance); computed by `scripts/data_visualization/inspect_labels.py`.

### Tracking terms (Chapter 8, §8.6)

**IDF1**: how much of each student's time is covered by one consistent track id, with a one-to-one
assignment of tracks to students; the tracker objective. **Coverage**: share of (student, frame)
pairs matched by any track. **ID switches** and **track ids per subject**: fragmentation.
**Subjects found**: share of annotated students matched at all.

### Tuning terms

**Trial**: one parameter set, evaluated. **Trial 0 / baseline**: the current config's values, so
every search reports its gain over what you have. **Random search**: trials drawn independently
from the space (`--randomize`). **Beam bisection**: a binary search per numeric parameter, k
branches per node and a beam of the best nodes per level (the default for the tracker; Chapter 8,
§8.5). **Level** (bisection): one round of branching, up to (2k + 1) × beam trials. **Search space**: the ranges or lists trials are
drawn from. **Profile**: a named tracker search setup (which tracker, which sessions, how many
trials, which parameters are fixed). **Objective**: the metric a search maximises. **Fingerprint**:
the settings a resumable run checks before reusing logged results. **`config_patch`**: the winning
values, in the shape to paste into the config.

## 9.5 Experimenting through the config

Every stage is built from `configs/deliverable3.json` (Chapter 3), so almost every experiment is a
config change. There are three ways to make one.

**`--set` on the command line**, for one-off changes. Every script takes it:

```
python -m scripts.run_pipeline extract --reuse --set window.length_s=10 tracklets.min_visible_s=2
python -m scripts.train_models --set engagement.aggregate='"early"' labels.tau_att=3
```

The rules, all enforced by `parse_sets` and `set_dotted`:

- each pair is `dotted.path=value`, and the value is **JSON**: numbers and `true`/`false` as is,
  strings in double quotes (`'gate.name="svm"'`), objects and lists as JSON;
- put every pair after **one** `--set`; a second `--set` replaces the first list;
- **list elements cannot be addressed** (`bfmu.0.still_threshold` fails): pass the whole list, e.g.
  `'bfmu=[{...}, {...}, {...}]'`;
- **keys that contain dots cannot be addressed**: `aggregation.thresholds.kg.stillness=0.5` silently
  creates a new nested key and leaves `"kg.stillness"` unchanged. Set the whole dict instead:
  `'aggregation.thresholds={"kg.stillness": 0.5, ...}'`. The same holds for `hand_weighted.weights`;
- PowerShell 5.1 strips the inner double quotes before Python sees them. There, write
  `'gate.name=\"svm\"'`, or use Git Bash.

**A copy of the config file**, for an experiment you will come back to:
`configs/no_scb.json`, run with `--config configs/no_scb.json`. The hash, not the file name,
identifies the results, so a copy that ends up identical to another config reuses its results.

**`output_dir`**, to keep a set of runs apart, e.g. `--set output_dir='"tmp/e2e6"'`. It is excluded
from every hash, so the same config in two output folders has the same hash.

### What a change costs

Each key feeds one of three hashes (Chapter 4, §4.4), and the hash decides what must re-run. In
the tables below:

| Mark | Feeds | After changing it, run | Cost |
|---|---|---|---|
| **F** | tracking hash (and so everything) | `extract` (with or without `--reuse`: the frame-pass cache misses either way), then `train_models` | minutes per session |
| **P** | feature hash | `extract --reuse` (feature pass on cached tracks), then `train_models` | seconds per session |
| **T** | config hash only | `train_models` | seconds |
| **–** | nothing (excluded) | only the script that reads it | |

So cheap experiments (labels, models, aggregation mode, window length, thresholds) can be run many
times on one extraction, while anything marked **F** should be batched.

### Habits that keep results comparable

- **Change one thing per run.** The results file records the whole config, but a difference
  between two runs is only interpretable if one key changed.
- **Keep a baseline run** with the current config and compare every variant with it, fold by fold
  (results files keep per-fold numbers).
- **Read classified first**, oracle second; read AUC and `retention_r` before F1 (§9.4).
- **Treat differences within the fold `std` as ties.** With three groups, a 0.01 change in mean
  AUC is noise.
- **Check `majority` and `hand_weighted`.** A learned model that does not beat them has learned
  nothing usable.
- **Tune only after the labels are fixed.** The expert objective depends on `labels.*`.

Chapter 10 has step-by-step recipes for common experiments: a tracker setting (§10.5), a
sensitivity sweep (§10.9), early versus late aggregation (§10.10), and the cost of a change
(§10.11).

## 9.6 `deliverable3.json`, key by key

Values are the current ones. "Re-run" uses the marks of §9.5.

### Scope and identity

| Key | Value | What it does | Re-run |
|---|---|---|---|
| `seed` | 0 | random state of the gate and expert models | T |
| `dataset.root` | `data/DIPSER` | where frames, subject labels and `abi_labels` live | T |
| `dataset.groups` / `.experiments` / `.cameras` | all 3 / all 6 / all 3 | which sessions are in scope: all combinations whose ZIP exists (52). `extract` and the tuners process them, and `train_models` and `plot_results` use only their cached features, even when the feature folder holds more | T |
| `classes` | `"01"`…`"06"` | activity ids; must match the experiment folder numbers; fixes the gate's column order | T |
| `output_dir` | `outputs` | where caches, results, models and plots go | – |

### Frame pass: ingestion, detection, tracking, scene

| Key | Value | What it does | Re-run |
|---|---|---|---|
| `ingestion.name` | `zip` | frame source (the only one) | F |
| `ingestion.stride` | 1 | keep every n-th frame. Faster, but lowers the measured fps, and frame-counted tracker settings (`max_lost_frames`) then span more seconds | F |
| `ingestion.max_frames` | null | stop after n frames (smoke tests); null = all | F |
| `detector.weights` | `models/yolo11n.pt` | YOLO checkpoint; only the person class is kept | F |
| `detector.conf` | 0.22 | confidence a person box needs to reach the tracker; the only confidence filter before tracking. Lower = more people found, more false boxes. Verified not to drop any student (`docs/experiments/02_boxmot.md`, §3; `scripts/scan_detector.py`) | F |
| `detector.imgsz` | 640 | YOLO input size in pixels. Higher = far students found better, slower | F |
| `detector.half` | true | half precision on the GPU (ignored on CPU) | F |
| `scene.size` | [64, 36] | the frame is shrunk to this many pixels before the gate's scene features are computed | F |
| `scene.grid` | 3 | grid×grid brightness cells among the scene features | F |

**The tracker (`tracker.name = "botsort"`, `tracker.options`).** BoT-SORT matches each frame's
boxes to existing tracks in two passes, predicting where every track should be with a Kalman
filter. All options go straight to BoxMOT 26; an unknown key is an error, so BoT-SORT options left
in place break StrongSORT (replace the whole `tracker` dict to switch). All are **F**.

| Option | Value (BoxMOT default) | Mechanism | Higher means |
|---|---|---|---|
| `track_high_thresh` | 0.25 (0.63) | boxes above it take part in the first matching pass; only these can start new tracks | fewer boxes in the main pass |
| `track_low_thresh` | 0.21 (0.10) | boxes between it and `track_high_thresh` get a second chance: they may continue an existing track that found no match in the first pass (strict, IoU ≥ ~0.71), never start one. Only boxes that passed `detector.conf` reach the tracker, so with the tuned values the second chance covers 0.22–0.25 | more weak boxes rescued |
| `new_track_thresh` | 0.67 (0.62) | an unmatched first-pass box starts a new track only above this; below it, it must match again on the next frame or is discarded | fewer spurious ids, slower pick-up of real people |
| `match_thresh` | 0.80 (0.77) | the largest matching cost accepted in the first pass. The cost is 1 − IoU × confidence, so 0.80 means IoU × confidence ≥ 0.20 | more lenient matching |
| `max_lost_frames` | 17 (40) | how many frames a lost track is remembered and can be re-found under the same id (17 frames ≈ 3.4 s at 5 fps). Counted in processed frames | longer identity memory, more risk of attaching the wrong person |
| `use_cmc` | false (true) | camera-motion compensation: estimate a global image shift (with detections masked) and correct the predictions. Needs pixels; slower | — |
| `use_embeddings` | false (true) | run the ReID appearance model (`osnet-x0-25-msmt17`) and let appearance similarity also justify a match. Much slower | — |
| `proximity_thresh` | 0.48 (0.61) | with embeddings on: appearance may only be used for pairs with IoU above 1 − this | appearance used for more distant pairs |
| `appearance_thresh` | 0.55 (0.62) | with embeddings on: appearance distance above this rejects the match | more lenient appearance matching |

BoxMOT also accepts `det_thresh`, but neither BoT-SORT nor StrongSORT reads it, so it is not in the
config; `detector.conf` is the confidence filter.

StrongSORT's options, used by the `full_strongsort` tuning profile, work differently: `min_conf`
(BoxMOT default **0.6**) drops every detection below it before tracking, on top of `detector.conf`.
At 0.6 that loses whole students, so the profile and the benchmark variant set the quick-tuned
0.2153; `max_cos_dist` and `max_iou_dist` are the largest appearance and 1 − IoU distances accepted;
`max_age` is how many frames a confirmed track survives unmatched; `n_init` is how many consecutive
hits confirm a new track. StrongSORT always uses ReID.

### Feature pass: tracklets, window, aggregation, BFMU

| Key | Value | What it does | Re-run |
|---|---|---|---|
| `tracklets.min_visible_s` | 3.0 | drop track pieces visible for less than this many seconds in total | P |
| `tracklets.max_gap_s` | 1.0 | split a track where it is missing for longer than this (counted in frames that exist, so camera dropouts do not split tracks) | P |
| `window.length_s` | 8 | window length in seconds | P |
| `window.overlap` | 0.5 | overlap of consecutive windows (step = length × (1 − overlap) = 4 s) | P |
| `aggregation.min_coverage` | 0.5 | a track counts as visible in a window if it spans at least this share of it; decides `n_tracks`, Vec and the per-student rows | P |
| `aggregation.heatmap_grid` | 3 | grid×grid occupancy columns `occ_ij`: share of visible students whose centre is in each cell | P |
| `aggregation.thresholds` | `kg.stillness` 0.8, `scb.hand_raising` 0.1, `scb.read` 0.5, `scb.write` 0.5 | for each listed per-student feature, a Vec column `<key>.frac` = share of visible students strictly above the value. Window level only | P |
| `cfmu.name` | `null` | cluster features (none in Deliverable III) | P |

`bfmu` is a **list**, one entry per feature family. Each produces per-student keys, which become
`trk.<key>` per student and `vec.<key>.mean` / `.std` per window. The whole list is in the feature
hash; the entries that read pixels (`residual_flow`, `scb`) are also in the tracking hash.

| Entry and option | Value | What it does | Re-run |
|---|---|---|---|
| `kalman_geometry.still_threshold` | 0.05 | speed (box diagonals per second) below which a frame counts as still, for `kg.stillness` | P |
| `kalman_geometry.use_kalman` | true | take velocity from the tracker's Kalman filter where available; false = finite differences of box centres only | P |
| `residual_flow.grid` | 2 | grid×grid cells inside each box, one flow value each (`rf.mag_ij`) | F |
| `residual_flow.scale` | 0.5 | frames are resized by this before optical flow. Higher = finer, slower (flow is the costliest stage) | F |
| `scb.mode` | `soft` | `soft`: raw detector scores per student (mean and peak per window); `flags`: thresholded 0/1 per class | F |
| `scb.min_overlap` | 0.5 | a behaviour box is assigned to a student only if at least this share of it lies inside the student's box | F |
| `scb.models[].weights` | SCB YOLOv7 checkpoint | the behaviour detector | F |
| `scb.models[].classes` | hand_raising, read, write | must equal the checkpoint's classes in order; names the `scb.*` keys | F |
| `scb.models[].conf` | 0.25 | per-class threshold, **flags mode only**; no effect in soft mode | F |

### Models

| Key | Value | What it does | Re-run |
|---|---|---|---|
| `gate.name`, `gate.params` | `random_forest`, 300 trees, `min_samples_leaf` 5 | the context gate. Names: `random_forest`, `svm`, `logreg`, `knn`; params go to the scikit-learn constructor | T |
| `gate_uses_vec` | false | also give the gate Vec, not only scene features | T |
| `expert.name`, `expert.params` | `logreg`, `C` 0.1 | the engagement expert. Names: `logreg`, `random_forest`, `hand_weighted`. `C` is inverse regularisation: lower = simpler model | T |
| `hand_weighted.weights` | stillness 1.0, write 1.0, read 0.5, hand raising 0.5 | the label-free baseline: a weighted sum of standardised features, reported as `hand_weighted` in every run | T |
| `hand_weighted.phone_key`, `.phone_sign` | `vec.scb.using_phone.mean`, signs per activity | flip the phone feature's sign per activity. **No effect now**: the SCB model has no phone class | T |
| `mixture.mode` | `soft` | `soft` weights all experts by the gate; `hard` uses the top activity's expert | T |
| `mixture.expert_scope` | `per_experiment` | six experts, one per activity; `shared` = one for all | T |
| `mixture.tau` | 0.5 | engaged cut-off in **early** aggregation (unused in late) | T |
| `engagement.aggregate` | `late` | `late`: score each student, average per window; `early`: score Vec (§7.5) | T |
| `engagement.late_context` | true | late only: append the window's Vec to every student row | T |
| `engagement.labelers` | `consensus` | `consensus`: train on the median-of-labelers label; `ensemble`: one model per labeler on their own labels, averaged over the labelers who rated each student (§7.5); late only | T |
| `engagement.ensemble_exclude` | `["self_labeling"]` | labelers left out of the ensemble (the self-ratings by default) | T |
| `context_mode` | `classified` | **unused**: training always reports both, inference always uses classified | T |

### Augmentation (training copies; Chapter 8, §8.4)

Not in the tracking or feature hash: the originals' caches never change. More or different copies
need `extract --reuse` (a frame pass per new copy), then `train_models`.

| Key | Value | What it does |
|---|---|---|
| `augmentation.copies` | 0 | augmented copies per session for training; 0 = off. Each copy costs a frame pass |
| `augmentation.seed` | 0 | seed of the per-(session, copy) parameter draw |
| `augmentation.brightness` | 35 | brightness offset drawn from ±35 grey levels |
| `augmentation.contrast` | 0.15 | contrast factor from 1 ± 0.15 |
| `augmentation.saturation` | 0.2 | saturation factor from 1 ± 0.2 |
| `augmentation.hue_deg` | 3 | hue shift from ±3°; kept tiny on purpose |
| `augmentation.gamma` | 0.15 | gamma from 1/1.15 to 1.15 (log-uniform) |
| `augmentation.noise.kinds`, `.max` | gaussian, speckle; 0.35 | noise kind drawn per copy; level from 0 to 0.35 (Gaussian σ up to ≈ 17 grey levels, speckle up to ±21%), redrawn per frame |
| `augmentation.hflip` | 0.5 | probability that a copy is mirrored (student boxes mirrored with it) |
| `augmentation.mask` | grey cutout, p 0.5, 1–3 masks, 4–12% of the frame side | fixed occluders per copy, with probability `p` |

### Labels

All **T**, so label experiments are cheap.

| Key | Value | What it does |
|---|---|---|
| `labels.tau_att` | 4 | a student is engaged if their consensus attention (median over labelers of each one's time-weighted mean, 1–5) is at least this |
| `labels.tau_share` | 0.5 | a window is engaged if the share of engaged labelled students is at least this; also the late-mode cut-off |
| `labels.min_labeled_students` | 3 | windows with fewer labelled students get no window label (still used for `retention_r`); per-student labels ignore it |
| `labels.exclude_self` | false | leave each student's self-rating out of the consensus |
| `labels.offset_s` | 0.0 | seconds added to every label timestamp, to align labels with the camera clock |
| `labels.students` | `camera` | `camera`: only students annotated as visible in this camera; any other value (e.g. `all`): every student of the session |

At the design's values (4 and 0.5) almost no window is engaged; the provisional values used so far
are `tau_att=3`, `tau_share=0.6` (Chapter 11).

### Tuning and benchmarking (excluded from every hash)

The search-space syntax is in Chapter 8, §8.5: a list is a choice, `{low, high}` a uniform range,
`"int": true` rounds, `"log": true` samples evenly in log space. Trial 0 is always the current config.

| Key | What it does |
|---|---|
| `tuning.seed` | seed of the random searches (separate from `seed`); the bisection search is deterministic and does not use it |
| `tuning.tracker.det_conf_floor` | detections are cached once at this confidence; each trial filters them at its own `det_conf` |
| `tuning.tracker.objective` | the tracking metric maximised (`idf1`). Only higher-is-better metrics make sense |
| `tuning.tracker.profiles.<name>` | `tracker`, `sessions` (`"all"` or a list), `max_frames`, `n_trials` (the trial budget of either search; `--trials` overrides it), `fixed` options pinned for every trial, and optionally `base_options`, trial 0's settings (otherwise `tracker.options` for the configured tracker, BoxMOT defaults for another). The bisection search starts from the profile's random-search winner when one exists, or from another profile's winner named by `start_from` (`full_reid_cmc` starts from `full`) (§8.6) |
| `tuning.tracker.space.<tracker>` | ranges per option. `det_conf` here stands in for `detector.conf`: the winning value is patched there |
| `tuning.gate` / `tuning.expert` | `n_trials` per search, and `space.<model name>`; only the space matching `gate.name` / `expert.name` is used |
| `benchmark.sessions` | sessions timed by `scripts/benchmark.py` |
| `benchmark.variants.<name>` | overrides per variant. A dotted key (`"tracker.options.use_cmc": true`) changes one value; a plain key with a dict (`"tracker": {...}`) replaces the whole section. Results resume by variant name, so rename a variant after editing it |

A tracker tuning log refuses to resume when anything in its fingerprint changed, which includes
the whole `detector` section: applying a tuned `detector.conf` means the next run of that profile
needs `--fresh`.

## 9.7 Settings that currently do nothing, or surprise

Collected here so they are not mistaken for working knobs:

- `context_mode`: unused.
- `hand_weighted.phone_key` / `phone_sign`: the column does not exist with the current SCB model.
- `proximity_thresh` and `appearance_thresh` while `use_embeddings` is false, and
  `scb.models[].conf` in soft mode: inactive, but in the tracking hash, so changing them forces a
  needless frame pass.
- `mixture.tau`: unused while `engagement.aggregate` is `late`.
- StrongSORT's `min_conf` defaults to 0.6 when not set, which drops whole students (see
  `detector.conf` above); anything that runs StrongSORT should set it.
- `--set` cannot reach list elements or keys that contain dots (§9.5).

# Chapter 8 — Scripts, caches and tuning

The `src/` package is a library: it has no `main`. Everything you run lives in `scripts/`. This
chapter covers each entry point, what it caches where, and how the tuning machinery works.

## 8.1 The workflow at a glance

```
                      ┌─────────────────────────────┐
                      │ scripts.data_visualization.* │  look at the data: labels, SCB, video
                      └─────────────────────────────┘
┌──────────────────┐   ┌───────────────────────────┐   ┌────────────────────┐
│ tune_tracker      │──►│ run_pipeline extract       │──►│ train_models        │──► results + models
│ (detector+tracker)│   │ (both passes, 52 sessions) │   │ (LOGO evaluation)   │
└──────────────────┘   └───────────────────────────┘   └────────────────────┘
                                                          ▲        │
                                                          │        ▼
                                               ┌────────────────┐ ┌──────────────────────┐
                                               │ tune_models     │ │ run_pipeline infer    │──► retention graph
                                               │ (gate, experts) │ │ (one ZIP + models)    │
                                               └────────────────┘ └──────────────────────┘

 checking:  scan_detector (detector threshold vs manual boxes)   plot_results (held-out curves)
            output_viewer (a session with tracks, per-student scores and labels)
            label_baselines (constant predictions vs the published label targets, benchmarks.md §1)
            label_agreement (labeler agreement and the human ceiling, experiment 06 Part 1)
            compare_results (two results files side by side, with confusion matrices)
```

One-off analyses behind specific doc tables are kept, frozen, in `scripts/analysis/` (see its
README); they are not maintained like the scripts here.

Every script takes `--config` (default `configs/deliverable3.json`) and `--set key=value`
overrides (Chapter 4).

## 8.2 `run_pipeline` (`scripts/run_pipeline.py`)

### `extract`

```
python -m scripts.run_pipeline extract [--session group_01/experiment_03/context_cam_02] [--reuse]
```

For each in-scope session (or the one given), it runs both passes and writes:

```
outputs/cache/tracks/<tracking hash>/<session>.npz        the SessionTrack (frame pass)
outputs/cache/features/<feature hash>/<session>.csv       window rows
outputs/cache/features/<feature hash>/<session>.manifest.json timing, stage seconds, hashes
outputs/cache/features/<feature hash>/<session>.tracks.csv per-student rows (late aggregation)
```

**By default nothing is read back.** Every session runs the full frame pass, so each manifest's
`stage_seconds` are real runtime numbers. With `--reuse` (for development):

- sessions whose window CSV and per-student CSV both exist are skipped;
- otherwise, a cached track for the current tracking hash is loaded instead of re-running the
  frame pass, and the manifest records `"frame_pass_cached": true`.

So after changing the window length, `extract --reuse` recomputes all features in about a minute.
After changing the tracker, the tracking hash changes, no cached track matches, and the frame pass
runs again.

### `infer`

```
python -m scripts.run_pipeline infer --zip <camera.zip> --models outputs/models/<hash>.joblib --out <dir>
```

Runs both passes on one ZIP with no caching, loads the model bundle, runs the gate and mixture,
and writes the retention graph JSON, PNG and manifest. It warns if the bundle was trained under a
different config hash.

## 8.3 The track store (`src/core/store.py`)

`save_track` writes a `SessionTrack` as one compressed `.npz`: all tracklets' arrays concatenated
(one row per observation, with a `tid` column), one array per signal, the per-frame scene matrix,
and a JSON header for everything else. `load_track` rebuilds identical `Tracklet` objects. The
round trip is tested (`test_track_store_round_trip`).

Two details matter:

- **Atomic writes.** It writes to a temporary file, then renames it. A crash halfway through
  never leaves a half-written cache file that a later `--reuse` would trust.
- **Read each array once.** Every `z["name"]` access on a compressed `.npz` decompresses the
  whole array again. Indexing `z["xyxy"][a:b]` inside a loop decompressed the full array per
  iteration and kept every copy alive, which once exhausted 16 GB of RAM. Load everything into a
  dict first (`a = {k: z[k] for k in z.files}`), then slice.

## 8.4 `train_models` (`scripts/train_models.py`)

```
python -m scripts.train_models [--set labels.tau_att=3]
```

**Per-labeler mode (the default, `engagement.labelers = "per_labeler"`; Chapter 7 §7.9; also
`"pooled"`, one model trained on every labeler's rows that answers for all of them):**

1. **Load** the window table (gate features, as below) and every session's **rating entries**
   (`label_entries`): every entry of every labeler of the camera's students, the shared
   session-start entry excluded, with its training value (`labels.train_target`).
2. **Label-centred windows**: one window of `window.length_s` per entry time;
   `Pipeline.window_features` computes their features from the cached `.npz`, cached as
   `<session>.labelwin.csv` next to the window CSVs, with `<session>.labelwin.centres.npy` listing
   every centre the cache covers (windows where nobody is tracked have no rows, so the rows alone
   cannot tell a complete cache from a stale one). Both are written to a temporary file and renamed,
   so parallel runs never read a half-written cache; the cache is rebuilt if an entry time is not in
   the centre list. Each entry joins its student's linked track(s) in its window (`join_entries`).
3. **Leave-one-group-out**: per held-out group, the gate on the window grid and one model per
   labeler (`PerLabelerModel`) for the configured and the shared scope; every held-out entry scored
   per label (`rating_metrics`) in two tracks (experts, self) for three predictions (own labeler,
   median, majority) and two baselines (constant 3, training mean). The same fold model is also
   scored on its own training entries (originals only, own labeler, classified context:
   `<track>/<variant>/train/own_labeler`), so held-out scores can be read against training ones
   (underfitting against overfitting). Every score carries the 5 x 5 rating confusion matrix and the
   engaged / not-engaged one at `tau_att`.
4. **Write** the results JSON (`"mode": "per_labeler"`, an `evaluation` block keyed
   `<track>/<variant>/<context>/<prediction>`), and the bundle with `"kind": "per_labeler"`, the
   gate, the per-labeler model, `gate_cols`, `row_cols` and `tau_att`.

Augmented copies (§8.4 below) join every training side and are never scored, as in the window-grid
modes.

**Window-grid modes** (`consensus`, `ensemble`):

1. **Load** every feature CSV for the current feature hash; attach group, experiment and window
   labels (`label_session`).
2. **Build the expert rows** (`engagement_rows`, §7.5): `vec.*` per window (early), or `trk.*`
   per student plus `vec.*` if `use_class_context` (late; read from `<session>.tracks.csv`, with each
   track linked to its student by `associate_sids` on the cached `.npz` to attach per-student
   labels). The gate gets `scene.*` (plus `vec.*` if `gate_uses_vec`).
3. **Leave-one-group-out**: for each held-out group, fit the gate and every mixture variant on
   the other two groups and score on the held-out one (Chapter 7).
4. **Write** `outputs/results/<config hash>.json`: the full config, data summary (windows, label
   coverage, class balance), every fold's metrics, and mean ± std summaries.
5. **Fit final models on all groups** and save `outputs/models/<config hash>.joblib`, the bundle
   `infer` loads. It contains the gate, the mixture, `gate_cols`, `row_cols`, the aggregation mode
   (`aggregate`), `fill` (score for windows with nobody visible) and `tau` (the engaged cut-off).

### Seeing the curves: `plot_results` (`scripts/reporting/plot_results.py`)

```
python -m scripts.reporting.plot_results [--set ...]        # same --set values as train_models
    [--sessions experiment_03/context_cam_01 ...] [--out DIR] [--class-curve] [--final]
```

`train_models` reports averages. `plot_results` draws out-of-sample curves per session.
`--sessions` keeps only the sessions whose key contains one of the given substrings (the models are
still fit on every training group), and `--out` picks the folder.

**Per-labeler mode** (1–5 scale). Each panel follows one student: the one with the most
self-reports in the session, ties broken by the larger variance of those self-reports, among the
students with a linked track. A class curve (the mean over the window's students) is too smooth to
show how the predictions move, and a student with many, varied self-reports shows the most. Each
labeler model's curve is drawn for that student alone (mean over the student's tracks per window;
gaps where it is not tracked), against only that student's entries. The self-report is the target
that matters most, so its model is a thick orange line drawn on top, with self entries as orange
dots. Each expert labeler's model is a thin line in its own light colour (labeler_01 blue, _02 green,
_03 purple, _04 cyan, _05 olive), with that labeler's entries as crosses of the same colour.
`tau_att` is drawn dashed. The title names the student and gives the MAE of each labeler's model
against that labeler's own entries (experts pooled, and self). `heldout_curves` in `train_models`
computes the curves. `--class-curve` plots the class curves and every entry instead; only then does
`--final` add the final curve (median or majority, `output.prediction`), which is off by default
because, with each curve at its own labeler's level, the median matches no one's entries.

**Window-grid modes** draw the predicted engagement score against the labelled share of engaged
students, with the engaged cut-off, and shade the background with the gate's predicted activity. Every curve is out-of-sample: it repeats the
leave-one-group-out fits (seconds) and plots each group from models that never saw it. The
title gives the true activity, the gate's hit rate and the correlation r with the labelled
share. Output: `<output_dir>/plots/<config hash>/`, one PNG per session plus `overview.png`.

### Comparing two runs: `compare_results` (`scripts/reporting/compare_results.py`)

```
python -m scripts.reporting.compare_results outputs/results/<A>.json outputs/results/<B>.json --names A B [--plot]
python -m scripts.reporting.compare_results c5341cb6385e 5df1987449dd     # config-hash prefixes work too
python -m scripts.reporting.compare_results --latest                      # newest MoE vs newest single decoder
```

For two per-labeler results files: both runs' summary metrics on the configured variant (AUC,
Spearman, balanced accuracy, macro F1, MAE, RMSE), with the constant-3 and training-mean baselines,
and the same on the training entries; AUC per held-out group with the difference; AUC and
Spearman per activity (each activity scored in the fold that holds it out; per-labeler results
carry a `per_experiment` breakdown in every fold). `--context oracle` and `--prediction
median|majority` pick other rows. Then the confusion matrices summed over the held-out folds (counts,
row shares, per-rating precision and recall, and the binary matrix); `--plot` saves them as heatmaps
to `outputs/plots/compare_<A>_<B>.png`. Made for `docs/experiments/06_full_pipeline_and_labels.md`.

### Label agreement: `label_agreement` (`scripts/labels/label_agreement.py`)

```
python -m scripts.labels.label_agreement [stats|alpha|ceiling|coarse|all] [--model-entries]
```

How much the labelers agree, and how well one labeler is predicted by the others, scored exactly like
the model (experiment 06 Part 1). `stats`: entries, rating shares, timing and co-occurrence per
labeler; `alpha`: Krippendorff's ordinal alpha over unit x interpolation x labeler set x scale;
`ceiling`: every entry predicted from the other labelers (each interpolation, lags, a best-match
upper bound, a learned combination) and from the labeler's own other ratings, with `--model-entries`
also on exactly the entries the pipeline is scored on; `coarse`: agreement on student x activity and
student x k-minute means. Writes `outputs/labels/agreement.json` (about 9 minutes for `all`). Run it
on any new dataset before training: it says what score is achievable at all.

### Watching a session: `output_viewer` (`scripts/data_visualization/output_viewer.py`)

```
python -m scripts.data_visualization.output_viewer group_01 experiment_03 context_cam_01 [--set ...] [--models <bundle>]
```

Plays a session with what the pipeline made of it, from cached results only (the frame-pass `.npz`,
the feature CSVs, and a model bundle), so it runs no network and scrubbing is instant:

- every track drawn on the video, coloured red → green by that student's predicted engagement in
  the current window; white if tracked but not scored in this window, thin grey if dropped by the
  tracklet filter;
- with labels on (`l`), each track's linked student and that student's own label, so per-student
  errors are visible: a red box labelled `L1` is a student the model thinks is disengaged and the
  labelers do not;
- a header with the gate's activity against the true one, the window score against the cut-off,
  and the labelled share;
- a timeline of the score curve and the labelled share, with the predicted and true activity as
  colour strips; click to seek.

With a per-labeler bundle, ratings are drawn on the same 0–1 axis as (rating − 1) / 4: each box
shows the student's final rating (`r2.6`: median or majority across the labeler models) and, with
labels on, their labelled rating (`L3.0`: the median over their labelers of each one's window
mean); the red curve is the final class rating, the grey one the labelled class rating, the dashed
line `tau_att`.

It uses the bundle trained under the same config (or `--models`), and the same `--set` values the
session was extracted with. It is a development tool: per-student scores and identities appear
only here and in training, never in the pipeline's outputs. `--snapshot out.png --start 30`
renders one frame without opening a window.

### Data augmentation: augmented training copies (`src/ingestion/augment.py`)

With few groups, the models see each room, camera and lighting once. Augmentation adds training
copies of every session seen through a different, plausible camera: shifted colour, brightness and
contrast, sensor noise, a mirrored room, fixed grey occluders. It is off by default
(`augmentation.copies: 0`).

**One transformation per session copy.** `sample_augmentation(aug_cfg, session, copy)` draws one fixed
parameter set per (session, copy), deterministically from `augmentation.seed`, and the same set is
applied to every frame of that copy. Tracking and the window features therefore see a consistent
video (another white balance, a mirrored room, an occluder that stays put), never per-frame
flicker. Only the sensor noise is redrawn each frame, at a level fixed for the copy, the way real
sensor noise behaves.

```python
params = sample_augmentation(cfg["augmentation"], "group_01/experiment_03/context_cam_02", copy=1)
src = AugmentedZipFrameSource(zip_path, params)   # a ZipFrameSource and a torch Dataset
len(src), src[120]                                # same timestamps as the original; frame 120, augmented
```

`AugmentedZipFrameSource` subclasses `ZipFrameSource` (same timestamps, iteration and `read`) and
`torch.utils.data.Dataset` (`len`, indexing).

**Copies are never written to disk.** A copy is the original camera ZIP read through
`AugmentedZipFrameSource`, augmenting each frame as it is decoded. The parameters are seeded by
(`augmentation.seed`, session, copy) and the noise by frame index, so a copy is the same video every
time it is read (`test_a_copy_is_the_same_video_every_time_it_is_read`): seeding, not storage, makes
runs comparable. The only cost is CPU: augmenting adds roughly 0.1–0.2 s per frame to whatever reads
the copy. (An earlier design wrote copies to disk as complete datasets; it was removed on 15
November 2026 as not worth about 40 GB per copy.)

**Transforms, in order:** horizontal flip → colour (brightness, contrast, saturation, hue, gamma) →
noise (Gaussian or speckle) → masks (grey cutouts at fixed positions). Ranges are in the config
(§9.6). The hue range is deliberately tiny (±3°): a larger hue shift turns skin and clothing
colours unnatural, while brightness, contrast and noise reflect what real classroom cameras do.

**Labels.** Colour, noise and masks leave the manual student boxes unchanged. A flip mirrors them
(`hflip_boxes`): before linking a flipped copy's tracks to students, `train_models` mirrors the
boxes with the frame width recorded in the copy's manifest. Time is unchanged, so window labels
and per-student labels are the original session's.

**Extraction.** With `copies: N`, `run_pipeline extract` first does every original session as usual,
then every session's copies 1…N, augmented on the fly. Their features go next to the
originals under their own names: `<session>.aug<k>-<settings hash>.{npz,csv,tracks.csv,manifest.json}`. The originals' caches are
untouched (augmentation is not part of the tracking or feature hash). The settings hash in the name
means a change to the ranges never reuses stale copies, while raising `copies` keeps the existing
ones. Each copy costs a full frame pass, so `copies: 2` triples extraction time; `--reuse` resumes as
usual. The manifest of a copy records its exact parameters.

**Training.** `train_models` (and `plot_results`) append the copies to the window table
(`with_augmented`), marked `aug`. `heldout_splits` keeps leave-one-group-out over the original
sessions: copies of the training groups join the training side of each fold, and **no augmented
row is ever scored**. The held-out group is always evaluated on footage as recorded. The final
models are trained on originals and copies. Per-labeler results report
`augmented_rows_for_training`; the window-grid modes report `augmentation_copies`, `augmented_windows_for_training` and `augmented_row_label_coverage` (the
share of copy rows linked to a labelled student; it should be close to the originals', which
confirms the mirrored boxes line up). `tune_models` does not use copies.

**Looking at it.** `python -m scripts.data_visualization.augment_samples` writes, for three sessions, the original
frame, the first copies this config produces, and two *extremes* (every range at its bright or
dark end at once, maximum noise, maximum masks, flipped) to
`outputs/augmentation/<settings hash>/`, as PNGs, a contact sheet and the exact parameters.
`python -m scripts.data_visualization.augment_viewer` explores the same transforms interactively.

Verified on six 100-frame sessions with one copy each: extraction wrote 6 originals and 6 copies,
training added 52 augmented windows to the training folds and scored only the 52 originals, and
94% of copy rows were linked to labelled students (originals: 93%).

## 8.5 How tuning works (`src/tuning/search.py`)

### Search spaces

Spaces live in the config's `tuning` section, as JSON:

```json
"det_conf":        {"low": 0.1, "high": 0.5},                           uniform in a range
"max_lost_frames": {"low": 10, "high": 150, "int": true, "log": true},  log-uniform, rounded
"use_embeddings":  [true, false]                                         pick one
```

`log: true` samples uniformly in log space: 10–30 is as likely as 50–150. Use it for parameters
whose effect is multiplicative (counts, regularisation strengths).

### Random search

```python
trials = random_search(space, objective, n_trials, seed, baseline=current_params)
```

- **Trial 0 evaluates the baseline** (the current config), so every search reports its gain.
- Every other trial draws each parameter independently. Trials do not learn from each other;
  there is no model of the objective. With up to 10 parameters and 20–30 trials this is a
  reasonable choice: usually only a few parameters matter, and random sampling tries many distinct
  values of each, where a grid would repeat the same few.
- The generator is seeded, so re-running a search draws exactly the same parameter sets.
- Results come back sorted, best first. Several trials within about 0.01 of the best should be
  treated as ties.

### Beam bisection (the default for `tune_tracker`)

```python
trials = beam_bisection_search(space, objective, n_trials, baseline, beam=2, inactive={...})
```

Random search spends trials evenly over the whole space and does not learn from them. Once a good
region is known, a directed search is cheaper. `beam_bisection_search` starts from a good point
(usually the best trial of an earlier search) and improves it one parameter at a time.

**It never has to guess which way to go.** For every parameter it tries both a lower and a higher
value, and keeps whichever of the three (lower, current, higher) scores best. The only thing that
changes from step to step is *how far* it looks: each step halves the distance.

#### One parameter, one step

Each parameter has a current value `u` and an interval `[lo, hi]` it is searched in. At the start
the interval is the parameter's **whole range from the config's search space** (for example
`track_high_thresh` 0.05–0.70), wherever the starting value sits in it. One step:

```
        lo ─────────── u ─────────────────── hi        current value u, interval [lo, hi]
              ▲                  ▲
          (lo + u)/2         (u + hi)/2                 two probes: halfway to each end

  lower probe wins   ->  new u = lower probe,  new interval [lo, u]      (the old u is the new top)
  upper probe wins   ->  new u = upper probe,  new interval [u, hi]      (the old u is the new bottom)
  u itself wins      ->  u stays,              new interval u ± (hi − lo)/4, cut at lo and hi
```

Whichever wins, the interval is at most half as wide afterwards, so the next step's probes land closer to
the current best: big moves first, then finer and finer adjustments. A "stay" is not wasted; it
means the best value is near `u`, and the next probes look closer to it. Parameters marked
`"log": true` (`max_lost_frames`) do the same arithmetic on a log scale, so the halfway points are
geometric means (√(10 × 96) ≈ 31).

Two trials per parameter per step, and each probe changes **only that one parameter**; every other
parameter stays at the node's current value.

#### All parameters: one level

A **level** does that step for every numeric parameter of a node (2 probes each), then adds one
more trial, the **combined move**: every parameter set to its own winner at once. Parameters are
judged one at a time, so their winning moves may or may not add up; the combined trial checks
whether they do.

Each parameter's winner (the node with just that one move, and that parameter's halved interval)
and the combined move become candidates for the next level (a **node** is a point plus its
intervals). Only the best `beam` of them (2) are expanded at the next level; the rest are dropped.
That keeps the cost of a level at most (2k + 1) × beam trials for k numeric parameters, instead of
growing with every level.

The search stops when the trial budget is spent, or when every interval is narrower than 1/64 of its
range. Categorical parameters (ReID on/off) and parameters with no effect under the profile's fixed
settings (`inactive`: BoT-SORT's appearance thresholds when ReID is off) are never searched; they
keep their starting values.

#### Worked example: T3 (experiment 03)

BoT-SORT + ReID + CMC re-tuned for the pose detector, starting from T2's winner, 20 trials, 7
numeric parameters (`outputs/tuning/tracker_botsort_full_reid_cmc_aug1-4f270e71_pose_bisect.csv`).
Trial 0 is the start: IDF1 **0.8734**. Level 1 probes each parameter both ways (trials 1–14, in
alphabetical order):

| Parameter | Range | Start `u` | Lower probe (IDF1) | Upper probe (IDF1) | Winner | Interval after |
|---|---|---|---|---|---|---|
| `appearance_thresh` | 0.20–0.80 | 0.286 | 0.243 (**0.8752**) | 0.543 (0.8651) | lower | 0.20–0.286 |
| `match_thresh` | 0.50–0.95 | 0.711 | 0.606 (**0.8740**) | 0.831 (0.8675) | lower | 0.50–0.711 |
| `max_lost_frames` (log) | 10–150 | 96 | 31 (**0.8739**) | 120 (0.8732) | lower | 10–96 |
| `new_track_thresh` | 0.05–0.70 | 0.548 | 0.299 (0.8591) | 0.624 (**0.8840**) | upper | 0.548–0.70 |
| `proximity_thresh` | 0.30–0.80 | 0.773 | 0.536 (0.8623) | 0.786 (0.8725) | stay (0.8734) | 0.648–0.80 (± 0.125, cut at the range end) |
| `track_high_thresh` | 0.05–0.70 | 0.285 | 0.168 (0.8603) | 0.493 (**0.8932**) | upper | 0.285–0.70 |
| `track_low_thresh` | 0.05–0.30 | 0.175 | 0.112 (0.8702) | 0.237 (**0.8767**) | upper | 0.175–0.30 |

The combined move (trial 15) takes all six winning moves at once: IDF1 **0.8999**, better than any
single move (the best, `track_high_thresh` alone, gave 0.8932), so here the moves did add up. The two
best nodes are trial 15 and trial 12, and level 2 expands trial 15 first: `appearance_thresh` probes
0.221 and 0.264 inside 0.20–0.286 (trials 16, 17), `match_thresh` probes 0.553 and 0.658 inside
0.50–0.711 (trials 18, 19). Trial 19 (`match_thresh` 0.658, IDF1 **0.9007**) is the winner. The
20-trial budget ran out there, partway through level 2; trial 12's node was never expanded.

Several level-1 winners beat the start by less than 0.001 (`match_thresh`, `max_lost_frames`), well
inside the noise between runs. The search takes them anyway, because it compares scores, not
significance; read small wins as ties.

#### What it is and is not

- A **local** search: it improves the starting point, one parameter at a time, from the start's
  neighbourhood outwards. It can step across the whole range early (the first probes are halfway
  to each end), but it does not explore the space the way random search does. Start it from a good
  point.
- **Greedy per parameter:** each parameter is judged with the others held fixed, so interactions are
  only seen through the combined move.
- **Deterministic:** the same start and space give the same trials, so a run can be resumed and
  no trial is computed twice (it shares the random search's per-session result log).

On a smooth test function it reaches the optimum within about 60 trials where random search is
still far off (`tests/test_tuning.py`); real tracking scores are noisier.

### Nested leave-one-group-out (for the gate and experts)

If you pick hyperparameters by their score on the test group, that score is optimistic: you have
selected on it. `nested_logo_search` avoids this:

```
for each outer held-out group G:
    search hyperparameters using only the other two groups (inner LOGO, 2 folds)
    refit the winner on those two groups; score ONCE on G      -> unbiased estimate
finally: search on all groups                                   -> the params to put in config
```

It returns both the selected parameters and the nested estimate. Report the nested number.

## 8.6 `tune_tracker` (`scripts/tuning/tune_tracker.py`)

```
python -m scripts.tuning.tune_tracker --profile <profile> [--trials N] [--fresh]       # beam bisection (default)
python -m scripts.tuning.tune_tracker --profile <profile> --randomize                   # seeded random search
python -m scripts.tuning.tune_tracker --profile <profile> --augment [COPY]              # on augmented data
python -m scripts.tuning.tune_tracker --profile <profile> --check
```

**Search method.** The default is the beam bisection search (§8.5), starting from the best trial of
the same profile's random search on the original data (`tracker_<tracker>_<profile>.json`) when it
exists, or of the profile named by its `start_from` (`full_reid_cmc` starts from vanilla BoT-SORT's
`full`), or from `--start <scan>.json`, else from the profile's baseline. `start_from` does not
affect scores and is not part of the resume fingerprint. `--beam` sets the beam
width (2). `--randomize` runs the seeded random search, the method of the first full-dataset
experiments; resume those with `--randomize`. Both methods read and extend the same
`<stem>.sessions.jsonl`, so a trial computed by one is never recomputed by the other. Bisection
results are written as `<stem>_bisect.{csv,json}`; random-search results keep the plain names.

**Augmented data.** `--augment [COPY]` runs the same profile on augmented copy COPY (default 1),
augmented on the fly: the floor detections, ReID embeddings and CMC frames all come from the
augmented frames. Boxes are mirrored where a copy is flipped, the detections and ReID embeddings get
their own cache (keyed by the copy's tag), and results go to their own files
(`<stem>_<aug tag>...`). The original data's fingerprints and caches are unchanged.

**The objective.** Tracking is scored against our manual boxes (`abi_labels`) with
`tracking_metrics` (`src/evaluation/tracking.py`):

| Metric | Meaning |
|---|---|
| `idf1` | identity F1: how much of each student's time is covered by *one* consistent track; the objective |
| `coverage` | share of (student, frame) pairs matched by some track |
| `id_switches_per_subject` | how often a student's track id changes |
| `tids_per_subject` | how many different track ids each student gets (fragmentation) |
| `subjects_found` | share of annotated students matched at all |

IDF1 uses a one-to-one assignment between tracks and students, so if a student's time is split
across two track ids, only the larger part counts. Scores are averaged over sessions and also
reported per group, so a configuration that only helps one room is visible.

### What is computed once, and what each trial re-runs

Most of the work in tracking a session does not depend on tracker settings, so it is cached:

1. **Detections.** Each session's detections are computed once at a low confidence floor (0.05)
   and stored in `outputs/cache/detections/`. Each trial filters them at its own `det_conf`, so
   detector confidence is tuned alongside the tracker at no extra cost.
2. **ReID embeddings.** An appearance embedding depends only on the frame and the box, never on
   tracker settings. If any trial can use ReID, every floor-level detection is embedded once with
   the trackers' default ReID model (`osnet-x0-25-msmt17`, shared by BoT-SORT and StrongSORT) and
   stored next to the detections (`<session>.reid-osnet-x0-25-msmt17.npz`, float32, about 55 MB
   per session on average, 25–85 MB). The embeddings are computed through BoxMOT's own embedding code
   (`BoxmotTracker.embed`), and trials hand them to the tracker
   (`BoxmotTracker.update_with_embeddings`), which then runs no ReID model.
3. **Frames are only decoded when a trial needs pixels**, which, with embeddings cached, means only
   when it turns camera-motion compensation (CMC) on. CMC estimates camera motion from consecutive
   frames while masking that trial's own detections, so it cannot be cached. Otherwise the tracker
   gets a blank frame of the right size. Frames are looked up by timestamp, so a frame that failed
   to decode cannot shift detections onto the wrong frame.

**Trust, but verify.** `--check` runs the profile's trial-0 settings on its first session twice:
once through the fast path (cached embeddings, blank frames where possible) and once through the
tracker's own path (decoded frames, its own ReID). It exits nonzero unless the tracks are
identical, so it can gate a chained run. Checks run so far: blank frames for BoT-SORT without ReID
(10 vs 10 tracks), cached embeddings with decoded frames for StrongSORT (42 vs 42) and BoT-SORT
with ReID and CMC (16 vs 16).

### Profiles (`tuning.tracker.profiles`)

| Profile | Tracker | Sessions | Trials | Fixed |
|---|---|---|---|---|
| `quick` | configured (BoT-SORT) | 3, first 300 frames each | 25 | none: ReID and CMC are searched |
| `full` | BoT-SORT | all 52, all frames | 20 | ReID off, CMC on |
| `full_strongsort` | StrongSORT | all 52 | 20 | CMC on |
| `full_reid_cmc` | BoT-SORT | all 52 | 20 | ReID on, CMC on |

A profile may name its `tracker`; its `fixed` parameters apply to every trial and are removed from
the search space. Trial 0 runs the profile's `base_options` if it has them, otherwise
`tracker.options` for the configured tracker and BoxMOT's defaults for another one.
`full_strongsort` sets `base_options` to the quick-tuned StrongSORT values, so both full-dataset
trackers start from their best known settings (BoxMOT's StrongSORT default `min_conf` of 0.6 would
drop whole students; `docs/experiments/02_detection.md`, §3).

**Is the detector the limit?** `python -m scripts.tuning.scan_detector` checks, without any tracker, what
detector confidence thresholds cost, from the cached floor detections. A student's recall is the
share of frames in which they are detected; the 0.05 floor is the reference.

- By default it **binary-searches the optimal threshold**: the highest one that keeps every
  student (none undetected, none below half their frames, none losing more than `--max-drop`, 0.10,
  against the floor). Recall only falls as the threshold rises, so bisection is exact and needs
  about 8 passes. Each probe also reports `unmatched_share`, the kept boxes that match no annotated
  student (teachers, TAs, false positives): the price of a lower threshold. Output:
  `outputs/tuning/detector_search.{csv,json}`, including whether the configured `detector.conf`
  keeps everyone.
- `--grid` prints the full table over `--thresholds` instead (`detector_scan.csv`). At the
  configured 0.22 no student is dropped on the original data.
- `--augment` augments every frame on the fly, 5 variants per frame by default (`--variants MIN
  MAX`; `--stride N`, `--workers`), and detects each variant at the floor; results go to
  `detector_search_augmented.*`. `--detect` runs the detector where the original data's floor
  detections are not cached yet (GPU). The full-dataset profiles keep CMC **on** everywhere: every trial then decodes
real frames, so no full-dataset result depends on the blank-frame shortcut, and the three
experiments differ only in what they compare (ReID on/off with the same tracker; StrongSORT vs
BoT-SORT with ReID). This costs time (up to about an hour per trial) in exchange for reliability.

### Resuming long runs

Full-dataset runs take a day or more, so they are resumable at the level of one session:

- every (trial settings, session) result is appended to
  `outputs/tuning/tracker_<name>_<profile>[_<aug tag>].sessions.jsonl` the moment it is computed;
- both search methods are deterministic: re-running the same command proposes the same trials and
  reuses every result already on disk, so an interruption (Ctrl+C, a crash, a reboot) loses at most
  one session's work. **Resume with the same method**: a random scan with `--randomize`; without it
  the command runs the bisection search instead (still reusing the logged results);
- the file's first line fingerprints everything that affects the scores (tracker, profile settings,
  search space, detector, association, seed, ReID model, and the augmented copy if any). A mismatch
  stops the run instead of mixing results; `--fresh` moves the old file aside and starts over;
- the trial count is *not* part of the fingerprint: raising `--trials` continues the same sequence.

While running, each trial shows a live progress line (`trial 3: 17/52 sessions, 21.4 min`) and
then a summary line with its scores and settings. The bisection search first prints its starting
point, the number of numeric parameters and the trials per level.

**Output**: `outputs/tuning/tracker_<name>_<profile>[_<aug tag>][_bisect].csv` (every trial, with
per-group scores) and `.json` (the search method, the dataset, the baseline and best scores, and a
`config_patch` to paste into the config). Random-search results keep the names without `_bisect`.

## 8.7 `tune_models` (`scripts/tuning/tune_models.py`)

```
python -m scripts.tuning.tune_models [--component gate|expert|both] [--trials N]
```

Runs `nested_logo_search` on cached features. The gate is scored by 6-way macro-F1. The experts
are scored **under oracle context**, so expert tuning is not confused by gate errors: in
per-labeler mode by the per-label AUC (entered rating ≥ `labels.tau_att`) of each entry's own
labeler model over every held-out entry, with the folds over the label rows; in the window-grid
modes by window AUC. AUC, not MAE: on these labels a constant 3 beats the models on MAE. Fix
`labels.tau_att` first: the objective depends on it.

## 8.8 The dataset tools (`scripts/data_visualization/`)

| Tool | Run as | What it is for |
|---|---|---|
| `inspect_labels.py` | `python -m scripts.data_visualization.inspect_labels` | label distribution, labeler agreement, class balance for a grid of thresholds |
| `check_scb.py` | `python -m scripts.data_visualization.check_scb` | SCB detections drawn on 12 sampled frames (the §11 sanity check) |
| `scb_viewer.py` | `python -m scripts.data_visualization.scb_viewer g e c` | play a camera with SCB boxes; per-class confidence sliders |
| `augment_viewer.py` | `python -m scripts.data_visualization.augment_viewer g e c` | preview data augmentations (colour, noise, hflip, masking) on a frame, with the manual boxes transformed alongside; the same transforms the augmented training copies use |
| `label_viewer.py` | `python scripts/data_visualization/label_viewer.py g e c` | play a camera with every labeler's live attention values |
| `plot_labels.py` | `python scripts/data_visualization/plot_labels.py g e s` or `--scan` | attention step plots per student and per session |
| `player.py` | `python scripts/data_visualization/player.py data/DIPSER/DIPSEER` | six-camera synchronised player |

The first four import `src` and need `-m`; the last three are standalone files. Two more files
are one-off helpers: `camera_previews.py` saves the first frame of each camera ZIP, and
`annotation_helper.py` (hard-coded paths, no command line) was used while drawing the manual boxes.

Next to the downloader, `python -m scripts.downloading.validate_dipser_remote [--crc]` checks the
local copy against the FTP server without re-downloading it: Python's `zipfile` reads each server
archive through FTP byte ranges, so only the archive directories and the label files are fetched
(about 0.5 MB per 1 GB archive). `docs/experiments/01_dipser.md`, §7 lists the checks and results.

## 8.9 `benchmark` (`scripts/tuning/benchmark.py`)

```
python -m scripts.tuning.benchmark [--name trackers] [--variants V ...] [--sessions g/e/c ...] [--max-frames N]
```

Times the extraction pipeline per stage. For each variant in the config's `benchmark` section (a
set of dotted config overrides, by default the three tracker options) and each session (by default
the three quick-tuning sessions), it runs a fresh extraction with nothing cached and appends one
row to `outputs/benchmarks/<name>.csv`: per-stage seconds, frames, wall time, model-load time, and
the machine's average CPU load during the run. Other jobs inflate the CPU-bound stages, and the
load column makes such rows visible. Re-running skips rows already logged. At the end it writes
`outputs/benchmarks/<name>.md`: per variant, mean ms/frame per stage, processing fps, and the time
projected for a 45-minute session. Run it on an otherwise idle machine.

## 8.10 Where files go: hashes, `outputs/` and `tmp/`

Every number in the experiment documents can be traced back to the exact settings that produced
it, and every file it depends on sits somewhere a fresh clone can find or rebuild. Two mechanisms
make that work: the **config hash**, which names results, and a fixed rule for **what goes in
`outputs/` and what in `tmp/`**.

### How a run gets its hash

The hashing functions themselves are in Chapter 4 (§4.4). This is the path from a command line to
the 12 characters you see in a table (`c5341cb6385e`):

1. **A base file.** Either the live config (`configs/deliverable3.json`) or an experiment's frozen
   snapshot (`docs/experiments/NN_name.json`, below).
2. **Derived values.** If the file has a `derived` key, `load_config` runs `configs/derived.py`
   with the result files the config names there, and writes what it returns into the config:
   ```
   "derived": {"module": "configs/derived.py",
               "detector_scan": "outputs/tuning/detector_search_augmented.json",
               "tracker_scan": "outputs/tuning/tracker_botsort_full_reid_cmc_aug1-4f270e71_bisect.json"}
   ```
   These are the settings that come *from experiments*, read from their result files rather than
   typed in: `detector.conf` from the detector scan, the tracker and its options from the tracker
   scan's winner (it refuses a pair tuned at different thresholds). The base config names D2 and T2;
   experiment 09 names D3 and T3 (the pose model). One module, so each config only says which
   experiments its values come from. Each value is recorded with its source in `derived_from`
   (`file#key`).
3. **`--set` overrides**, applied last, so a command line always wins.
4. **The resolved config is hashed.** SHA-256 of the canonical JSON (sorted keys, no whitespace),
   leaving out the keys that do not change what a run produces: `output_dir`, `tuning`,
   `benchmark`, and `derived` / `derived_from` (only the *names* of the sources; the values they
   produced are in the config and hashed).
5. **Cited by its first 12 hex characters.** `train_models` prints `config <12 chars>` and writes
   `outputs/results/<full hash>.json`; the reporting scripts (`results_table`, `compare_results`) accept a prefix. 12 characters are
   48 bits, so an accidental clash among a few hundred runs is not a practical concern.

From the same resolved config, two partial hashes name the caches (§4.4): the **tracking hash**
(what the frame pass depends on: ingestion, detector, tracker, scene, the pixel-reading feature
units) and the **feature hash** (the tracking hash plus everything the feature pass reads).

What follows from this:

- **The file name means nothing.** Two configs with the same resolved settings have the same hash,
  wherever they live. `08_pca.json` began as a copy of `07_knn.json`, so their base runs share one
  hash (`b3fbc8430d44`).
- **Results from tuning flow into every hash.** Re-run D2 with a different outcome and
  `detector.conf` changes, so every tracking hash, feature hash and config hash changes with it,
  and nothing cached under the old threshold is reused by mistake.
- **Renaming a key changes the hash** even when behaviour does not. Experiment 08's runs were made
  when the class-context switch was called `engagement.late_context`; it is now
  `engagement.use_class_context`. The code still reads the old name, so the old results stay valid
  and are cited under the hashes they were made with.
- **Re-running the same config overwrites the same file.** That is intended: a result is a function
  of its config and the data, not of when it was run.
- **A document's hash can go stale before anything runs.** Experiment 09's header named its hash
  when the config was first written; switching the detector to the pose model changed the tracking
  hash and so the config hash. Recompute with `load_config` + `config_hash` before citing a hash
  for a run that has not happened yet.

**The other hashes in file names**, each the same `config_hash` on a smaller dict:

| Where | Name | Hashed from |
|---|---|---|
| `outputs/results/`, `outputs/models/`, `outputs/predictions/` | `<config hash>.json` / `.joblib` / `.csv.gz` | the whole resolved config |
| `outputs/cache/tracks/<tracking hash>/` | one `.npz` per session | the frame-pass settings |
| `outputs/cache/features/<feature hash>/` | one `.csv` (+ `.tracks.csv`, manifest) per session | tracking hash + feature-pass settings |
| augmented copies in both caches | `<session>.aug<k>-<8 chars>.*` | the `augmentation` settings and the augmentation code's version (§8.4) |
| `outputs/cache/detections*/` | one folder per detector setting | detector settings, frame limit, data source |
| `outputs/tuning/tracker_*_aug1-4f270e71_*` | the augmented copy's suffix | as above: the copy the scan ran on |

### Frozen experiment configs

The live config keeps changing (a new default, a renamed key, a new D-scan). So from experiment 06
on, each experiment document has a **frozen snapshot** of the config it used next to it
(`docs/experiments/06_full_pipeline_and_labels.json`, `07_knn.json`, ...):

1. take the resolved `config` from the experiment's base results file
   (`outputs/results/<hash>.json`), with the derived values already filled in;
2. drop the `derived` key, so loading the snapshot never re-reads today's tuning files; keep
   `derived_from` as a record of where the values came from;
3. write it next to the document and check that `load_config` on it reproduces the stored hash;
4. describe every arm of the experiment as `--set` overrides against that file.

A snapshot is never edited after the first run against it; a later experiment starts from a copy.
Experiment 09 is the exception that proves the rule: its config is written *before* its runs and
still has `derived` (naming D3 and T3), so it is frozen after the extraction, not before.

### `outputs/` or `tmp/`: the rule

**`outputs/` holds products; `scripts/experiments/` holds the scripts that made them; `tmp/` holds
the rest of the process.**
- **`outputs/`:** a file belongs here if a document cites it or a script reproduces it from the
  config and the data. It is named by hash or by experiment, and it is meant to survive.
- **`scripts/experiments/`:** an experiment's run chains and table builders, so a run can be
  repeated or extended. They are shared.
- **`tmp/`:** everything else that only *produced* those files, such as running logs, drafts, and
  scripts that pasted a table into a document once. It may be deleted once its products are archived.

| Folder | Contents | In git? |
|---|---|---|
| `outputs/results/` | evaluation results, by config hash | yes |
| `outputs/models/` | trained model bundles, by config hash; up to ~320 MB each (k-NN stores its training rows), rebuilt by retraining | no |
| `outputs/predictions/` | per-entry predictions of every training run, by config hash; a few MB each, rebuilt by retraining | no |
| `outputs/cache/detections*/` | detections and ReID embeddings for tuning: large, regenerable | no |
| `outputs/cache/tracks/`, `outputs/cache/features/` | frame-pass and feature-pass results: regenerable with `run_pipeline extract` | no |
| `outputs/experiments/<NN_name>/` | an experiment's own products: figures, CSVs, pilot overlays; once frozen, its run index and logs | yes |
| `outputs/tuning/`, `outputs/benchmarks/` | tuning and timing results (the D and T scans that `configs/derived*.py` read) | yes |
| `outputs/augmentation/`, `outputs/scb_check/` | augmentation samples and SCB detector overlays: previews of what the pipeline sees | yes |
| `outputs/labels/` | label agreement and the human ceiling, read by the scripts and cited by the documents | yes |
| `outputs/label_plots/` | rating plots per subject: inspection only, regenerable | no |
| `outputs/experiments/*/candidates/` | every clustering run's candidate features, 1–2 GB per run: an intermediate, regenerable with `scripts.clustering.candidates` | no |
| `docs/experiments/NN_name.json` | the frozen config snapshots | yes |
| `scripts/experiments/<NN_name>/` | an experiment's run chains and table builders | yes |
| `docs/run_reports/` | how a run went (time, errors, choices), when someone other than the owner ran it | yes |
| `tmp/` | working files that must outlive a session: running logs, drafts, smoke-test output folders | no |

The working files, concretely:
- **Run chains** (`scripts/experiments/08_pca/knn.sh`): one shell script per batch of arms, each arm
  a `train_models --config <snapshot> --set ...` line writing `<arm>.log`. They are **resumable**: an
  arm whose log already contains its `config` line is skipped, so a chain interrupted by a reboot is
  simply started again. Long ones run in your own terminal.
  - **Where the logs go:** while an experiment is running, its logs can go to `tmp/expNN/`. Once it
    is frozen, the chain points at `outputs/experiments/<NN_name>/logs/`.
- **Table builders** (`scripts/experiments/08_pca/build_head.py`): read the hashes, call
  `scripts/reporting/results_table.py`, and write the Markdown for the document to `tmp/`. A script
  that only pastes text into a document once stays in `tmp/` and is deleted afterwards.
- **Smoke tests and check harnesses**: `--set output_dir='"tmp/e2e6"'` keeps a trial run apart
  from the real results (`output_dir` is not hashed, so the hash is still the real one); the LaTeX
  check for `docs/test/test_cases.tex` lives in `tmp/tests_doc/tex/`.

Throwaway files that need not survive the session (a quick check, a one-line patch script) go in
the session's own temporary folder, not even `tmp/`.

**Two ways out of `tmp/`:**
- **Freezing an experiment** copies what its document depends on into `outputs/`: each cited run's
  training log goes to `outputs/experiments/<NN_name>/logs/`, and a `README.md` there lists every
  arm with its setting, results hash and log. Every listed hash is checked to have its results file,
  and the document gets a "Frozen" status line. Its run chains and table builders move to
  `scripts/experiments/<NN_name>/`, with their log paths pointed at the archive. After that,
  deleting `tmp/expNN/` loses nothing a document cites.
- **Promoting a script.** A general-purpose `tmp/` script that turns out to be needed again moves
  into `scripts/`, by role (`reporting/`, `analysis/`, `tuning/`, ...), with a docstring and a command
  line.

**Git** follows the same rule. Everything in `outputs/` is shared except what a fresh clone can
rebuild (the caches, the predictions and the model bundles), and `tmp/` is not shared at all; `.gitignore` names each
ignored folder on its own line, with a comment saying why. Model bundles (`.joblib`) are pickles:
only load ones you trust.

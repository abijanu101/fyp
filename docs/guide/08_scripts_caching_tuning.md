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
```

Every script takes `--config` (default `configs/deliverable3.json`) and `--set key=value`
overrides (Chapter 4).

## 8.2 `run_pipeline` (`scripts/run_pipeline.py`)

### `extract`

```
python -m scripts.run_pipeline extract [--session group_01/experiment_03/context_cam_02] [--reuse]
```

For each in-scope session (or the one given), it runs both passes and writes:

```
outputs/tracks/<tracking hash>/<session>.npz              the SessionTrack (frame pass)
outputs/features/<feature hash>/<session>.csv             window rows
outputs/features/<feature hash>/<session>.manifest.json   timing, stage seconds, hashes
outputs/features/<feature hash>/<session>.tracks.csv      per-student rows (late aggregation)
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
python -m scripts.train_models [--set labels.tau_att=3 labels.tau_share=0.6]
```

**Per-labeler mode (the default, `engagement.labelers = "per_labeler"`; Chapter 7 §7.9):**

1. **Load** the window table (gate features, as below) and every session's **rating entries**
   (`label_entries`): every entry of every labeler of the camera's students, the shared
   session-start entry excluded, with its training value (`labels.train_target`).
2. **Label-centred windows**: one window of `window.length_s` per entry time;
   `Pipeline.window_features` computes their features from the cached `.npz`, cached as
   `<session>.labelwin.csv` next to the window CSVs (recomputed if an entry time is missing). Each
   entry joins its student's linked track(s) in its window (`join_entries`).
3. **Leave-one-group-out**: per held-out group, the gate on the window grid and one model per
   labeler (`PerLabelerModel`) for the configured and the shared scope; every held-out entry scored
   per label (`rating_metrics`) in two tracks (experts, self) for three predictions (own labeler,
   median, majority) and two baselines (constant 3, training mean).
4. **Write** the results JSON (`"mode": "per_labeler"`, an `evaluation` block keyed
   `<track>/<variant>/<context>/<prediction>`), and the bundle with `"kind": "per_labeler"`, the
   gate, the per-labeler model, `gate_cols`, `row_cols` and `tau_att`.

Augmented copies (§8.4 below) join every training side and are never scored, as in the window-grid
modes.

**Window-grid modes** (`consensus`, `ensemble`):

1. **Load** every feature CSV for the current feature hash; attach group, experiment and window
   labels (`label_session`).
2. **Build the expert rows** (`engagement_rows`, §7.5): `vec.*` per window (early), or `trk.*`
   per student plus `vec.*` if `late_context` (late; read from `<session>.tracks.csv`, with each
   track linked to its student by `associate_sids` on the cached `.npz` to attach per-student
   labels). The gate gets `scene.*` (plus `vec.*` if `gate_uses_vec`).
3. **Leave-one-group-out**: for each held-out group, fit the gate and every mixture variant on
   the other two groups and score on the held-out one (Chapter 7).
4. **Write** `outputs/results/<config hash>.json`: the full config, data summary (windows, label
   coverage, class balance), every fold's metrics, and mean ± std summaries.
5. **Fit final models on all groups** and save `outputs/models/<config hash>.joblib`, the bundle
   `infer` loads. It contains the gate, the mixture, `gate_cols`, `row_cols`, the aggregation mode
   (`aggregate`), `fill` (score for windows with nobody visible) and `tau` (the engaged cut-off).

### Seeing the curves: `plot_results` (`scripts/plot_results.py`)

```
python -m scripts.plot_results [--set ...]        # same --set values as train_models
```

`train_models` reports averages. `plot_results` draws, per session, out-of-sample curves with the
gate's predicted activity as background shading. In per-labeler mode, on the 1–5 scale: each
labeler model's class curve (experts blue, self orange), the final curve in black (median or
majority, `output.prediction`), every held-out rating entry as a dot at its time (experts ×, self
●) and `tau_att` dashed; the title gives the final curve's MAE against the expert entries
(`heldout_curves` in `train_models` computes the curves). In the window-grid modes it draws the
predicted engagement score against the labelled share of engaged students, with the engaged
cut-off. Every curve is out-of-sample: it repeats the
leave-one-group-out fits (seconds) and plots each group from models that never saw it. The
title gives the true activity, the gate's hit rate and the correlation r with the labelled
share. Output: `<output_dir>/plots/<config hash>/`, one PNG per session plus `overview.png`.

### Watching a session: `output_viewer` (`scripts/output_viewer.py`)

```
python -m scripts.output_viewer group_01 experiment_03 context_cam_01 [--set ...] [--models <bundle>]
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
models are trained on originals and copies. The results file reports
`augmentation_copies`, `augmented_windows_for_training` and `augmented_row_label_coverage` (the
share of copy rows linked to a labelled student; it should be close to the originals', which
confirms the mirrored boxes line up). `tune_models` does not use copies.

**Looking at it.** `python -m scripts.augment_samples` writes, for three sessions, the original
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
region is known, a directed search is cheaper. `beam_bisection_search` is a binary search per
parameter, organised as branch and bound:

- A **node** is a point plus, for each numeric parameter, an interval around it (in [0, 1] units,
  log scale where the space says `log`). The root is the starting point with full intervals.
- **Branching.** For each of the k numeric parameters, probe the midpoints of the lower and upper
  halves of its interval, keep the best of the three points, and halve the interval around it:
  one bisection step per parameter, k branches per node. One more branch applies every branch's
  winning move at once.
- **Bound.** Only the `beam` best nodes of each level are expanded further. The search stops when
  the trial budget is spent or every interval is narrower than 1/64 of its range.
- Categorical parameters, and numeric parameters with no effect under the profile's fixed settings
  (`inactive`, e.g. BoT-SORT's appearance thresholds when ReID is off), keep their starting values.
- Deterministic: re-running proposes the same trials, so resuming works as for random search.

Each level costs up to (2k + 1) × beam trials: 13 × 2 = 26 for BoT-SORT without ReID (6 numeric
parameters). On a smooth test function it reaches the optimum within about 60 trials where random
search is still far off (`tests/test_tuning.py`); real tracking scores are noisier.

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

## 8.6 `tune_tracker` (`scripts/tune_tracker.py`)

```
python -m scripts.tune_tracker --profile <profile> [--trials N] [--fresh]       # beam bisection (default)
python -m scripts.tune_tracker --profile <profile> --randomize                   # seeded random search
python -m scripts.tune_tracker --profile <profile> --augment [COPY]              # on augmented data
python -m scripts.tune_tracker --profile <profile> --check
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
drop whole students; `docs/experiments/02_boxmot.md`, §3).

**Is the detector the limit?** `python -m scripts.scan_detector` checks, without any tracker, what
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
- `--augment [COPY] --detect` runs the same on augmented copy COPY; `--detect` runs the detector
  where that copy's detections are not cached yet (GPU). The full-dataset profiles keep CMC **on** everywhere: every trial then decodes
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

## 8.7 `tune_models` (`scripts/tune_models.py`)

```
python -m scripts.tune_models [--component gate|expert|both] [--trials N]
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
| `augment_viewer.py` | `python -m scripts.data_visualization.augment_viewer g e c` | preview data augmentations (colour, noise, hflip, masking) on a frame, with the manual boxes transformed alongside; not yet used by the pipeline |
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

## 8.9 `benchmark` (`scripts/benchmark.py`)

```
python -m scripts.benchmark [--name trackers] [--variants V ...] [--sessions g/e/c ...] [--max-frames N]
```

Times the extraction pipeline per stage. For each variant in the config's `benchmark` section (a
set of dotted config overrides, by default the three tracker options) and each session (by default
the three quick-tuning sessions), it runs a fresh extraction with nothing cached and appends one
row to `outputs/benchmarks/<name>.csv`: per-stage seconds, frames, wall time, model-load time, and
the machine's average CPU load during the run. Other jobs inflate the CPU-bound stages, and the
load column makes such rows visible. Re-running skips rows already logged. At the end it writes
`outputs/benchmarks/<name>.md`: per variant, mean ms/frame per stage, processing fps, and the time
projected for a 45-minute session. Run it on an otherwise idle machine.

## 8.10 Where files go, and what is shared

| Folder | Contents | In git? |
|---|---|---|
| `outputs/cache/` | detections and ReID embeddings for tuning: large, regenerable | no |
| `outputs/tracks/`, `outputs/features/` | frame-pass and feature-pass results | yes |
| `outputs/results/`, `outputs/models/` | evaluation results, trained model bundles | yes |
| `outputs/plots/` | held-out engagement curves from `plot_results` | yes |
| `outputs/augmentation/` | augmentation samples and extremes from `augment_samples` | yes |
| `outputs/tuning/`, `outputs/benchmarks/` | tuning and timing logs | yes |
| `outputs/labels/`, `outputs/label_plots/`, `outputs/scb_check/` | dataset inspection outputs | yes |
| `tmp/` | working files that need to live a while (test runs, one-off scripts) | no |

Everything under `outputs/` except the cache is meant to be shared. `outputs/tracks/` grows by
about 50 MB per tracker configuration extracted; if that becomes heavy, it is the next candidate
for the ignore list. Model bundles (`.joblib`) are pickles: only load ones you trust.

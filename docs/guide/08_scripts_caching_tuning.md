# Chapter 8 — Scripts, caches and tuning

The `src/` package is a library: it has no `main`. Everything you run lives in `scripts/`. This
chapter covers each entry point, what it caches where, and how the tuning machinery works.

## 8.1 The workflow at a glance

```
                      ┌───────────────────────────┐
                      │ scripts.dipser.*           │  look at the data: labels, SCB, video
                      └───────────────────────────┘
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
```

**By default nothing is read back.** Every session runs the full frame pass, so each manifest's
`stage_seconds` are real runtime numbers. With `--reuse` (for development):

- sessions whose CSV already exists are skipped;
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

1. **Load** every feature CSV for the current feature hash; attach group, experiment and window
   labels (`label_session`).
2. **Select columns by prefix**: `vec.*` for the experts, `scene.*` for the gate (plus `vec.*`
   if `gate_uses_vec`).
3. **Leave-one-group-out**: for each held-out group, fit the gate and every mixture variant on
   the other two groups and score on the held-out one (Chapter 7).
4. **Write** `outputs/results/<config hash>.json`: the full config, data summary (windows, label
   coverage, class balance), every fold's metrics, and mean ± std summaries.
5. **Fit final models on all groups** and save `outputs/models/<config hash>.joblib`, the bundle
   `infer` loads. It contains the gate, the mixture, and the exact column lists.

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
  there is no model of the objective. With about 10 parameters and 25–40 trials this is a
  reasonable choice: usually only a few parameters matter, and random sampling tries many distinct
  values of each, where a grid would repeat the same few.
- The generator is seeded, so re-running a search draws exactly the same parameter sets.
- Results come back sorted, best first. Several trials within about 0.01 of the best should be
  treated as ties.

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
python -m scripts.tune_tracker --profile quick|full [--tracker botsort|strongsort] [--trials N]
python -m scripts.tune_tracker --profile <name> --check
```

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
across two track ids, only the larger part counts.

**Why it is fast.** Detection is independent of tracker settings, so each session's detections
are computed once at a low confidence floor (0.05) and cached in `outputs/cache/detections/`.
Each trial filters them at its own `det_conf` (so detector confidence is tuned at no extra cost)
and re-runs only the tracker.

**Why blank frames.** BoxMOT's tracker takes the image as an input. With ReID and camera-motion
compensation off, BoT-SORT never reads it, so the full profile feeds a blank frame of the right
size instead of decoding PNGs: about 2 minutes per trial on all 52 sessions instead of 35.
`--check` verifies that this fast path (blank frames, and cached ReID embeddings for profiles that
use ReID) gives identical tracks to the tracker's own path before relying on it.

**Profiles** (`tuning.tracker.profiles`):

- `quick`: 3 sessions × first 300 frames; ReID may be searched.
- `full`: all sessions, all frames; ReID and CMC fixed off (`fixed`).

**Output**: `outputs/tuning/tracker_<name>_<profile>.csv` (every trial, with per-group scores)
and `.json`, which includes a `config_patch` to paste into the config.

## 8.7 `tune_models` (`scripts/tune_models.py`)

```
python -m scripts.tune_models [--component gate|expert|both] [--trials N]
```

Runs `nested_logo_search` on cached features. The gate is scored by 6-way macro-F1. The experts
are scored by AUC **under oracle context**, so expert tuning is not confused by gate errors. Fix
the label thresholds first: the expert objective depends on them.

## 8.8 The dataset tools (`scripts/dipser/`)

| Tool | Run as | What it is for |
|---|---|---|
| `inspect_labels.py` | `python -m scripts.dipser.inspect_labels` | label distribution, labeler agreement, class balance for a grid of thresholds |
| `check_scb.py` | `python -m scripts.dipser.check_scb` | SCB detections drawn on 12 sampled frames (the §11 sanity check) |
| `scb_viewer.py` | `python -m scripts.dipser.scb_viewer g e c` | play a camera with SCB boxes; per-class confidence sliders |
| `label_viewer.py` | `python scripts/dipser/label_viewer.py g e c` | play a camera with every labeler's live attention values |
| `plot_labels.py` | `python scripts/dipser/plot_labels.py g e s` or `--scan` | attention step plots per student and per session |
| `player.py` | `python scripts/dipser/player.py data/DIPSER/DIPSEER` | six-camera synchronised player |

The first three import `src` and need `-m`; the last three are standalone files.

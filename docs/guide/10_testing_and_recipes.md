# Chapter 10 — Tests and recipes

This chapter covers how the tests are organised and how to write one, then gives step-by-step
recipes for the changes you are most likely to make.

## 10.1 Running the tests

```
python -m unittest discover -s tests -t .
```

The suite uses only the standard library's `unittest` (pytest is not installed, though it would
run these tests unchanged). It needs no GPU, no weights and almost no data. It runs in a few
seconds, so run it before every commit.

To run one file or one test:

```
python -m unittest tests.test_labels
python -m unittest tests.test_stages.TestTracklets.test_camera_frame_drop_is_not_a_gap
```

## 10.2 What is tested where

| File | Covers |
|---|---|
| `tests/test_labels.py` | clock parsing, step functions, time-weighted means, out-of-range values, window labelling rules |
| `tests/test_spread.py` | CFMU spread maths: within² + between² = total², edge cases |
| `tests/test_aggregation_modes.py` | early rows reproduce the window-level mixture exactly; late rows carry window context; window scores average rows and fill empty windows; per-student labels agree with the window share; the feature pass emits one row per visible track |
| `tests/test_augment.py` | box mirroring, parameters fixed per (session, copy), hue range, copy tags, the same transform on every frame, the augmented source as a torch Dataset with the original timestamps, a copy reading back as the same video every time, augmented rows never in a test fold |
| `tests/test_label_targets.py` | per-labeler training: interpolation modes (step values, NaN before the first entry, 1–5 clipping in every mode), window means and the centred-window blend, normalising and reading back ratings with clipping, rounding halves toward 3, the majority vote's ties and resolution, rating metrics, soft labels in logistic regression (calibrated) and the SVM/k-NN fallback (sign kept), one model per labeler learning only its own labels, window curves in both prediction modes, the rating graph's records, joining entries to linked tracks |
| `tests/test_per_labeler.py` | the labeler ensemble: a labeler who did not rate a student has no vote, opposed labelers cancel where both rated, all labelers vote without labeler information (inference), self-ratings excluded, consensus mode unchanged |
| `tests/test_mixture.py` | soft/hard/oracle mixing, shared scope, per-experiment training, hand-weighted signs, LOGO splits |
| `tests/test_stages.py` | windows, tracklet gap filtering, SID association (including passing TAs), motion features, Vec key stability, config hashing and cache keys, track store round trip, the feature pass without video, graph privacy |
| `tests/test_tuning.py` | Krippendorff's alpha (against the published example), tracking metrics, random and nested search |
| `tests/test_scb.py` | SCB keys in soft and flags mode, soft score assignment (best-IoU track, max per class, one track per candidate), mean and peak window features, flags matching, per-class thresholds, the safe loader refusing bad checkpoints |

## 10.3 The testing style: small synthetic inputs with known answers

Most tests build a tiny input by hand where the right answer can be worked out on paper. For
example, from `tests/test_stages.py`:

```python
def moving_tracklet(tid, n=20, dt=0.4, vx=10.0, x0=100.0, y0=100.0):
    t = np.arange(n) * dt
    x = x0 + vx * t
    xyxy = np.stack([x, np.full(n, y0), x + 50, np.full(n, y0 + 100)], 1)
    return Tracklet(tid, t, xyxy, np.ones(n), np.full((n, 2), np.nan))

def test_finite_difference_speed(self):
    kg = FEATURE_EXTRACTORS.build("kalman_geometry", use_kalman=True)
    f = kg.window_features(moving_tracklet(1, vx=10.0), fps=2.5)
    self.assertAlmostEqual(f["kg.speed"], 10.0 / np.hypot(50, 100), places=6)
```

A box 50×100 px moving at 10 px/s has speed 10 / diagonal in diagonals per second. The test
checks exactly that.

Guidelines:

- **Test behaviour you could get wrong**, not that Python works. "A camera dropout does not split
  a track" is worth a test; "the dataclass stores its fields" is not.
- **Construct through the registry** (`FEATURE_EXTRACTORS.build(...)`) where you can, so the test
  also checks registration.
- **No GPU, no large data.** Things that need weights are either constructed lazily (so building
  them in a test is free) or skipped with `@unittest.skipUnless(...)`.
- **Name tests as sentences** about behaviour: `test_walker_passing_a_student_stays_unassociated`.

## 10.4 Recipe: add a BFMU feature family

Chapter 3, section 3.14 walks through a complete example. In short:

1. Write a `FeatureExtractor` subclass in `src/bfmu/<name>.py`, decorated with
   `@FEATURE_EXTRACTORS.register("<name>")`, with `keys` and `window_features`.
2. If it needs pixels, set `uses_frames = True`, implement `frame_signals` (per-track, per-frame
   values) and `reset` (clear per-session state), and read the values back from `tr.signals` in
   `window_features`.
3. Import the module in `src/bfmu/__init__.py`.
4. Add an entry to `bfmu` in the config.
5. Add a test with a synthetic `Tracklet`.
6. Re-extract: `run_pipeline extract --reuse` is enough for a window-only extractor; a pixel
   extractor changes the tracking hash and needs a frame pass.

Optionally add thresholds for the new keys under `aggregation.thresholds` to get `.frac` columns.

## 10.5 Recipe: try a different tracker setting

For a one-off comparison, no code changes are needed:

```
python -m scripts.run_pipeline extract --set tracker.options.max_lost_frames=40
```

For a systematic search, edit `tuning.tracker.space.botsort` and run `tune_tracker` (Chapter 8,
§8.6), then paste the resulting `config_patch` into the config:

```
python -m scripts.tuning.tune_tracker --profile quick --randomize          # explore: seeded random search
python -m scripts.tuning.tune_tracker --profile quick --trials 26          # refine: beam bisection from its winner
python -m scripts.tuning.tune_tracker --profile quick --augment 1          # the same on augmented data
```

Explore first, refine second: the bisection search starts from the profile's random-search winner
automatically. Prototype on the `quick` profile (minutes per trial) before a full-dataset profile
(about an hour per trial with CMC on).

## 10.6 Recipe: add a tracker

1. If it is in BoxMOT, subclass `BoxmotTracker` and set `name`:

   ```python
   @TRACKERS.register("bytetrack")
   class ByteTrack(BoxmotTracker):
       name = "bytetrack"
   ```

   Check that the Kalman-velocity read in `_kalman_velocities` still applies; if the tracker's
   state layout differs, velocities may be wrong rather than missing.
2. Otherwise, subclass `Tracker` and implement `update(frame, dets) -> list[TrackState]`.
3. Add a search space under `tuning.tracker.space.<name>` to tune it. In `scripts/tuning/tune_tracker.py`,
   check `needs_pixels`, `needs_embeddings` and the encoder construction, which name trackers
   explicitly: a tracker that always uses ReID must be added to `needs_embeddings`. Then run
   `--check` to confirm the cached-embedding path reproduces the tracker's own tracks.

## 10.7 Recipe: add a gate or expert model

For another scikit-learn classifier, add one line to `CLASSIFIERS` in `src/core/classifiers.py`:

```python
"gboost": lambda seed, **p: GradientBoostingClassifier(random_state=seed, **p),
```

The `GATES` registry picks it up automatically (its registration loop iterates over
`CLASSIFIERS`). For experts, also add a registration line in `src/experts/expert.py`:

```python
EXPERTS.register("gboost")(partial(SklearnExpert, "gboost"))
```

Then set `gate.name` or `expert.name` in the config, and add a search space under
`tuning.gate.space` or `tuning.expert.space` to tune it.

For something that is not scikit-learn, subclass `EngagementExpert` (`fit`, `predict_proba`) and
register it. Be careful that `fit` uses only its arguments: an expert that looked at global data
would leak across folds.

## 10.8 Recipe: switch on CFMU

1. Implement a `ClusterFeatureExtractor` in `src/cfmu/`, register it (for example as
   `spread_sd`), and import it in `src/cfmu/__init__.py`.
2. Its `window_features(window, tracks, per_track)` receives the window's visible tracklets
   (already sliced) and their BFMU feature dicts, in the same order. Cluster the tracks
   (§10.3), build descriptors, call `spread_decomposition`, and return `cluster.*` keys.
3. Set `"cfmu": {"name": "spread_sd", ...}` in the config.
4. Return the same keys in every window (NaN when there are too few tracks to cluster), like the
   aggregator does. Otherwise feature tables get ragged columns.

Nothing else changes: Vec widens, and the experts are retrained on the wider Vec.

## 10.9 Recipe: run a sensitivity sweep

The design suggests sweeping window length and label thresholds, each under its own config hash.
Because of the two-level cache, a window sweep is cheap once tracks exist:

```
python -m scripts.run_pipeline extract                                  # once: frame pass
for L in 6 8 10 12:
    python -m scripts.run_pipeline extract --reuse --set window.length_s=$L
    python -m scripts.train_models --set window.length_s=$L
```

Each `train_models` run writes its own `outputs/results/<hash>.json`.

## 10.10 Recipe: compare per-labeler training settings

All are **T** except the window length, which is **P** (training only; inference settings are free):

```
python -m scripts.train_models                                              # train_with_interp, step labels
python -m scripts.train_models --set labels.train_target='"train_vanilla"'
python -m scripts.train_models --set labels.interp.mode='"gauss"' labels.interp.sigma_s=2
python -m scripts.run_pipeline extract --reuse --set window.length_s=8      # another window length (P): feature pass first
python -m scripts.train_models --set window.length_s=8
```

Each writes its own results file. Compare the `own_labeler` rows of both tracks (experts, self)
against the `constant_3` and `train_mean` baselines on AUC, Spearman and balanced accuracy; MAE and
accuracy flatter constants on these labels (`docs/misc/targets.md` §3). `output.prediction` and
`output.majority_resolution` need no retraining: training evaluates both predictions every run.

## 10.11 Recipe: compare early and late aggregation (window-grid modes)

No code changes; the features already contain both kinds of row:

```
python -m scripts.train_models --set engagement.labelers='"consensus"' engagement.aggregate='"early"' labels.tau_att=3 labels.tau_share=0.6
python -m scripts.train_models --set engagement.labelers='"consensus"' engagement.aggregate='"late"'  labels.tau_att=3 labels.tau_share=0.6
```

Each writes its own `outputs/results/<hash>.json` (the setting is part of the config hash) with the
same window-level metrics. Also check `data.row_label_coverage` in the late results: the share of
student rows that could be linked to a labelled student. Low coverage means association, not the
model, is limiting late aggregation. `engagement.late_context=false` tests whether students'
own features carry any signal without the classroom context.

Compare the two on AUC first. F1 and balanced accuracy depend on each mode's
cut-off (`mixture.tau` for early, `labels.tau_share` for late, §7.5), and the late score is not a
calibrated share.

## 10.12 Recipe: measure the cost of a change

Add a variant to the config's `benchmark.variants` as dotted overrides, for example a cheaper flow
setting:

```json
"flow_quarter_res": {"bfmu": [{"name": "kalman_geometry", "still_threshold": 0.05},
                              {"name": "residual_flow", "grid": 2, "scale": 0.25}, {...scb entry...}]}
```

then run `python -m scripts.tuning.benchmark --name flow --variants botsort_motion flow_quarter_res` on an
idle machine and compare the two rows of `outputs/benchmarks/flow.md`. Check what the change does
to the features, not only to the time: a faster pipeline that changes Vec needs the downstream
evaluation re-run.

## 10.13 Recipe: run something that takes hours

Long jobs are run in your own terminal, so their progress is visible, and are written to resume:
they append each finished unit of work to a log as it completes and skip logged units on restart
(`tune_tracker`'s `.sessions.jsonl`, `benchmark`'s CSV, `run_pipeline extract --reuse`). If you
write a new long-running script, follow the same pattern: small units, append-as-you-go, a
fingerprint of the settings so a restart cannot mix results, and a progress line on stdout.

## 10.14 Before you commit

- Tests pass.
- If you changed a contract (a dataclass, config keys, a cache format), update the matching
  chapter of this guide.
- If you changed anything feeding a cache key, expect existing caches to be ignored, not reused.
  That is by design.

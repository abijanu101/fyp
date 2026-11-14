# Chapter 9 — Tests and recipes

This chapter covers how the tests are organised and how to write one, then gives step-by-step
recipes for the changes you are most likely to make.

## 9.1 Running the tests

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

## 9.2 What is tested where

| File | Covers |
|---|---|
| `tests/test_labels.py` | clock parsing, step functions, time-weighted means, out-of-range values, window labelling rules |
| `tests/test_spread.py` | CFMU spread maths: within² + between² = total², edge cases |
| `tests/test_mixture.py` | soft/hard/oracle mixing, shared scope, per-experiment training, hand-weighted signs, LOGO splits |
| `tests/test_stages.py` | windows, tracklet gap filtering, SID association (including passing TAs), motion features, Vec key stability, config hashing and cache keys, track store round trip, the feature pass without video, graph privacy |
| `tests/test_tuning.py` | Krippendorff's alpha (against the published example), tracking metrics, random and nested search |
| `tests/test_scb.py` | SCB keys, soft score assignment, flags matching, per-class thresholds, the safe loader refusing bad checkpoints |

## 9.3 The testing style: small synthetic inputs with known answers

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

## 9.4 Recipe: add a BFMU feature family

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

## 9.5 Recipe: try a different tracker setting

For a one-off comparison, no code changes are needed:

```
python -m scripts.run_pipeline extract --set tracker.options.max_lost_frames=40
```

For a systematic search, edit `tuning.tracker.space.botsort` and run `tune_tracker` (Chapter 8),
then paste the resulting `config_patch` into the config.

## 9.6 Recipe: add a tracker

1. If it is in BoxMOT, subclass `BoxmotTracker` and set `name`:

   ```python
   @TRACKERS.register("bytetrack")
   class ByteTrack(BoxmotTracker):
       name = "bytetrack"
   ```

   Check that the Kalman-velocity read in `_kalman_velocities` still applies; if the tracker's
   state layout differs, velocities may be wrong rather than missing.
2. Otherwise, subclass `Tracker` and implement `update(frame, dets) -> list[TrackState]`.
3. Add a search space under `tuning.tracker.space.<name>` to tune it, and extend `needs_pixels`
   in `scripts/tune_tracker.py` if it can run without reading images.

## 9.7 Recipe: add a gate or expert model

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

## 9.8 Recipe: switch on CFMU

1. Implement a `ClusterFeatureExtractor` in `src/cfmu/`, register it (for example as
   `spread_sd`), and import it in `src/cfmu/__init__.py`.
2. Its `window_features(window, tracks, per_track)` receives the window's visible tracklets
   (already sliced) and their BFMU feature dicts, in the same order. Cluster the tracks
   (§10.3), build descriptors, call `spread_decomposition`, and return `cluster.*` keys.
3. Set `"cfmu": {"name": "spread_sd", ...}` in the config.
4. Return the same keys in every window (NaN when there are too few tracks to cluster), like the
   aggregator does. Otherwise feature tables get ragged columns.

Nothing else changes: Vec widens, and the experts are retrained on the wider Vec.

## 9.9 Recipe: run a sensitivity sweep

The design suggests sweeping window length and label thresholds, each under its own config hash.
Because of the two-level cache, a window sweep is cheap once tracks exist:

```
python -m scripts.run_pipeline extract                                  # once: frame pass
for L in 6 8 10 12:
    python -m scripts.run_pipeline extract --reuse --set window.length_s=$L
    python -m scripts.train_models --set window.length_s=$L
```

Each `train_models` run writes its own `outputs/results/<hash>.json`.

## 9.10 Before you commit

- Tests pass.
- If you changed a contract (a dataclass, config keys, a cache format), update the matching
  chapter of this guide.
- If you changed anything feeding a cache key, expect existing caches to be ignored, not reused.
  That is by design.

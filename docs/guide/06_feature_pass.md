# Chapter 6 — The feature pass

The feature pass turns one `SessionTrack` (Chapter 5) into one `SessionFeatures`: a row of numbers
per time window. It does no image processing and runs in about a second per session, which is
why it is separate from the frame pass.

```python
def features(self, st: SessionTrack) -> SessionFeatures:
    fps = st.timing.get("fps") or 1.0
    tl = self.cfg["tracklets"]
    tracklets = filter_visible(st.tracklets, st.frame_t, fps, tl["min_visible_s"], tl["max_gap_s"])
    win = self.cfg["window"]
    windows = make_windows(st.timing["t_first"], st.timing["t_last"], win["length_s"], win["overlap"])
    vec, track_rows = [], []
    for i, w in enumerate(windows):
        v, rows = self.aggregator.aggregate_with_tracks(w, tracklets, fps, st.frame_wh)
        vec.append(v)
        track_rows += [{"window": i, **r} for r in rows]
    scene = [self.scene.window_features([st.scene_rows[i] for i in frames_in(st.frame_t, w)])
             for w in windows]
    return SessionFeatures(st.key, windows, vec, scene, manifest, track_rows)
```

Four steps: filter tracklets, cut windows, build Vec per window, summarise scene features per
window. This chapter takes them in order.

## 6.1 Filtering tracklets

The tracker reports everything, including people seen for half a second, a teacher walking past,
and tracks that break when someone is hidden. `filter_visible` (`src/tracking/tracklets.py`)
cleans this up in two steps.

**Split where the tracker lost the person.** If a track is missing for more than `max_gap_s`
(1 s), it is split into separate pieces. The subtle part is how the gap is measured:

```python
pos = np.searchsorted(frame_t, tr.t)           # which frame each observation came from
missing_s = (np.diff(pos) - 1) / fps           # frames that EXIST but lack this track, in seconds
cuts = np.flatnonzero(missing_s > max_gap_s) + 1
```

The gap counts *frames that were decoded but in which the person was not tracked*, not
wall-clock time. DIPSER cameras sometimes drop several seconds of frames. If the tracker kept the
same TID across such a drop, the tracker never lost the person; a wall-clock rule would cut every
track at every camera dropout. `SessionTrack.frame_t` exists for exactly this calculation.

**Drop short pieces.** Pieces visible for less than `min_visible_s` (3 s), measured as number of
observations ÷ fps, are dropped (§4: "a track is dropped below ~3 s cumulative visible duration").

## 6.2 Windows

```python
make_windows(t_first, t_last, length_s=8, overlap=0.5)
# -> [Window(t0, t0+8), Window(t0+4, t0+12), Window(t0+8, t0+16), ...]
```

The step is `length × (1 − overlap)`: 4 s for 8-second windows at 50% overlap. A trailing partial
window is dropped, so every window has the full length. A 5-minute session yields 73 windows.

Overlapping windows mean consecutive rows of the feature table share half their data. That is
one reason the evaluation never splits windows randomly into train and test (Chapter 7).

## 6.3 Per-track window features (stage 5, window part)

For each window, each visible track is sliced to the window (`tr.slice(t0, t1)`), and every
extractor's `window_features` turns that slice into a few numbers.

### Kalman/geometry motion (`src/bfmu/kalman_geometry.py`)

This extractor works only from boxes and velocities, no pixels. First it gets a velocity per
frame, preferring the tracker's Kalman estimate:

```python
def velocity(self, tr, fps):
    fd = np.gradient(tr.centers, tr.t, axis=0)       # finite differences, px/s
    kalman = tr.velocity * fps                        # Kalman px/frame -> px/s
    return np.where(np.isfinite(kalman), kalman, fd)  # Kalman where available
```

Then it divides by the track's mean box diagonal, so units are **box diagonals per second**. A
student in the back row whose box is half the size, moving the same real distance, gets the same
number as one in the front row.

| Key | Definition | Reads as |
|---|---|---|
| `kg.vx`, `kg.vy` | mean velocity components | drifting sideways / up-down |
| `kg.speed` | mean speed | how much they move |
| `kg.accel` | mean absolute change in speed per second | jerky vs smooth |
| `kg.stillness` | share of frames with speed below `still_threshold` (0.05) | sitting still |
| `kg.energy` | root-mean-square speed | overall motion energy |
| `kg.events` | velocity sign reversals per second | fidgeting, back-and-forth |
| `kg.aspect_trend` | slope of box width/height over time | posture change, leaning |
| `kg.area_trend` | slope of box area (relative) | moving toward/away from camera |
| `kg.cy_trend` | slope of box centre height (in box heights/s) | slumping, head going down |

Trends are slopes of a straight-line fit over the window (`np.polyfit` degree 1).

### Residual flow and SCB

These two already computed per-frame values in the frame pass and stored them in
`tr.signals`. Their `window_features` only summarise:

- residual flow: the mean of each grid cell's signal over the window (`rf.mag_00`, ...);
- SCB soft mode: the mean **and** the peak of each score (`scb.read`, `scb.read.max`, ...). The
  peak keeps brief events, such as a hand going up for one second, that a mean would wash out.

A shared helper, `signal_mean` in `src/bfmu/base.py`, returns NaN rather than a number when a
track has no valid values.

## 6.4 Vec: from people to the classroom (`src/aggregation/vec.py`)

`WindowAggregator.aggregate_with_tracks` turns the students visible in one window into Vec, and
also returns the per-student rows it built Vec from (step 2 below), which late aggregation feeds
to the experts directly. `aggregate` returns Vec alone. For one window:

**1. Which tracks are visible?** A track counts if it covers at least `min_coverage` (50%) of the
window:

```python
s = tr.slice(window.t_start, window.t_end)
if len(s.t) >= 2 and (s.t[-1] - s.t[0]) >= self.min_coverage * length:
    visible.append(s)
```

**2. Per-track features.** Every extractor's `window_features` for every visible track gives a
list of dicts, one per student.

**3. Summarise across students.** For every per-track key:

```python
vec[f"{key}.mean"] = mean over students        # e.g. kg.stillness.mean: how still the class is
vec[f"{key}.std"]  = std over students         # how much students differ from each other
if key in thresholds:
    vec[f"{key}.frac"] = share of students above the threshold
                                               # e.g. kg.stillness.frac: share of "still" students
```

**4. Spatial occupancy.** The frame is cut into a 3×3 grid; `occ_ij` is the share of visible
students whose centre falls in cell (i, j).

**5. Count.** `n_tracks`: how many students were visible.

**6. CFMU.** The cluster extractor's features are appended. Currently the `null` extractor
returns nothing.

### Every window has the same keys

When no track is visible, every `.mean` and `.std` is NaN, but the keys are still there. The
aggregator builds its key list from the extractors' declared `keys`, never from the data. So every
row of the feature table has the same columns, whatever happened in that window. A test checks
this (`test_vec_keys_stable_and_nan_when_empty`).

### The width of Vec

With the current config: 10 motion keys, 4 flow keys and 8 SCB keys make 22 per-track features;
each gets `.mean` and `.std` (44 columns), plus 4 `.frac` columns, 9 occupancy cells and
`n_tracks`, for 58 columns. With early aggregation the experts see these 58 numbers per window;
with late aggregation they see each student's 22 per-track features (as `trk.*` columns), plus
these 58 as context if `engagement.late_context` is on.

## 6.5 Scene windows

The gate's features are summarised the same way but over frames instead of students:
`frames_in(frame_t, window)` finds the frames inside the window, and
`SceneFeatureExtractor.window_features` returns the mean and standard deviation of each per-frame
scene feature across them (`scene.motion.mean`, `scene.motion.std`, ...).

## 6.6 CFMU: the reserved slot (`src/cfmu/`)

The *Cluster Feature Mining Unit* is part of FYP-1 but not Deliverable III (§10). Its slot exists
so that turning it on changes nothing except the width of Vec:

```python
class ClusterFeatureExtractor(ABC):
    @abstractmethod
    def window_features(self, window, tracks, per_track) -> dict[str, float]: ...

@CLUSTER_EXTRACTORS.register("null")
class NullClusterExtractor(ClusterFeatureExtractor):
    def window_features(self, window, tracks, per_track):
        return {}
```

It receives the window's visible tracks *and* their BFMU features, which is everything §10.4
needs. The mathematical core of §10.4 is already written and tested in `src/cfmu/spread.py`:

```python
spread_decomposition(x, labels, blocks)
# within-cluster SD, between-cluster SD, total SD per feature, and the between-share per block
```

The tests check the identity the design relies on: within² + between² = total² for every feature.

## 6.7 The saved feature table

`to_frame(feats)` builds the window table, and `run_pipeline extract` saves it as one CSV per
session under `outputs/features/<feature hash>/`, with one row per window:

```
t_start, t_end, vec.n_tracks, vec.kg.speed.mean, vec.kg.speed.std, ..., scene.motion.mean, ...
```

A second file per session, `<session>.tracks.csv`, holds the per-student rows: `window` (row
index in the window CSV), `tid`, and `trk.*` per-track features.

`scripts/train_models.py` reads all these CSVs, adds group, experiment and label columns, and
trains on them (Chapter 7). The window CSV is written **last**, after the session's manifest
and its `.tracks.csv`, so its existence marks a completed session.

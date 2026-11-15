# Chapter 5 — The frame pass

The frame pass is everything that touches pixels: reading frames, finding people, following them
over time, and computing per-frame signals that need the image. It is the slow part of the
pipeline (minutes per session) and produces one `SessionTrack` per session, which is cached.

This chapter follows one frame through `Pipeline.track()`, then looks at each stage in turn.

## 5.1 The loop

Here is the core of `Pipeline.track()` in `src/core/pipeline.py`, slightly trimmed:

```python
def track(self, path, key=None) -> SessionTrack:
    timer = StageTimer()
    source = _build(FRAME_SOURCES, {**self.cfg["ingestion"], "path": path})
    tracker = self.new_tracker()                                 # fresh state per session
    frame_extractors = [e for e in self.extractors if e.uses_frames]
    for e in frame_extractors:
        e.reset()
    self.scene.reset()
    builder = TrackletBuilder()
    frame_t, scene_rows = [], []

    for frame in source:                                         # one decoded frame at a time
        dets = self.detector.detect(frame)                       # stage 2
        tracks = tracker.update(frame, dets)                     # stage 3
        signals = {}
        for e in frame_extractors:                               # stage 5, pixel part
            for tid, vals in e.frame_signals(frame, tracks).items():
                signals.setdefault(tid, {}).update(vals)
        builder.add(frame.t, tracks, signals)                    # stage 4, accumulate
        frame_t.append(frame.t)
        scene_rows.append(self.scene.frame_features(frame))      # gate input

    return SessionTrack(key, np.asarray(frame_t), frame_wh, builder.build(), scene_rows,
                        timing_audit(np.asarray(frame_t)), timer.report())
```

(Trimmed. The real code also initialises `frame_wh = (1, 1)` and sets it from each image's
shape, reads frames with `next(frames, None)` inside `while True` so decoding can be timed, and
wraps each step in `with timer.stage("..."):` so the manifest records where time went. Pixel-level extractors get one timer each, named after their class
(`bfmu_frame.ResidualFlow`, `bfmu_frame.ScbBehaviour`), because their costs differ a lot.)

Things to notice:

- **One frame in memory at a time.** The source is a generator (Chapter 2). The pass never holds
  the video, only the growing per-track arrays.
- **Only pixel-level extractors run here.** `KalmanGeometry` has `uses_frames = False`, so it is
  skipped; it works entirely from boxes later. `ResidualFlow` and `ScbBehaviour` need pixels, so
  they run now and leave their results in `signals`.
- **`reset()` before each session.** `ResidualFlow` remembers the previous frame to compute
  motion; that memory must not leak from one recording into the next.

## 5.2 Stage 1: ingestion (`src/ingestion/`)

### `DipserIndex`: finding the files

`src/datasets/dipser.py` knows the dataset layout so nothing else has to:

```python
index = DipserIndex.from_config(cfg)
index.sessions()                 # in-scope SessionKeys whose ZIP exists (52 of the possible 54)
index.frames_zip(key)            # .../DIPSEER/group_01/experiment_03/context_cam_02.zip
index.subject_zips(key)          # {"subject_01": path, ...} label archives
index.annotations(key)           # {"subject_01": xyxy array}: our static boxes for this camera
index.experiment_info            # from experiments.tsv: layout, device use, activity
```

### `ZipFrameSource`: decoding frames

`src/ingestion/source.py`. On construction it reads only the ZIP's table of contents, sorts the
image names by their timestamp (not by ZIP order, which is arbitrary), and keeps the timestamps.
Iterating decodes one image at a time:

```python
source = ZipFrameSource(path, stride=1, max_frames=None)
source.timestamps                 # sorted float array, available before decoding anything
for frame in source: ...          # Frame(index, t, image), in time order
source.read(120)                  # random access to one frame (used by the viewers)
```

Two data-quality rules live here:

1. **Zero-byte files are not frames.** One in-scope session
   (`group_03/experiment_03/context_cam_03`) contains 314 empty PNGs, 28% of its frames. They are
   dropped when the ZIP is indexed, so timestamps, the measured fps and the gap audit only count
   real frames.
2. **Undecodable frames are skipped**, not fatal. OpenCV *raises* on some corrupt buffers rather
   than returning `None`, so decoding is wrapped in `try`.

For training only, `AugmentedZipFrameSource` (`src/ingestion/augment.py`) subclasses
`ZipFrameSource` and `torch.utils.data.Dataset`: the same frames and timestamps, each passed through
one fixed augmentation (colour, noise, flip, masks) drawn per session copy.
`Pipeline.track(..., augment=params)` reads a training copy through it, augmenting frames as they
are decoded (never written to disk); Chapter 8, §8.4 explains the whole scheme.

### The timing audit

`src/ingestion/manifest.py` measures what the camera actually delivered:

```python
def timing_audit(t, gap_factor=3.0):
    dt = np.diff(t)
    median = float(np.median(dt))
    gaps = dt > gap_factor * median          # any interval over 3x the typical one
    return {"fps": 1.0 / median, "n_frames": ..., "duration_s": ..., "gaps": [...], ...}
```

The fps comes from the *median* interval, so occasional drops do not distort it. DIPSER cameras
run at about 5 fps, with 9–45 gaps per session of up to 6 s. **Never assume a nominal frame
rate**: velocities are computed from real timestamps, and the measured fps is used only where a
per-frame quantity must be converted to per-second.

## 5.3 Stage 2: person detection (`src/detection/`)

`YoloDetector` wraps an ultralytics YOLO model (YOLO11n by default), restricted to COCO class 0,
"person":

```python
@DETECTORS.register("yolo")
class YoloDetector(PersonDetector):
    def __init__(self, weights="yolo11n.pt", conf=0.3, imgsz=640, device=None, half=True):
        self.model = YOLO(weights)
        self.kw = dict(classes=[0], conf=conf, imgsz=imgsz, device=..., verbose=False,
                       quantize="fp16" if half and device != "cpu" else None)

    def detect(self, frame: Frame) -> Detections:
        boxes = self.model.predict(frame.image, **self.kw)[0].boxes
        return Detections(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())
```

The `conf` threshold is tuned together with the tracker (0.22 currently), because the two
interact: a lower detector threshold gives the tracker more, noisier boxes to work with.

## 5.4 Stage 3: tracking (`src/tracking/tracker.py`)

### What a tracker does

A detector sees each frame independently: it says "there are 14 people here" with no memory of
the previous frame. A tracker links detections over time, so the person at (300, 400) now is
recognised as the same person who was at (302, 398) a moment ago, and keeps the same **track id
(TID)**.

BoT-SORT does this mainly with a **Kalman filter**, a small motion model per track that predicts
where each box should be next. Each frame, it matches new detections to predicted boxes; matched
tracks update, unmatched detections can start new tracks, and tracks unmatched for too long
(`max_lost_frames`) are dropped. Optionally it also compares appearance (ReID embeddings of each
person crop) and compensates for camera motion (CMC). The current config has both off, the
winner of the quick tuning search; which tracker and options become the default is being
decided by three full-dataset experiments (Chapter 8, and `docs/experiments/02_boxmot.md`).

### The wrapper

`BoxmotTracker` adapts BoxMOT to our contract. BoxMOT works on numpy rows
`[x1, y1, x2, y2, conf, class]`:

```python
class BoxmotTracker(Tracker):
    def __init__(self, **options):
        from boxmot import create_tracker
        self.tracker = create_tracker(self.name, **options)

    def update(self, frame, dets):
        rows = np.column_stack([dets.xyxy, dets.conf, np.zeros(len(dets))])
        out = self.tracker.update(rows, frame.image)
        velocities = self._kalman_velocities()
        # BoxMOT numpy output columns: x1, y1, x2, y2, id, conf, cls, det_index
        return [TrackState(int(r[4]), r[:4], float(r[5]), velocities.get(int(r[4])))
                for r in np.asarray(out).reshape(-1, 8)]
```

`_kalman_velocities()` reads each active track's Kalman state vector, whose entries 4–5 are the
centre velocity (vx, vy) in pixels per frame, for both trackers in BoxMOT 26. This is the "Kalman
access" the design asked us to verify (§4). If a future BoxMOT changes its internals, the
velocity becomes `None` and the motion features fall back to finite differences automatically.

Two more methods, `embed()` and `update_with_embeddings()`, let `scripts/tune_tracker.py` compute
ReID embeddings once per session and feed them back on every trial (Chapter 8).

## 5.5 Stage 4 (first half): collecting tracklets

`TrackletBuilder` (`src/tracking/tracklets.py`) accumulates `TrackState`s frame by frame, grouped
by TID, together with any per-frame signals. `build()` turns each TID's rows into one `Tracklet`
with aligned arrays. If an extractor did not report a signal for some frame, that entry is NaN.

These are **raw** tracklets: every track the tracker ever reported, however short. Filtering is
the first step of the feature pass (Chapter 6).

## 5.6 Stage 5, pixel part: frame-level BFMU signals

BFMU (*Base Feature Mining Unit*) computes per-person features. The extractor contract has two
hooks (Chapter 3): `frame_signals` runs here, in the frame pass, for extractors that need pixels;
`window_features` runs later. Two extractors use the first hook.

### Residual optical flow (`src/bfmu/residual_flow.py`)

Optical flow estimates, for every pixel, how far it moved since the previous frame. Inside each
person's box we want *articulated* motion (arms, head, posture) rather than the whole body
sliding, which the Kalman velocity already captures. So:

1. compute Farneback flow between the previous and current frame (on a half-size grey image);
2. inside each box, subtract the box's own mean flow vector (removing whole-body translation);
3. take the magnitude of what remains, divide by the box diagonal and the time between frames
   (units: box diagonals per second, so it is comparable across near and far students);
4. average it over a 2×2 grid of cells (top-left, top-right, bottom-left, bottom-right of the
   box), giving four signals per person per frame: `rf.mag_00` … `rf.mag_11`.

### SCB behaviour detector (`src/bfmu/scb.py`)

SCB is a YOLOv7 model trained by the SCB dataset authors on classroom images, with classes
`hand-raising`, `read`, `write`. On DIPSER its class decisions transfer poorly (`read` fires on
anyone looking down at a desk or a device). So by default (`mode: "soft"`) we do not use its
decisions at all; we pass its raw scores along and let our own models learn what they mean here:

1. run YOLOv7 and keep every candidate box with objectness of at least 0.01 (`obj_floor`, which
   only drops empty anchors) with its per-class probabilities, before any class threshold or
   non-maximum suppression;
2. assign each candidate to the single track it overlaps best, if it lies at least half inside
   that track's box;
3. per track: the highest `objectness × class probability` per class, and the highest objectness
   alone. Zero if no candidate landed on the person.

That gives four signals per person per frame: `scb.hand_raising`, `scb.read`, `scb.write`,
`scb.m0.obj`. The alternative `mode: "flags"` (thresholded detections, 0/1 per class) is kept so
the two can be compared.

Loading the checkpoint safely and the environment quirks around it are covered in Chapter 11.

## 5.7 Scene features for the gate (`src/spu/scene.py`)

The context gate decides which activity is running from the *whole frame*, not from tracks, so
it does not depend on tracking quality. `GlobalStats.frame_features` shrinks each frame to 64×36
and computes cheap global descriptors:

| Feature | Meaning |
|---|---|
| `motion` | mean absolute difference from the previous frame (how much is moving) |
| `sat`, `val` | mean colour saturation and brightness (screens on, lights) |
| `edges` | share of edge pixels (visual clutter: open laptops, papers) |
| `int_00` … `int_22` | mean brightness of each cell of a 3×3 grid (where things are) |

These are stored per frame in `SessionTrack.scene_rows`; the feature pass summarises them per
window.

## 5.8 What the frame pass costs

One fresh extraction of `group_01/experiment_01/context_cam_01` (1,468 frames, 4.9 minutes of
footage) on the GTX 1080, with the current config (BoT-SORT without ReID or CMC, residual flow,
SCB soft mode):

| Stage | Seconds | Per frame | Share |
|---|---|---|---|
| PNG decoding | 62.2 | 42 ms | 22% |
| Person detection (YOLO11n) | 44.9 | 31 ms | 16% |
| Tracking | 4.0 | 3 ms | 1% |
| Residual optical flow | 99.2 | 68 ms | 35% |
| SCB behaviour (YOLOv7, soft) | 67.1 | 46 ms | 24% |
| Scene features | 3.5 | 2 ms | 1% |
| Feature pass (filtering + aggregation) | 1.0 | | <1% |
| **Total (timed stages)** | **281.9 s** | **~190 ms** | |

Two tracker-tuning jobs kept the CPU at about 90% during this run, so the CPU-bound stages
(decoding, flow, tracking) are upper bounds. Even so, the pipeline runs at about real time
(5.2 frames/s processed against a 5 fps camera; six later full-length sessions averaged 185 ms,
range 143–211, `docs/experiments/05_six_session_run.md`): a 45-minute session would take about 43 minutes,
roughly 14× over the design's 3-minute target. Residual flow is the largest single cost: it
computes dense flow over the whole half-resolution frame but only uses it inside student boxes.

ReID tracking (StrongSORT, or BoT-SORT with `use_embeddings`) adds roughly 120–165 ms per frame on
top. `scripts/benchmark.py` measures all of this per tracker variant on several sessions
(Chapter 8). Every extraction run's manifest records these stage times per session.

# Chapter 4 — Contracts and configuration

Chapter 3 showed how components are chosen. This chapter covers the two things every component
agrees on: the **dataclasses** that carry data between stages, and the **configuration file** that
says which components to use and how. It ends with config hashing, which is what makes results
reproducible and caches safe.

## 4.1 The time convention

Every timestamp in the codebase is a `float`: **seconds since local midnight**.

Frame filenames (`10_40_43_442466.png`) and label events (`"10:40:43:047895"`) both use the same
wall clock, so one parser handles both. From `src/core/clock.py`:

```python
_CLOCK = re.compile(r"(\d{1,2})[_:](\d{2})[_:](\d{2})[_:](\d+)")

def parse_clock(text: str) -> float:
    """Return seconds since midnight. The fractional part may have any number of digits."""
    h, mi, s, frac = _CLOCK.fullmatch(text).groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(frac) / 10 ** len(frac)
```

Because both sources share one clock, aligning labels with video needs no conversion (Chapter 7
shows the measured offset is under 2.2 s, and there is a config knob for it). Relative times
("+95.2 s into the session") appear only in outputs and plots.

## 4.2 The dataclasses, in pipeline order

All of these live in `src/core/types.py`. Read them in this order and you have read the
pipeline's data flow.

### `SessionKey`: which recording

```python
@dataclass(frozen=True, order=True)
class SessionKey:
    group: str          # "group_01"
    experiment: str     # "experiment_03"
    camera: str         # "context_cam_02"

    @property
    def exp_id(self) -> str:   # "03": the activity label the gate predicts
        return self.experiment.rsplit("_", 1)[1]
```

Frozen, so it can be a dict key; `order=True`, so lists of them sort predictably.

### `Frame`: one decoded image

```python
@dataclass
class Frame:
    index: int          # position in the session's frame list (0, 1, 2, ...)
    t: float            # seconds since midnight
    image: np.ndarray   # H x W x 3, uint8, BGR (OpenCV's channel order)
```

BGR, not RGB: OpenCV decodes images that way, and every component that consumes images
(YOLO via ultralytics, OpenCV flow, the YOLOv7 wrapper) handles BGR.

### `Detections`: people found in one frame

```python
@dataclass
class Detections:
    xyxy: np.ndarray    # (N, 4): x1, y1, x2, y2 in pixels
    conf: np.ndarray    # (N,)
```

`xyxy` (corner coordinates) is the box format everywhere in our code. The manual annotations use
`x, y, w, h`; `DipserIndex.annotations` converts them on load.

### `TrackState`: one person in one frame, according to the tracker

```python
@dataclass
class TrackState:
    tid: int                              # track id, stable across frames while tracked
    xyxy: np.ndarray                      # (4,)
    conf: float
    velocity: np.ndarray | None = None    # Kalman centre velocity (vx, vy) in px/frame
```

### `Tracklet`: one person's whole time series

```python
@dataclass
class Tracklet:
    tid: int
    t: np.ndarray          # (T,)    timestamps
    xyxy: np.ndarray       # (T, 4)  boxes
    conf: np.ndarray       # (T,)
    velocity: np.ndarray   # (T, 2)  NaN where the tracker did not expose Kalman state
    signals: dict[str, np.ndarray] = field(default_factory=dict)   # per-frame BFMU channels
```

This is the central data structure for per-person work. Every array has one row per frame in
which the person was tracked, all aligned. `signals` holds per-frame values that pixel-based
feature extractors computed during the frame pass, for example `"scb.read"`, the read score in
each frame (Chapter 5).

Two helpers: `tr.slice(t0, t1)` returns the part of the tracklet inside a time range (with all
signals sliced consistently), and `tr.centers` returns box centres.

### `Window`: a slice of time

```python
@dataclass(frozen=True)
class Window:
    t_start: float
    t_end: float        # exclusive
```

### `SessionTrack`: output of the frame pass

```python
@dataclass
class SessionTrack:
    key: SessionKey | None
    frame_t: np.ndarray                  # timestamps of every decoded frame
    frame_wh: tuple[int, int]            # frame width, height
    tracklets: list[Tracklet]            # RAW tracker output, unfiltered
    scene_rows: list[dict[str, float]]   # per-frame scene features, aligned with frame_t
    timing: dict                         # fps, gaps, duration (see ingestion)
    stage_seconds: dict[str, float]      # how long each stage took
```

Two details here are deliberate:

- `tracklets` are **raw**. Filtering short or broken tracks happens in the feature pass, so its
  thresholds can change without re-running tracking.
- `frame_t` lists every frame that was actually decoded. The tracklet filter needs it to tell
  "the tracker lost this student" apart from "the camera dropped frames" (Chapter 6).

### `SessionFeatures`: output of the feature pass

```python
@dataclass
class SessionFeatures:
    key: SessionKey | None
    windows: list[Window]
    vec: list[dict[str, float]]      # Vec, one dict per window
    scene: list[dict[str, float]]    # scene (gate) features, one dict per window
    manifest: dict
```

`vec[i]` and `scene[i]` describe `windows[i]`. Both are flat `{name: value}` dicts rather than
arrays, so features are always identified **by name**. `to_frame()` in `pipeline.py` turns this
into a pandas table with prefixed columns (`vec.kg.speed.mean`, `scene.motion.mean`), which is
what gets saved to CSV and what the models read.

### Why dicts of named features instead of arrays?

Because feature sets change. Adding `box_jitter` adds columns; turning CFMU on adds columns. If
features were positional arrays, every model trained before the change would silently read the
wrong columns. With names, the training script selects columns by name, and the saved model
bundle records exactly which names it was trained on (`vec_cols`, `gate_cols`). At inference,
`df.reindex(columns=models["vec_cols"])` puts columns in the trained order and fills any missing
one with NaN.

## 4.3 The configuration file

`configs/deliverable3.json` holds every tunable value. Here are its sections, in pipeline order,
with what reads each:

| Section | Read by | Controls |
|---|---|---|
| `seed` | models, tuning | random seeds for reproducibility |
| `dataset` | `DipserIndex` | data root; which groups, experiments, cameras are in scope |
| `classes` | gate, mixture | the context labels, `["01", ..., "06"]` |
| `ingestion` | frame source | `stride` (keep every n-th frame), `max_frames` (for quick tests) |
| `detector` | YOLO wrapper | weights file, confidence threshold, input size, half precision |
| `tracker` | tracker wrapper | `name` plus BoxMOT `options` (tuned values) |
| `tracklets` | feature pass | `min_visible_s`, `max_gap_s` for the tracklet filter |
| `association` | evaluation | IoU floor and match share for SID–TID association |
| `bfmu` | extractors | **list** of extractor entries; order defines column order |
| `cfmu` | aggregator | cluster extractor (`null` in Deliverable III) |
| `scene` | gate features | scene feature extractor |
| `window` | feature pass | `length_s`, `overlap` |
| `aggregation` | aggregator | thresholds for "fraction of students above", heatmap grid, coverage |
| `gate`, `gate_uses_vec`, `context_mode` | training, inference | gate model and its inputs |
| `expert`, `hand_weighted`, `mixture` | training, inference | expert model, baseline weights, mixing |
| `labels` | label pipeline | `tau_att`, `tau_share`, minimum labelled students, offset |
| `tuning` | tuning scripts only | search profiles and spaces |
| `benchmark` | `scripts/benchmark.py` only | sessions and tracker variants to time |
| `output_dir` | scripts | where caches and results go |

### Overriding values per run

Every script accepts `--set` with dotted keys and JSON values, so you never need to copy the
config file to try a variation:

```
python -m scripts.run_pipeline extract --set window.length_s=10 ingestion.max_frames=200
python -m scripts.train_models --set labels.tau_att=3 labels.tau_share=0.6
```

This is implemented in `load_config` and `set_dotted` (`src/core/config.py`): it walks the dotted
path, creating intermediate dicts if needed, and assigns the value. Values are parsed as JSON, so
`=10` is a number, `=true` a boolean, `='"text"'` a string, and `='["a","b"]'` a list. On Windows
PowerShell, quoting JSON strings needs care; numbers and booleans need none.

## 4.4 Config hashing

Every result must say exactly which configuration produced it (§11 reporting rule). Rather than
copying the whole config into every table, we compute a **fingerprint**: the SHA-256 hash of the
config written as canonical JSON.

```python
HASH_EXCLUDED_KEYS = ("output_dir", "tuning", "benchmark")

def config_hash(cfg: dict) -> str:
    clean = copy.deepcopy(cfg)
    for k in HASH_EXCLUDED_KEYS:
        clean.pop(k, None)
    canonical = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

"Canonical" means the same config always produces the same text: keys sorted, no whitespace
variation. So the same settings always give the same hash, and any change, however small, gives
a different one.

Three sections are excluded because they do not change what a run produces: `output_dir` (where
results go), and `tuning` and `benchmark` (only their own scripts read them).

### Three hashes, three purposes

| Hash | Defined in | Covers | Used for |
|---|---|---|---|
| `config_hash(cfg)` | `core/config.py` | everything except output_dir, tuning, benchmark | results tables, model bundles, manifests |
| `tracking_hash(cfg)` | `core/pipeline.py` | ingestion, detector, tracker, scene, pixel-level BFMU | the frame-pass cache folder |
| `feature_hash(cfg)` | `core/pipeline.py` | tracking hash + tracklets, all BFMU, CFMU, window, aggregation | the feature-pass cache folder |

```python
def tracking_hash(cfg: dict) -> str:
    return config_hash({k: cfg[k] for k in ("ingestion", "detector", "tracker", "scene")}
                       | {"bfmu_frame": _frame_level_bfmu(cfg)})

def feature_hash(cfg: dict) -> str:
    return config_hash({"tracking": tracking_hash(cfg)}
                       | {k: cfg[k] for k in ("tracklets", "bfmu", "cfmu", "window", "aggregation")})
```

`_frame_level_bfmu` keeps only the BFMU entries whose class has `uses_frames = True`. It asks the
registry (`FEATURE_EXTRACTORS.get(name).uses_frames`), which is why these hash functions live in
`pipeline.py` rather than `config.py`: they need to know about extractors.

The consequence, which the tests check (`test_cache_keys_split_frame_pass_from_feature_pass`):

| You change | tracking hash | feature hash | full hash | Cost to recompute |
|---|---|---|---|---|
| `labels.tau_att` | same | same | **new** | retrain only |
| `window.length_s` | same | **new** | **new** | feature pass: seconds |
| `kalman_geometry.still_threshold` | same | **new** | **new** | feature pass: seconds |
| `residual_flow.grid` | **new** | **new** | **new** | frame pass: minutes per session |
| `tracker.name` | **new** | **new** | **new** | frame pass: minutes per session |

The feature hash *includes* the tracking hash, so features can never be computed from tracks made
under different settings.

### The reporting rule, mechanically

Each results JSON is named after its full config hash (`outputs/results/<hash>.json`) and
contains the whole config. Two numbers can be compared only if they come from the same hash, or
from hashes that differ only in the setting being compared.

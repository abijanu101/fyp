# BoxMOT: detection and tracking

This document records how person detection and multi-object tracking were brought up on DIPSER
with BoxMOT, how the tracker was evaluated and tuned, and what was found.

## 1. Goal

Design §4: one shared person detector feeding two trackers, StrongSORT and BoT-SORT, compared
head-to-head on ID switches, fragmentation and downstream usefulness. The winner becomes the
default; the other stays selectable. Tracks feed the per-student features (BFMU), so tracking
errors propagate into every downstream number.

## 2. Phase 1: validation notebook (`notebooks/dipser_tracking.ipynb`)

The first pass ran BoxMOT's own detector and tracker on one camera, end to end, to check that the
library works on DIPSER at all:

- detector: BoxMOT's `yolov10n` wrapper, confidence 0.30, person class only;
- tracker: BoT-SORT with `track_high_thresh=0.3`, `track_low_thresh=0.10`, `new_track_thresh=0.15`,
  otherwise BoxMOT defaults (including ReID with `osnet_x0_25_msmt17`);
- 60 frames of one session; detections and tracks visualised side by side;
- tracks linked to subjects by nearest neighbour on mean box position.

Findings:

1. **BoxMOT runs on DIPSER frames** with the canonical `Frame` input and produces sensible tracks.
2. **The tracker parameters clearly needed tuning.** Defaults and hand-picked values produced
   fragmented tracks and missed students in the back rows.
3. **The association had no distance threshold**, so walking TAs were assigned to seated students
   (see the DIPSER document, §3).

## 3. Phase 2: integration into the pipeline

### Detector

The pipeline uses one shared detector, ultralytics **YOLO11n** (`models/yolo11n.pt`), restricted
to the person class, at 640 px input and FP16 (`src/detection/detector.py`). Using ultralytics
directly rather than BoxMOT's detector wrapper keeps the detector independent of the tracking
library, so both trackers see identical detections.

### Is the confidence threshold dropping students? (verified: no)

`detector.conf` (0.22) is the only confidence filter before tracking: BoxMOT's `det_thresh` is
stored but never read by BoT-SORT or StrongSORT (BoxMOT 26: only `trackers/common/base.py:188`
assigns it), so it was removed from the config and from tuning. The detector threshold is therefore
the setting to check, and it can be checked without any tracker.

`scripts/scan_detector.py` takes the detections cached for tuning (every frame of all 52 sessions,
at a 0.05 floor), keeps those at or above each threshold, and matches them per frame one-to-one to
the manual student boxes (Hungarian, IoU ≥ 0.3, the tracker-scoring rule). Per student, recall is
the share of frames in which they are matched. The floor is the reference: a student missed even
at 0.05 is not lost to the threshold.

| Threshold | Mean recall | 10th percentile | Lowest | Never detected | Below 50% | Lost > 10 pts vs floor | Lost > 25 pts |
|---|---|---|---|---|---|---|---|
| 0.05 (floor) | 0.999 | 0.995 | 0.939 | 0 | 0 | 0 | 0 |
| 0.15 | 0.997 | 0.990 | 0.926 | 0 | 0 | 0 | 0 |
| 0.20 | 0.995 | 0.988 | 0.770 | 0 | 0 | 2 | 0 |
| **0.22 (config)** | **0.994** | **0.987** | **0.699** | **0** | **0** | **3** | **1** |
| 0.25 | 0.992 | 0.982 | 0.594 | 0 | 0 | 7 | 2 |
| 0.30 | 0.986 | 0.969 | 0.442 | 0 | 2 | 21 | 3 |
| 0.40 | 0.952 | 0.882 | 0.090 | 0 | 15 | 68 | 38 |
| 0.50 | 0.873 | 0.568 | 0.002 | 0 | 56 | 170 | 105 |
| 0.60 | 0.727 | 0.118 | 0.000 | 6 | 142 | 325 | 227 |

621 student-sessions (each annotated student in each of the 52 sessions); full table in
`outputs/tuning/detector_scan.csv`, per student in `detector_scan_students.csv`.

**At 0.22 no student is dropped**: every one is detected, none in fewer than half the frames, and
the mean is within 0.005 of the floor. Three students lose more than 10 percentage points against
the floor, the largest `subject_11` in `group_02/experiment_06/context_cam_01` (0.70 against 1.0
at the floor); these are partial losses of a hard-to-see student, not drops. Below 0.25 the curve
is flat, so the threshold sits on the plateau and a finer search (e.g. bisection) would not change
the conclusion. Above 0.30 losses grow quickly.

The table also quantifies a hazard: StrongSORT's own `min_conf` filter defaults to **0.6**, where
6 students are never detected and 142 are detected in under half their frames. The full StrongSORT
experiment therefore starts from the quick-tuned values (`min_conf` 0.2153; §7), not BoxMOT's
defaults.

### Tracker wrapper

`src/tracking/tracker.py` wraps BoxMOT 26.0 through its numpy interface: detections go in as rows
`[x1, y1, x2, y2, conf, class]`, and tracks come out as `[x1, y1, x2, y2, id, conf, class,
det_index]`. Confirmed on real runs with both trackers.

### Kalman state access (verified)

The design asked to verify whether BoxMOT exposes the Kalman state before building the motion
features on it (§4, §13). In BoxMOT 26.0, both trackers keep `tracker.active_tracks`; each track
has `.id` and a Kalman `.mean` vector. Indices 4–5 are the centre velocity (vx, vy) in pixels per
frame for both BoT-SORT's XYWH filter and StrongSORT's XYAH filter. On a real 150-frame run,
**100% of track observations** carried a finite Kalman velocity with either tracker. If a future
version changes this, velocities become missing and the motion features fall back to finite
differences.

## 4. How tracking is evaluated

### Ground truth

The manual static boxes (`abi_labels`, DIPSER document §2): one box per student per camera.
Assumption: every annotated student is present in every frame of the evaluated span. A student
who briefly leaves lowers coverage equally for every configuration, so comparisons stay fair even
though absolute numbers are a lower bound.

### Matching

Per frame, Hungarian assignment of track boxes to student boxes on IoU, with a floor of 0.3, as in
SID–TID association.

### Metrics (`src/evaluation/tracking.py`)

| Metric | Definition |
|---|---|
| **IDF1** (objective) | identity F1 with a one-to-one track↔student mapping chosen to maximise matched frames: a student's time split across two track ids only gets credit for the larger part |
| coverage | share of (student, frame) pairs matched by some track |
| ID switches per subject | changes of the matched track id along each student's timeline |
| track ids per subject | distinct track ids matched to each student (fragmentation) |
| subjects found | share of annotated students matched at least once |

Track boxes on people outside the annotation (teacher, TAs) count as false positives in IDF1,
equally for every configuration.

An early version mapped tracks to students many-to-one (majority vote) when computing IDF1, which
gave a fragmented student full credit for both fragments. A unit test caught it, and IDF1 now uses
the one-to-one assignment.

## 5. Tuning method

`scripts/tune_tracker.py`; details in the contributor's guide, Chapter 8.

- **Detections cached once** per session at a confidence floor of 0.05. Each trial filters them at
  its own `det_conf`, so detector confidence is tuned jointly with the tracker at no extra cost,
  and only the tracker re-runs.
- **Seeded random search**; trial 0 is the current configuration. Search space for BoT-SORT:
  `det_conf`, `track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `max_lost_frames`
  (log scale), `match_thresh`, `proximity_thresh`, `appearance_thresh`, `use_embeddings` (ReID
  on/off), `use_cmc` (camera-motion compensation on/off).
- **Profiles**:
  - `quick`: 3 sessions, one per group (`group_01/experiment_03/context_cam_02`,
    `group_02/experiment_04/context_cam_03`, `group_03/experiment_01/context_cam_01`), first 300
    frames (60 s) each, 25 trials; ReID and CMC are searched;
  - `full`: BoT-SORT, all 52 sessions, all frames, 20 trials, ReID fixed **off**, CMC fixed **on**;
  - `full_reid_cmc`: BoT-SORT, all 52 sessions, 20 trials, ReID and CMC fixed **on**;
  - `full_strongsort`: StrongSORT, all 52 sessions, 20 trials, CMC fixed **on**; trial 0 runs the
    quick-tuned StrongSORT values (`base_options`), not BoxMOT's defaults (§7).

  CMC is on in every full-dataset trial, so every trial decodes real frames and no result depends
  on the blank-frame shortcut. The only remaining shortcut is the ReID embedding cache, verified
  with decoded frames and CMC on (StrongSORT 42 vs 42 tracks, BoT-SORT with ReID 16 vs 16).
  An earlier CMC-off run of `full` was stopped at 16 of 40 trials
  (`outputs/tuning/old/partial_botsort_full_cmc_off_16trials.log`).
- **Blank-frame shortcut**: BoxMOT's tracker takes the image as input, but with ReID and CMC off
  BoT-SORT never reads it. Trials therefore pass a blank frame instead of decoding PNGs, cutting a
  full-dataset trial from about 35 minutes to about 2. **Verified**: the check produced identical
  tracks (10 tracks, all ids, timestamps and boxes) from blank and decoded frames.
- **ReID embedding cache**: ReID embeddings depend only on the frame and the detection box, never
  on tracker settings. Every floor-level detection is therefore embedded once, with the trackers'
  own default ReID model (`osnet-x0-25-msmt17`, shared by BoT-SORT and StrongSORT) through
  BoxMOT's own embedding code, and stored at full float32 precision (about 55 MB per session on
average, 25–85 MB).
  Trials pass the cached embeddings to the tracker, which then runs no ReID model. Frames are still
  decoded whenever a trial turns camera-motion compensation (CMC) on: CMC estimates camera motion
  from consecutive frames while masking the trial's own detections, so it cannot be cached.
  `--check` compares this fast path with the tracker's own path (decoded frames, its own ReID) on
  the profile's first session and must report identical tracks before a full run starts.

## 6. Results

### 6.1 BoT-SORT, quick profile

| | IDF1 | Coverage | ID switches / subject | Track ids / subject |
|---|---|---|---|---|
| BoxMOT defaults + detector conf 0.30 (trial 0) | 0.908 | 0.88 | 0.18 | 1.11 |
| **Best of 25 trials** | **0.950** | 0.92 | 0.14 | 1.04 |

Best parameters (now in `configs/deliverable3.json`):

```
detector.conf 0.22      track_high_thresh 0.25   track_low_thresh 0.21   new_track_thresh 0.67
match_thresh 0.80       max_lost_frames 17       proximity_thresh 0.48   appearance_thresh 0.55
use_embeddings false    use_cmc false
```

97% of annotated students were found. Trials 6, 16 and 22 scored within 0.013 of the best, so
the optimum is better described as a cluster of similar configurations than a single point.

### 6.2 Does ReID help? (quick profile, same search)

ReID was a search parameter, so trials with and without it ran on identical footage and
detections:

| | ReID off (13 trials) | ReID on (11 trials) |
|---|---|---|
| Mean IDF1 | 0.877 | 0.918 |
| Best IDF1 | **0.950** | 0.940 |
| Mean ID switches per subject | 0.68 | 0.40 |
| Tracking cost per frame | ~2 ms | ~120–165 ms |

ReID makes the tracker **more forgiving of poor settings**: averaged over random configurations it
switches ids less and scores higher. But a **well-tuned motion-only tracker matches the best ReID
result** (0.950 vs 0.940, within noise) at about 60× less compute. In this setting (fixed cameras,
students seated in fixed places) position and motion are enough to keep identities apart.

### 6.3 StrongSORT, quick profile

StrongSORT relies on ReID throughout. Tuned on the same three sessions and frames, from BoxMOT's
defaults:

| | IDF1 | Group 01 / 02 / 03 |
|---|---|---|
| BoxMOT defaults (trial 0) | 0.851 | 0.832 / 0.865 / 0.858 |
| **Best of 25 trials** | **0.932** | 0.994 / 0.887 / 0.915 |

Best parameters: `det_conf 0.23, max_age 15, max_cos_dist 0.33, max_iou_dist 0.86, min_conf 0.22,
n_init 3, use_cmc true`. Group 02 was the weakest room. Unlike BoT-SORT's quick winner, the best
StrongSORT trial had CMC **on**.

### 6.4 BoT-SORT, full profile (all 52 sessions)

*In progress.* The configuration from §6.1 scores **IDF1 0.888** over all 52 sessions (per group:
0.857 / 0.918 / 0.890; coverage 0.90; 0.74 ID switches and 1.27 track ids per subject). That is
lower than its 0.950 on the quick profile's 60-second clips: full sessions are five minutes long,
with more occlusions, more people moving and more chances to switch ids. The 20-trial search is
running; interim, as of 15 November 2026: 11 of 20 trials complete, best so far trial 6 at IDF1
0.905 (mean over the 52 sessions, not yet a nested estimate). Final results go here.

## 7. Tracker-level experiments on the full dataset

An earlier draft dropped StrongSORT on compute grounds after the quick comparison (§6.2, §6.3). That
decision is withdrawn until the full-dataset comparison exists. Three experiments, all on the 52
in-scope sessions, scored identically:

| # | Tracker | ReID | CMC | Profile | Status |
|---|---|---|---|---|---|
| 1 | BoT-SORT | off | on | `full` | running: 11 of 20 trials, best IDF1 so far 0.905 (trial 0: 0.888) |
| 2 | StrongSORT | on (always) | on | `full_strongsort` | 6 of 20 trials done, best so far 0.853; to be restarted with the quick-tuned baseline (below), keeping trials 1–5 |
| 3 | BoT-SORT | on | on | `full_reid_cmc` | check passed; queued |

StrongSORT's old trial 0 (BoxMOT defaults) scored 0.715; that number reflects `min_conf` 0.6
dropping students (§3), not StrongSORT's tracking.

With CMC held on everywhere, experiments 1 and 3 differ only in ReID (same tracker), and 2 and 3
compare the two trackers with ReID.

**Every experiment starts from its best known settings.** Trial 0 of each search is a baseline:
for BoT-SORT the quick-tuned configuration (§6.1); for StrongSORT, after a restart on 15 November
2026, the quick-tuned StrongSORT values (§6.3), given to the profile as `base_options`. Before
that, StrongSORT's trial 0 ran BoxMOT's defaults, including `min_conf` 0.6, which discards most
weak detections (§3). Trials 1 and later sample every StrongSORT option and were unaffected, so
their logged results were kept; only trial 0 re-ran.

The quick-profile evidence so far: tuned motion-only BoT-SORT 0.950, best ReID-on BoT-SORT 0.940,
tuned StrongSORT 0.932 (same three 60-second clips). ReID costs about 120–165 ms per frame against
about 2 ms for motion-only tracking; that cost stands regardless of accuracy and must be weighed
against the 3-minute compute target. Results and the decision go here when the runs complete.

## 8. Engineering issues encountered

Recorded because each cost a failed multi-hour run:

| Issue | Symptom | Fix |
|---|---|---|
| A new YOLO detector was built per session in the tuning script | memory growth | one shared detector |
| Indexing a compressed `.npz` inside a per-frame loop (every access decompresses the whole array, and slices kept each copy alive) | `MemoryError` after ~18 sessions | read each array once, then slice |
| 314 empty PNGs in one session; OpenCV raises on an empty buffer instead of returning nothing | crash at session 43 | zero-byte files dropped when the ZIP is indexed |

## 9. Open

- Fill in §6.4 and §7 when the runs complete, choose the default tracker, and apply the winner to
  the config.
- Check whether the winner helps all three groups or only some (per-group columns in
  `outputs/tuning/tracker_botsort_full.csv`).
- The downstream criterion in §4 (context-prediction accuracy per tracker) has not been measured;
  it matters if the tracking metrics of two configurations come out close.

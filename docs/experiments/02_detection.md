# Person detection

This document records the person detector every later stage depends on and how its confidence
threshold was set from data. Tracking, which consumes these detections, is experiment 03
([03_tracking.md](03_tracking.md)).

## 1. Goal

Design §4: one shared person detector feeding the tracker. A student the detector misses gets no
track and therefore no features, so the detector is a single point of failure: its confidence
threshold must not drop students, on clean or on degraded footage.

**Experiments in this document** (all on the 52 in-scope sessions, every frame):

| | Experiment | Status | Headline |
|---|---|---|---|
| D1 | Full grid scan of the confidence threshold, clean data (§3) | done | 0.22 drops no student; mean recall 0.994 |
| D2 | Binary search for the optimal threshold, clean data (§3) | done | 0.166 keeps every student within 10 points of the floor |
| D3 | Binary search under on-the-fly augmentation (§3) | done | **0.110** keeps every student within 10 points of the floor; 0.22 fails under augmentation |
| D4 | The same scan for the YOLO pose model as the detector, stricter criterion (1 point) | pending | experiment 09 §1 |

## 2. The detector

The pipeline uses one shared detector, ultralytics **YOLO11n** (`models/yolo11n.pt`), restricted
to the person class, at 640 px input and FP16 (`src/detection/detector.py`). Using ultralytics
directly rather than BoxMOT's detector wrapper keeps the detector independent of the tracking
library, so both trackers see identical detections.

## 3. Is the confidence threshold dropping students? D1–D3

`detector.conf` (0.22 before D3) is the only confidence filter before tracking: BoxMOT's `det_thresh` is
stored but never read by BoT-SORT or StrongSORT (BoxMOT 26: only `trackers/common/base.py:188`
assigns it), so it was removed from the config and from tuning. The detector threshold is therefore
the setting to check, and it can be checked without any tracker.

**Method, shared by D1–D3** (`scripts/tuning/scan_detector.py`). Detections at a 0.05 confidence floor
are kept for every frame. For a threshold, each frame's boxes at or above it are matched one-to-one
to the manual student boxes (Hungarian on IoU ≥ 0.3, the tracker-scoring rule). Per student,
**recall** is the share of frames in which they are matched; the floor is the reference, since a
student missed even at 0.05 is not lost to the threshold. Columns:

- *never detected*, *below 50%*: students with recall 0, and below 0.5;
- *largest drop*: the largest loss of one student's recall against the floor;
- *lost > 10 / 25 pts*: students losing more than 10 / 25 points against the floor;
- *unmatched boxes*: kept boxes that match no annotated student (teachers, TAs, false positives),
  the price of a lower threshold.

### D1. Full grid scan, clean data

All 52 sessions, every frame, 621 student-sessions (each annotated student in each session), 12
thresholds. `python -m scripts.tuning.scan_detector --grid`; `outputs/tuning/detector_scan.csv`, per
student in `detector_scan_students.csv`.

| Threshold | Mean recall | 10th pct | Lowest | Never detected | Below 50% | Largest drop | Lost > 10 pts | Lost > 25 pts | Unmatched boxes |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 (floor) | 0.999 | 0.995 | 0.939 | 0 | 0 | 0.000 | 0 | 0 | 41.1% |
| 0.10 | 0.998 | 0.992 | 0.931 | 0 | 0 | 0.038 | 0 | 0 | 29.9% |
| 0.15 | 0.997 | 0.990 | 0.926 | 0 | 0 | 0.067 | 0 | 0 | 24.3% |
| 0.18 | 0.996 | 0.989 | 0.845 | 0 | 0 | 0.155 | 2 | 0 | 21.9% |
| 0.20 | 0.995 | 0.988 | 0.770 | 0 | 0 | 0.230 | 2 | 0 | 20.4% |
| **0.22 (config)** | **0.994** | **0.987** | **0.699** | **0** | **0** | **0.301** | **3** | **1** | **19.1%** |
| 0.25 | 0.992 | 0.982 | 0.594 | 0 | 0 | 0.406 | 7 | 2 | 17.3% |
| 0.30 | 0.986 | 0.969 | 0.442 | 0 | 2 | 0.558 | 21 | 3 | 14.5% |
| 0.35 | 0.973 | 0.942 | 0.215 | 0 | 4 | 0.785 | 43 | 20 | 11.9% |
| 0.40 | 0.952 | 0.882 | 0.090 | 0 | 15 | 0.903 | 68 | 38 | 9.3% |
| 0.50 | 0.873 | 0.568 | 0.002 | 0 | 56 | 0.996 | 170 | 105 | 5.5% |
| 0.60 | 0.727 | 0.118 | 0.000 | 6 | 142 | 1.000 | 325 | 227 | 3.3% |

**At 0.22 no student is dropped**: every one is detected, none in fewer than half the frames, and
the mean is within 0.005 of the floor. Three students lose more than 10 points against the floor,
the largest `subject_11` in `group_02/experiment_06/context_cam_01` (0.70 against 1.0 at the floor):
partial losses of a hard-to-see student, not drops. Up to about 0.25 the mean barely moves; above
0.30 students start to be lost quickly. The last column is the other side of the trade-off: at the
floor 41% of boxes are not students, at 0.22 19%.

The table also quantifies a hazard: StrongSORT's own `min_conf` filter defaults to **0.6**, where
6 students are never detected and 142 are detected in under half their frames. The full StrongSORT
experiment therefore starts from the quick-tuned values (`min_conf` 0.2153; T1), not BoxMOT's
defaults.

### D2. Binary search for the optimal threshold, clean data

Same 52 sessions, every frame, 621 student-sessions. `python -m scripts.tuning.scan_detector` (the
default mode); `outputs/tuning/detector_search.{csv,json}`. Criterion for "keeps every student":
none undetected, none below 50%, none losing more than 10 points against the floor. Recall only
falls as the threshold rises, so the criterion is monotone and bisection over [0.05, 0.95] finds
the highest threshold that meets it; the search stops at a width of 0.005. Every probe:

| Threshold | Keeps every student | Mean recall | Lowest | Largest drop | Lost > 10 pts | Unmatched boxes |
|---|---|---|---|---|---|---|
| 0.0500 (floor) | yes | 0.999 | 0.939 | 0.000 | 0 | 41.1% |
| 0.1625 | yes | 0.997 | 0.914 | 0.086 | 0 | 23.2% |
| **0.1660 (optimal)** | **yes** | **0.997** | **0.903** | **0.097** | **0** | **23.0%** |
| 0.1695 | no | 0.997 | 0.886 | 0.114 | 1 | 22.7% |
| 0.1765 | no | 0.996 | 0.856 | 0.144 | 1 | 22.1% |
| 0.1906 | no | 0.996 | 0.808 | 0.192 | 2 | 21.1% |
| 0.2188 | no | 0.995 | 0.701 | 0.299 | 3 | 19.1% |
| 0.2750 | no | 0.990 | 0.533 | 0.467 | 12 | 15.9% |
| 0.5000 | no | 0.874 | 0.002 | 0.996 | 170 | 5.5% |

The highest threshold that keeps every student within 10 points of the floor is **0.166**. The
configured 0.22 fails this stricter criterion because of the single student named in D1; every
other student is within 10 points. Lowering `detector.conf` from 0.22 to 0.166 would keep that
student but add 4 points of boxes that are not students, which the tracker then has to handle. The
tracker searches choose `det_conf` on IDF1, which weighs both sides; the detector search isolates
the recall side. Whether to lower `detector.conf` is therefore settled by the tracker experiments.

### D3. Binary search under augmentation (done, 16 November 2026)

The detector is a single point of failure, so it is also tested on degraded footage: all 52
sessions, **every frame augmented on the fly 5 times**, each variant with its own independently
drawn parameters from the config's `augmentation` (brightness ±35, contrast ±15%, saturation ±20%,
hue ±3°, gamma 1/1.15–1.15, Gaussian or speckle noise up to 0.35, horizontal flip with p 0.5, grey
occluders with p 0.5), each detected at the floor, student boxes mirrored for flipped variants.
About 375,000 images (~75,000 frames × 5); only the detections are kept, per session, for resuming.
The floor reference is the augmented variants' own floor detections. Run in the user's terminal
(one progress line per frame), roughly 8 hours with 4 workers; `--stride 2` halves it:
`python -m scripts.tuning.scan_detector --augment --workers 4`;
`outputs/tuning/detector_search_augmented.{csv,json}`.

| | Optimal threshold | Configured 0.22 keeps every student | Mean recall at 0.22 | Lowest at 0.22 | Unmatched boxes at 0.22 |
|---|---|---|---|---|---|
| D2, clean | 0.166 | no (one student) | 0.994 | 0.699 | 19.1% |
| D3, augmented | **0.1098** | **no (22 students lose more than 10 points)** | 0.978 | 0.590 | 14.5% |

343,135 augmented images (every frame of the 52 sessions, 5 variants each), 621 student-sessions.
Under augmentation the detector's confidences drop, so 0.22, which loses one student on clean data,
loses 22 by more than 10 recall points (4 by more than 25; the worst drops 39 points). Per threshold:

| Threshold | Mean recall | 10th percentile | Lowest | Lose > 10 points vs floor | Unmatched boxes |
|---|---|---|---|---|---|
| 0.05 (floor) | 0.993 | 0.988 | 0.943 | 0 | 37.9% |
| **0.1098 (optimal)** | **0.990** | **0.983** | **0.877** | **0** | **24.9%** |
| 0.22 (configured until D3) | 0.978 | 0.954 | 0.590 | 22 | 14.5% |

**Decision: `detector.conf` = 0.1098 for the rest of FYP-1** (set by `configs/derived.py` from
`outputs/tuning/detector_search_augmented.json`). It is the highest threshold that keeps every
student, on degraded footage, within 10 points of the floor; the price is more boxes that are not
students (24.9% against 14.5%), which the tracker's own thresholds handle (T2 tunes them with this
threshold fixed). The clean-data optimum (0.166, D2) is above it, so 0.1098 also keeps every student
on clean footage.

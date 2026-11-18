# Person detection

This document records the person detector every later stage depends on and how its confidence
threshold was set from data. Tracking, which consumes these detections, is experiment 03
([03_tracking.md](03_tracking.md)).

## 1. Goal

Design §4: one shared person detector feeding the tracker. A student the detector misses gets no
track and therefore no features, so the detector is a single point of failure: its confidence
threshold must not drop students, on clean or on degraded footage.

**Experiments in this document** (all on the 52 in-scope sessions, every frame), numbered like the
tracker experiments in experiment 03: 1 on clean data, 2 on augmented data, 3 with experiment 09's
pose model:

| | Experiment | Status | Headline |
|---|---|---|---|
| D1 | Grid scan, then binary search for the optimal threshold, clean data (§3) | done | 0.166 keeps every student within 10 points of the floor; 0.22 drops none but loses 30 points of one |
| D2 | Binary search under on-the-fly augmentation (§3) | done | **0.110** keeps every student within 10 points of the floor; 0.22 fails under augmentation |
| D3 | The same augmented scan for the YOLO pose model as the detector, stricter criterion (1 point; experiment 09 §1) | done | **0.0535**, at the floor: mean recall barely moves above it, but the hardest student loses 7.5 points by 0.11; the price is 41% non-student boxes |

## 2. The detector

The pipeline uses one shared detector, ultralytics **YOLO11n** (`models/yolo11n.pt`), restricted
to the person class, at 640 px input and FP16 (`src/detection/detector.py`). Using ultralytics
directly rather than BoxMOT's detector wrapper keeps the detector independent of the tracking
library, so both trackers see identical detections.

Experiment 09 replaces it with the YOLO11s pose model at 1280 px (`models/yolo11s-pose.pt`), which
returns each person's keypoints with the box; its threshold is D3's (09 §1).

## 3. Is the confidence threshold dropping students? D1–D3

`detector.conf` (0.22 before D2) is the only confidence filter on the detector side: BoxMOT's `det_thresh` is
stored but never read by BoT-SORT or StrongSORT (BoxMOT 26: only `trackers/common/base.py:188`
assigns it), so it was removed from the config and from tuning. The detector threshold is therefore
the setting to check, and it can be checked without any tracker.

**Method, shared by D1–D3** (`scripts/tuning/scan_detector.py`; step by step in Appendix A). Detections at a 0.05 confidence floor
are kept for every frame. For a threshold, each frame's boxes at or above it are matched one-to-one
to the manual student boxes (Hungarian on IoU ≥ 0.3, the tracker-scoring rule). Per student,
**recall** is the share of frames in which they are matched; the floor is the reference, since a
student missed even at 0.05 is not lost to the threshold. Columns:

- *never detected*, *below 50%*: students with recall 0, and below 0.5;
- *largest drop*: the largest loss of one student's recall against the floor;
- *lost > 10 / 25 pts*: students losing more than 10 / 25 points against the floor;
- *unmatched boxes*: kept boxes that match no annotated student (teachers, TAs, false positives),
  the price of a lower threshold.

### D1. Grid scan, then binary search for the optimal threshold, clean data

All 52 sessions, every frame, 621 student-sessions (each annotated student in each session).

**First, a grid scan** of 12 thresholds (`python -m scripts.tuning.scan_detector --grid`;
`outputs/tuning/detector_scan.csv`, per student in `detector_scan_students.csv`). It shows where
the answer lies and what each side of the trade-off costs:

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

- **The configured 0.22 drops no student**: every one is detected, none in fewer than half the
  frames, and the mean is within 0.005 of the floor. But three students lose more than 10 points,
  the worst `subject_11` in `group_02/experiment_06/context_cam_01` (0.70 against 1.0 at the floor).
- Up to about 0.25 the mean barely moves; above 0.30 students start to be lost quickly. The answer
  lies between 0.15 and 0.18, where the first student loses more than 10 points.
- The other side of the trade-off: at the floor 41% of boxes are not students, at 0.22 19%.
- A hazard for the tracker experiments: StrongSORT's own `min_conf` filter defaults to **0.6**,
  where 6 students are never detected and 142 are detected in under half their frames. StrongSORT
  therefore starts from the quick-tuned values (`min_conf` 0.2153; experiment 03, T1), not BoxMOT's
  defaults.

**Then the binary search** (`python -m scripts.tuning.scan_detector`, the default mode;
`outputs/tuning/detector_search.{csv,json}`) pins the threshold down. Criterion for "keeps every
student": none undetected, none below 50%, none losing more than 10 points against the floor.
Recall only falls as the threshold rises, so the criterion is monotone and bisection over
[0.05, 0.95] finds the highest threshold that meets it, stopping at a width of 0.005. The search
does not read the grid; the grid shows the answer is in a narrow, low range, which the bisection
reaches in 8 probes. Every probe:

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
configured 0.22 fails this criterion because of `subject_11` alone; every other student is within
10 points. Lowering `detector.conf` from 0.22 to 0.166 would keep that student but add 4 points of
boxes that are not students, which the tracker then has to handle. The tracker searches choose
`det_conf` on IDF1, which weighs both sides; the detector search isolates the recall side. Whether
to lower `detector.conf` was therefore left to the tracker experiments, until D2 showed that
degraded footage settles it.

### D2. Binary search under augmentation (done, 16 November 2026)

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
| D1, clean | 0.166 | no (one student) | 0.994 | 0.699 | 19.1% |
| D2, augmented | **0.1098** | **no (22 students lose more than 10 points)** | 0.978 | 0.590 | 14.5% |

343,135 augmented images (every frame of the 52 sessions, 5 variants each), 621 student-sessions.
Under augmentation the detector's confidences drop, so 0.22, which loses one student on clean data,
loses 22 by more than 10 recall points (4 by more than 25; the worst drops 39 points). Per threshold:

| Threshold | Mean recall | 10th percentile | Lowest | Lose > 10 points vs floor | Unmatched boxes |
|---|---|---|---|---|---|
| 0.05 (floor) | 0.993 | 0.988 | 0.943 | 0 | 37.9% |
| **0.1098 (optimal)** | **0.990** | **0.983** | **0.877** | **0** | **24.9%** |
| 0.22 (configured until D2) | 0.978 | 0.954 | 0.590 | 22 | 14.5% |

**Decision: `detector.conf` = 0.1098 for the rest of FYP-1** (set by `configs/derived.py` from
`outputs/tuning/detector_search_augmented.json`). It is the highest threshold that keeps every
student, on degraded footage, within 10 points of the floor; the price is more boxes that are not
students (24.9% against 14.5%), which the tracker's own thresholds handle (T2 tunes them with this
threshold fixed). The clean-data optimum (0.166, D1) is above it, so 0.1098 also keeps every student
on clean footage.

*Note added 19 November 2026:* the tracker's thresholds do more than handle the extra boxes.
BoT-SORT ignores every box at or below its `track_low_thresh` (BoxMOT 26, `split_by_confidence`),
and T2's winner sets it to 0.1745, so in the tracked pipeline the effective floor is 0.1745, not
0.1098. D2's guarantee holds for the detector's output, not for what reaches a track. T2 chose that
value on IDF1, which weighs missed students against false ones; experiment 09 §1 (T3) re-tunes it
for the pose detector, where coverage matters more.

### D3. The pose model as the detector, augmented, 1-point criterion (done, 19 November 2026)

Experiment 09 replaces YOLO11n with the YOLO11s pose model at 1280 px (§2), so D2's threshold no
longer applies. D3 is D2's scan for the new detector, on the same augmented images (every frame of
the 52 sessions, 5 variants each, the same seeds: 343,135 images, 621 student-sessions), with one
change: a **stricter criterion**. In experiment 09 a missed student costs more than a spurious box,
so no student may lose more than **1 point** of recall against the floor, instead of 10.
`python -m scripts.tuning.scan_detector --config docs/experiments/09_vfoa.json --augment --max-drop 0.01 --tag pose`;
`outputs/tuning/detector_search_augmented_pose.{csv,json}` (the tag keeps D2's files intact). Every
probe, plus D2's threshold for comparison:

| Threshold | Keeps every student (≤ 1 point) | Mean recall | Lowest | Largest drop | Unmatched boxes |
|---|---|---|---|---|---|
| 0.05 (floor) | yes | 0.996 | 0.855 | 0.000 | 41.8% |
| **0.0535 (optimal)** | **yes** | **0.995** | **0.850** | **0.005** | **40.9%** |
| 0.0570 | no | 0.995 | 0.844 | 0.011 | 40.2% |
| 0.0640 | no | 0.995 | 0.834 | 0.021 | 38.8% |
| 0.0781 | no | 0.995 | 0.814 | 0.041 | 36.5% |
| 0.1063 | no | 0.994 | 0.784 | 0.071 | 33.0% |
| 0.1098 (D2, YOLO11n's) | no | 0.994 | 0.780 | 0.075 | 32.7% |
| 0.1625 | no | 0.993 | 0.737 | 0.118 | 28.2% |
| 0.2750 | no | 0.991 | 0.641 | 0.214 | 22.0% |
| 0.5000 | no | 0.973 | 0.464 | 0.391 | 13.4% |

- **The criterion puts the threshold at the floor: 0.0535.** Mean recall hardly moves between 0.05
  and 0.11 (0.996 to 0.994); what moves is the hardest student, who loses 7.5 points by 0.11 and
  11.8 by 0.16. Keeping that student costs boxes that are not students: 41% of kept boxes at 0.0535
  against 33% at 0.11. Experiment 09 accepts that trade (`configs/derived_vfoa.py` reads the value).
- **Against YOLO11n (D2, same images, each at its own floor):** the pose model finds more students
  on average (mean recall 0.996 against 0.993, 10th percentile 0.993 against 0.988) but its hardest
  student is harder (0.855 against 0.943). No student is undetected at any threshold probed, and none is below 50%
  up to 0.275.
- **The floor bounds the search.** The criterion is measured against the 0.05 floor, so the search
  cannot ask for less; the hardest student is missed in 15% of frames even at the floor.
- **The tracker comes after.** BoT-SORT drops boxes at or below its own `track_low_thresh` (the note
  under D2), and T2's 0.1745 would throw away everything D3 keeps below it; experiment 03's T3
  re-tunes the tracker for this detector.

## Appendix A: how the threshold search works

`scripts/tuning/scan_detector.py` answers one question: **what is the highest confidence threshold
at which the detector still finds every student?** "Highest", because every box the threshold lets
through that is not a student (a teacher, a bag, a reflection) is work and risk for the tracker.

**1. Detect once, at the floor.** The detector runs once on every image at a very low confidence,
the *floor* (0.05, `tuning.tracker.det_conf_floor`), and every box is kept with its confidence.
Every threshold is then a filter on these cached boxes (keep a box if its confidence ≥ the
threshold), so trying a threshold costs a matching pass, not a detector run. On clean data the
boxes are the ones `tune_tracker` already caches; under augmentation (`--augment`) each frame is
augmented K times (K = 5 by default, `--variants`), each variant with its own random settings drawn
from the config's `augmentation` block, seeded per session so a re-run produces the same images,
and each variant is detected at the floor. The detections are cached per session, so an
interrupted scan resumes where it stopped.

**2. Score a threshold.** For every image:
- keep the boxes with confidence ≥ the threshold;
- match them one-to-one to the manual student boxes of that camera (Hungarian assignment on IoU,
  the overlap of two boxes divided by their union); a student counts as *found* in the image if
  their matched box overlaps at least 0.3, the same rule the tracker scoring uses (experiment 03
  §4). For a horizontally flipped variant the student boxes are mirrored first.

Per student and session, **recall** is the share of images in which they were found. Also
counted: the share of kept boxes matched to no student (*unmatched boxes*), the price side of the
trade-off.

**The manual boxes are static** (one per student per camera, experiment 01 §2), and they are not
right in every frame:
- a student who leans, turns or stands moves away from their box, so a correct detection can
  overlap it by less than 0.3 and count as a miss; a student who leaves still counts as missed;
- teaching assistants walking around, often in the background, have no box: their detections count
  as *unmatched*, so the unmatched share includes real people, not only false detections;
- a TA passing through a student's box can be matched to it, and counted as the student found in a
  frame where the student's own detection is missing.

None of this is measured. It affects every threshold alike on the same images, so the comparison
between thresholds, which is all the search uses, holds; the absolute recall and unmatched figures
are approximate.

**3. Compare with the floor, not with 1.** Some students are hard to see at any setting: the
hardest student in D2 is found in 94% of images even at the floor. A threshold cannot be blamed for
that, so each student's recall at a threshold is compared with their own recall at the floor. The
*drop* is floor recall minus recall at the threshold.

**4. The criterion: "keeps every student".** A threshold passes if, at once:
- no student is never detected (recall 0);
- no student is detected in fewer than half their images;
- no student's drop exceeds `--max-drop` (0.10, ten points of recall, in D1 and D2; 0.01 in D3).

One student failing is enough to fail the threshold: the criterion protects the worst-off student,
not the average. That is deliberate. A student the detector loses gets no track and no features,
and the average would hide them (D1: at 0.22 the mean is within 0.005 of the floor while one
student has lost 30 points).

**5. Why bisection works.** Raising the threshold only ever removes boxes, so a student's recall
can only stay the same or fall, and the criterion, once failed, stays failed for every higher
threshold. (Strictly, removing a box can occasionally reshuffle the one-to-one matching; D1's full
grid confirms the pattern holds on this data.) So there is a single boundary: everything below it
passes, everything above fails, and a binary search finds it:

```
lo = floor (0.05)            # known to pass: the floor is its own reference, every drop is 0
hi = 0.95                    # treated as failing
while hi - lo > 0.005:       # --tolerance
    mid = round((lo + hi) / 2, 4)
    if passes(mid): lo = mid  # the boundary is above mid
    else:           hi = mid  # the boundary is below mid
answer = lo                  # the highest threshold tried that passed
```

Each probe halves the interval, so 0.90 wide shrinks below 0.005 in 8 probes (0.90 / 2^8 ≈ 0.0035).
The answer is the last *passing* probe, so it is never above the boundary and at most 0.005 below
it. If no probe passes, the answer is the floor.

**A worked example: D3.** The probes, in the order they were run:

| Probe | Interval before | Threshold | Passes? | Interval after |
|---|---|---|---|---|
| 1 | 0.05–0.95 | 0.5000 | no | 0.05–0.5 |
| 2 | 0.05–0.5 | 0.2750 | no | 0.05–0.275 |
| 3 | 0.05–0.275 | 0.1625 | no | 0.05–0.1625 |
| 4 | 0.05–0.1625 | 0.1063 | no | 0.05–0.1063 |
| 5 | 0.05–0.1063 | 0.0781 | no | 0.05–0.0781 |
| 6 | 0.05–0.0781 | 0.0640 | no | 0.05–0.064 |
| 7 | 0.05–0.064 | 0.0570 | no (largest drop 0.011) | 0.05–0.057 |
| 8 | 0.05–0.057 | 0.0535 | yes (largest drop 0.005) | 0.0535–0.057 |

The interval is now 0.0035 wide, below the tolerance, so the search stops at **0.0535**. The true
boundary lies between 0.0535 and 0.057. Every search starts with the same probes (0.5, 0.275,
0.1625, ...) and branches at its first pass. D2: 0.1063 passes, so the search moves up: 0.1344 fails,
0.1203 fails, 0.1133 fails, 0.1098 passes, and the interval 0.1098–0.1133 ends it.

**6. What is written.** `outputs/tuning/detector_search<suffix>.csv` (every probe, one row each, with
the columns of the tables above) and `.json`: the criterion, the detector, the answer
(`optimal_threshold`), the full row at the answer, at the floor and at the configured threshold, and
whether the configured threshold passes. `configs/derived.py` (D2) and `configs/derived_vfoa.py` (D3)
read `optimal_threshold` from these files, so the config's threshold always comes from a scan.
`--grid` instead scores a fixed list of thresholds (D1's table), which is how the boundary's shape
is checked rather than assumed.

**7. What it does not answer.**
- **Not what reaches a track.** The tracker applies its own thresholds afterwards (note under D2).
- **Not the cost of false boxes.** It reports them but does not weigh them; the tracker searches
  (experiment 03), scored on IDF1, are where missed students and false ones are traded against each
  other.
- **Not identity.** A student found in every image can still be split across two tracks; that is
  the tracker's job and experiment 03's measure.

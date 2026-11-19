# Visual focus of attention: head direction from the classroom cameras

*Started 18 November 2026. Config: [`09_vfoa.json`](09_vfoa.json): experiment 08's base
([`08_pca.json`](08_pca.json)) with the head-pose unit added and 4 augmented copies. The config has the class context off
(`engagement.use_class_context=false`); every arm also runs with it on (§5). Hash `b1c2d485d3db` (D3's threshold, T3's tracker);
frame-pass cache `outputs/cache/tracks/2a11240a2d42/`, features `outputs/cache/features/4c761244568f/`. Extraction
pending.*

**Why this experiment.** Experiments 07 and 08 tried every transformation of the 95 inputs within
reach (k-NN's k, PCA, the class context, three model families). None moved the held-out ranking of
the experts' labels by more than ±0.02, and the self-report stayed at chance in every arm. The limit
is the information in the input, not its transformation. None of the 95 inputs measures the head,
and head direction is what the classroom attention literature leans on:
- visible engagement estimated from gaze, head pose and facial expression, improved by a synchrony
  feature (Goldberg et al., *Educational Psychology Review*, 2019);
- classroom synchrony measured as near-synchronous head rotations (Raca, Tormey and Dillenbourg,
  *Journal of Learning Analytics*, 2016);
- in the one outside work on DIPSER, head pose alone (3 numbers) carried most of the signal
  (`docs/misc/benchmarks.md` §3.4).

Head direction is estimated from the classroom (context) cameras, as everything else in FYP-1, and
used as new inputs to the per-labeler rating models, scored like 06–08.

## 1. One model: the pose model as the person detector

The pilot (§3) ran the pose model next to the person detector. Since it also finds people, it can
replace the detector: one network per frame instead of two, and every track gets the keypoints of
the very detection its update used (the tracker reports it, BoxMOT's `det_index`), so no matching is
needed. `src/detection/detector.py` returns the keypoints with the boxes when the model is a pose
model; `src/tracking/tracker.py` hands each track its detection's keypoints; the head-pose unit reads
them (`mode: "tracks"`). A 40-frame check on `group_03/experiment_01/context_cam_03`: every
track-frame got its keypoints (pose matched 100%, yaw defined 100%).

But the detector feeds everything after it, and the threshold (D2, 0.1098) and the tracker settings
(T2, IDF1 0.930) were tuned for YOLO11n (`docs/experiments/02_detection.md`, `03_tracking.md`). So the switch is
validated the same way, data-driven:

**D3: the augmented detector scan for the pose model.** The method of D2
(`scripts/tuning/scan_detector.py`): every frame of every session, augmented on the fly 5 times,
detected at the 0.05 floor; each threshold's boxes matched one-to-one to the manual student boxes;
bisection for the highest threshold that keeps every student. The criterion is stricter than D2's:
in this experiment a missed student costs more than a spurious box, so **no student may lose more
than 1 point of recall against the floor** (`--max-drop 0.01`, against D2's 0.10).

```
python -m scripts.tuning.scan_detector --config docs/experiments/09_vfoa.json --augment --max-drop 0.01 --tag pose
python -m scripts.tuning.scan_detector --config docs/experiments/09_vfoa.json --grid --detect --tag pose   # clean data, the full grid table (D1's first step)
```

The result (`outputs/tuning/detector_search_augmented_pose.json`) sets `detector.conf`: `09_vfoa.json`
names it under `derived` (`detector_scan`), and `configs/derived.py` reads the threshold from it, as
it reads D2 for the base config: the threshold is read from the scan, never typed in.

**D3 result (19 November 2026): 0.0535**, essentially the floor. Mean recall barely moves between
0.05 and 0.11 (0.996 to 0.994), but the hardest student loses 7.5 points by 0.11; keeping them
costs 41% non-student boxes against 33%. Against YOLO11n on the same images, the pose model finds
more students on average but its hardest student is harder (0.855 against 0.943 at the floor).
Every probe and the full reading: experiment 02, D3 (`docs/experiments/02_detection.md`).

**But the tracker has its own thresholds, and they come after the detector.** BoT-SORT
(BoxMOT 26, `trackers/common/detections` `split_by_confidence`) ignores every detection with
confidence at or below `track_low_thresh`, matches in a second pass those between it and
`track_high_thresh`, and starts a new track only above `new_track_thresh`. T2's winner, tuned for
YOLO11n, has 0.1745, 0.2852 and 0.5479. With those, everything D3 keeps between 0.05 and 0.17 would
be thrown away before tracking. So T2's options cannot simply be reused.

**T3: re-tune the tracker for the pose detector.** T2's search (BoT-SORT + ReID + CMC, beam
bisection, augmented copy 1, IDF1 against the manual boxes), with the detector threshold fixed at
D3's, starting from T2's winner, and the threshold ranges widened down to the floor
(`track_high_thresh` and `new_track_thresh` from 0.05; `track_low_thresh` already is). Run in your
terminal; the check first verifies the fast path on one session:

```
python -m scripts.tuning.tune_tracker --config docs/experiments/09_vfoa.json --profile full_reid_cmc --augment 1 --det-conf-from outputs/tuning/detector_search_augmented_pose.json --tag pose --check
python -m scripts.tuning.tune_tracker --config docs/experiments/09_vfoa.json --profile full_reid_cmc --augment 1 --det-conf-from outputs/tuning/detector_search_augmented_pose.json --tag pose --start outputs/tuning/tracker_botsort_full_reid_cmc_aug1-4f270e71_bisect.json
```

Results: `outputs/tuning/tracker_botsort_full_reid_cmc_aug1-4f270e71_pose_bisect.json`; resumable.
The first run also detects augmented copy 1 with the pose model at 1280 and embeds every box for
ReID. Compared with T2's 0.930 IDF1, and on coverage, the share of student-frames on a track
(T2: 0.958), which is what experiment 09 needs. Its options then replace T2's in experiment 09's
config.

**T3 result (19 November 2026): IDF1 0.901, coverage 0.982** (trial 19 of 20, every group 0.891–0.913;
`outputs/tuning/tracker_botsort_full_reid_cmc_aug1-4f270e71_pose_bisect.json`). Run with
`scripts/parallel_runs.py tracker-scans` (the same search, with frames shared across the trials of
each level) after the first 6 trials with `tune_tracker`.

| | T2 (YOLO11n, `detector.conf` 0.1098) | T3 start (T2's options on the pose model, 0.0535) | **T3 best** |
|---|---|---|---|
| IDF1 | **0.930** | 0.873 | 0.901 |
| Coverage | 0.958 | **0.990** | 0.982 |
| ID switches per subject | 2.06 | 3.30 | **1.39** |
| Track ids per subject | 1.34 | 1.39 | **1.34** |
| Subjects found | 0.989 | 1.000 | **1.000** |

Best options: `track_high_thresh 0.4926, track_low_thresh 0.2373, new_track_thresh 0.624,
match_thresh 0.6583, max_lost_frames 31, proximity_thresh 0.7726, appearance_thresh 0.2429`.

What it shows:
- **The pose model with T2's options covers more but switches more.** At the start, coverage is
  0.990 (T2: 0.958) and every student is found, but with 3.3 identity switches per student IDF1 falls
  to 0.873: the extra low-confidence boxes give the tracker more chances to swap identities.
- **Raising `track_high_thresh` fixed most of it.** From 0.29 to 0.49 (trial 12), switches halved
  (3.3 to 1.5) and IDF1 rose to 0.893, for 0.6 points of coverage: fewer boxes in the first
  association pass, so fewer wrong matches. A shorter memory (`max_lost_frames` 96 to 31) and a
  stricter `new_track_thresh` (0.55 to 0.62) added the rest.
- **The search raised the floor instead of lowering it.** The widened ranges allowed thresholds down
  to 0.05, but every step down lost IDF1 (`track_low_thresh` 0.11: 0.870; `track_high_thresh` 0.17:
  0.860; `new_track_thresh` 0.30: 0.859). The winner's `track_low_thresh` is 0.2373, above T2's
  0.1745, so the tracked pipeline drops every box below 0.24 whatever D3 keeps: any detector
  threshold up to 0.24 gives the same tracks, and D3's 0.0535 makes no difference to them. The pose
  model still covers more students than YOLO11n did (0.982 against 0.958).
- **IDF1 is 0.03 below T2's even with fewer identity switches.** This run does not show why. For
  experiment 09, coverage and every student found matter more than the last points of IDF1.

`09_vfoa.json` now names T3 under `derived` (`tracker_scan`), so `configs/derived.py` applies T3's options.

## 2. The head-pose unit

The design choices behind it are explained in guide 14 (`docs/guide/14_head_direction.md`).

`src/bfmu/head_pose.py` (BFMU frame unit `head_pose`, tests in `tests/test_head_pose.py`). An
ultralytics YOLO pose model gives each person's COCO keypoints; pose instances are matched to the
tracked students one-to-one by box IoU (at least 0.3). Per student and frame:

| Signal | What it is |
|---|---|
| `hp.yaw` | nose x minus the midpoint of the ears (or of the shoulders when an ear is missing), over that pair's width: 0 = facing the camera's axis, the sign = which side |
| `hp.head_height` | shoulder midpoint y minus nose y, over shoulder width: low = head down (notes, phone, laptop) |
| `hp.face_conf` | mean keypoint confidence of nose and eyes: high = face visible to this camera |
| `hp.ear_asym` | left-ear minus right-ear confidence: one ear visible = head turned |
| `hp.matched` | 1 if a pose was matched to the track |

Window features per student: mean and SD of yaw and of head height, mean |yaw|, the share of frames
with the head low (head height below 0.1), mean face confidence, mean ear asymmetry, and the share of
frames with a matched pose (9 columns).

## 3. Pilot: which pose input?

`python -m scripts.analysis.head_pose_pilot` (`scripts/analysis/head_pose_pilot.py`): the unit in
five variants on 40 frames from each of three sessions (one per group), on the tracks experiment 06
cached, so no new frame pass. Heads are about 35–50 px wide in the 1280×720 frames (student boxes
are a median 131×147 px), so the input resolution was the open question.

| Variant | Seconds per frame | Pose matched | Yaw defined | Head height defined | Face confidence | Yaw p10 / p50 / p90 |
|---|---|---|---|---|---|---|
| nano, full frame at 640 | 0.026 | 0.97 | 0.97 | 0.97 | 0.92 | −0.39 / −0.04 / +0.35 |
| nano, full frame at 1280 | 0.026 | 0.97 | 0.95 | 0.95 | 0.95 | −0.37 / −0.01 / +0.33 |
| nano, upper-body crops at 256 | 0.061 | 1.00 | 0.89 | 0.89 | 0.88 | −0.37 / −0.02 / +0.34 |
| **small, full frame at 1280** | **0.027** | **0.99** | **0.97** | **0.97** | **0.94** | −0.36 / −0.04 / +0.32 |
| small, upper-body crops at 256 | 0.062 | 1.00 | 0.91 | 0.91 | 0.88 | −0.38 / −0.03 / +0.31 |

Overlays (`outputs/experiments/09_vfoa/pilot/<variant>/`) show the signals behave: students turned
to one side read yaw ±0.3–0.5, students facing forward about 0, and students looking down at a phone
or notes read a low or negative head height (−0.1 to −0.2) against 0.1–0.3 for heads up. The cameras
look down on students who face them, which is why faces are visible in almost every frame.

**Choice: the small model on the full frame at 1280** (`yolo11s-pose.pt`): the best coverage
(pose matched 99%, yaw defined 97%) at 27 ms per frame, about 15% on top of the existing frame pass
(about 190 ms per frame). Crops cost twice as much and leave yaw undefined more often.

## 4. Extraction

The unit is a frame unit, so it changes the frame-pass cache key: the 52 recordings and their
augmented copies are re-extracted from scratch with detection, tracking and every unit: the pose
model as the detector at D3's threshold, and BoT-SORT + ReID + CMC with T3's options (§1), both read
by `configs/derived.py` from the scans `09_vfoa.json` names under `derived`.

```
python -m scripts.parallel_runs extract --config docs/experiments/09_vfoa.json --workers 3   # resumable
```

The originals first, then 4 augmented copies (5 passes of 52 recordings). Head pose is the kind of
input that overfits to camera angle, seat and lighting, which the copies vary; the arms test the dose.

## 5. Arms (after extraction)

Every input arm runs twice, with and without the class context (`engagement.use_class_context`, a
training-time switch: the context columns are computed in the feature pass either way, so both
variants come from one extraction). With the context on, the class columns now include the class
mean and SD of every head-pose column (`vec.hp.*`): a crude "where is the rest of the class looking",
which previews the class-relative feature of arm 4.

| Input | Without the class context | With it (+ the class mean and SD of every per-student column) |
|---|---|---|
| 1. the 27 per-student columns (08's baseline) | 27 columns | 27 + 54 + 14 = 95 |
| 2. plus head pose | 27 + 9 = 36 | 36 + 72 + 14 = 122 |
| 3. head pose alone | 9 | only the head-pose class columns added: 9 + 18 = 27 |
| 4. plus class-relative head direction (the student's yaw and head height against the class median in the same window, and the share of the class pointing the same way) | 36 + the new columns | the same, with the context |
| 5. the augmentation dose (0, 1, 4 copies) on the best of 2–4 | | |

*Column counts with the context are approximate until the feature pass has run (the 14 are the
threshold shares, occupancy cells and the student count).*

Each with logistic regression and the random forest, in the standard tables (profile and metrics,
train / test, human-ceiling rows), with per-labeler tables for the best arm, and the feature study
(`scripts/analysis/feature_study.py`) re-run so the head-pose columns appear against every labeler.
The questions: does head direction carry more signal than optical flow; does the class's head
direction add to the student's own (the context arms); and does any of it reach the self-report?

## 6. Result

*Pending: extraction.*

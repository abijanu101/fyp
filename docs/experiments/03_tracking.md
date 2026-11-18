# Multi-object tracking

This document records how multi-object tracking was brought up on DIPSER with BoxMOT, how the
tracker was evaluated and tuned, and what was found. Its input is experiment 02's detector at the
D2 threshold ([02_detection.md](02_detection.md)).

## 1. Goal

Design §4: two trackers, StrongSORT and BoT-SORT, fed by the shared person detector and compared
head-to-head on ID switches, fragmentation and downstream usefulness. The winner becomes the
default; the other stays selectable. Tracks feed the per-student features (BFMU), so tracking
errors propagate into every downstream number.

**Experiments in this document** (all on the 52 in-scope sessions, every frame):

| | Experiment | Status | Headline |
|---|---|---|---|
| T1 | Trackers on clean data: BoT-SORT without ReID, StrongSORT; random search, 20 trials (§6) | done | BoT-SORT 0.905 IDF1, StrongSORT 0.878 |
| T2 | Trackers on augmented data: BoT-SORT, BoT-SORT + ReID + CMC, StrongSORT; beam bisection (§6) | done | **BoT-SORT + ReID + CMC 0.930 IDF1**, BoT-SORT 0.914, StrongSORT 0.877 |
| T3 | T2's search re-run with the pose model as the detector (experiment 02, D3), thresholds widened to the floor (experiment 09 §1) | running | BoT-SORT's own `track_low_thresh` (T2's 0.1745) would drop every box D3 keeps below it |

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
Assumption: every annotated student is present, inside their box, in every frame of the evaluated
span. That is not true in every frame (experiment 01 §2 lists the cases):
- a student who leans, turns, stands or briefly leaves is scored as uncovered even when tracked;
- teaching assistants walking around, often in the background, have no box, so their tracks count
  as false positives in IDF1 although they are real people;
- a TA passing through a student's box can be matched to it, so a TA's track can briefly count as
  that student, inflating coverage or causing an apparent ID switch.

None of these is measured. They act the same way on every configuration scored on the same frames,
so comparisons between trackers stay fair, but the absolute IDF1 and coverage are approximate.

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
equally for every configuration (see Ground truth above).

An early version mapped tracks to students many-to-one (majority vote) when computing IDF1, which
gave a fragmented student full credit for both fragments. A unit test caught it, and IDF1 now uses
the one-to-one assignment.

## 5. Tuning method

`scripts/tuning/tune_tracker.py`; details in the contributor's guide, Chapter 8.

- **Detections cached once** per session at a confidence floor of 0.05. Each trial filters them at
  its own `det_conf`, so detector confidence is tuned jointly with the tracker at no extra cost,
  and only the tracker re-runs.
- **Search method.** The first scans (quick profiles and the three full-dataset experiments of
  T1) used a **seeded random search**: trial 0 is the starting configuration, every other trial
  draws each parameter independently from the space. From 15 November 2026 the default is a
  **beam bisection search** (guide §8.5), now run with `--randomize` to get the random search:
  - starting point: the winner of the same profile's random search;
  - per node and per numeric parameter, a binary-search step: probe the midpoints of the lower and
    upper halves of the parameter's interval, keep the best of the three, halve the interval
    (k branches per node), plus one branch combining every winning move;
  - a beam of 2 nodes per level as the bound; parameters with no effect under the profile's fixed
    settings (BoT-SORT's `proximity_thresh` and `appearance_thresh` without ReID) are held;
  - deterministic and resumable, sharing the random search's per-session result log, so no trial
    is computed twice.

  One level costs up to (2k + 1) × 2 trials: 26 for BoT-SORT without ReID (k = 6), 34 with ReID
  (k = 8), 22 for StrongSORT (k = 5; `n_init` is categorical and held).
- **Search space** for BoT-SORT: `det_conf`, `track_high_thresh`, `track_low_thresh`,
  `new_track_thresh`, `max_lost_frames` (log scale), `match_thresh`, `proximity_thresh`,
  `appearance_thresh`, `use_embeddings` (ReID on/off), `use_cmc` (camera-motion compensation
  on/off). For StrongSORT: `det_conf`, `min_conf`, `max_cos_dist`, `max_iou_dist`, `max_age` (log
  scale), `n_init`, `use_cmc`.
- **Augmented data.** `--augment [COPY]` runs any profile on augmented copy COPY of the dataset
  (augmented on the fly: one fixed colour, brightness, noise, flip and occlusion setting per
  session copy, reproducible from its seed; boxes mirrored where flipped), with its own detection and ReID caches and result
  files. This compares the trackers' robustness on the same sessions under worse footage.
- **Profiles**:
  - `quick`: 3 sessions, one per group (`group_01/experiment_03/context_cam_02`,
    `group_02/experiment_04/context_cam_03`, `group_03/experiment_01/context_cam_01`), first 300
    frames (60 s) each, 25 trials; ReID and CMC are searched;
  - `full`: BoT-SORT, all 52 sessions, all frames, 20 trials, ReID fixed **off**, CMC fixed **on**;
  - `full_reid_cmc`: BoT-SORT, all 52 sessions, 20 trials, ReID and CMC fixed **on**;
  - `full_strongsort`: StrongSORT, all 52 sessions, 20 trials, CMC fixed **on**; trial 0 runs the
    quick-tuned StrongSORT values (`base_options`), not BoxMOT's defaults (T1).

  CMC is on in every full-dataset trial, so every trial decodes real frames and no result depends
  on the blank-frame shortcut. The only remaining shortcut is the ReID embedding cache, verified
  with decoded frames and CMC on (StrongSORT 42 vs 42 tracks, BoT-SORT with ReID 16 vs 16).
  An earlier CMC-off run of `full` was stopped at 16 of 40 trials when CMC was fixed on.
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

### T1. Clean data: BoT-SORT and StrongSORT, full dataset

Starting values for both came from earlier random searches on three 60-second clips (`quick`
profile, 25 trials): BoT-SORT went from 0.908 to 0.950 IDF1 there, StrongSORT from 0.851 to 0.932,
and BoT-SORT with ReID reached 0.940 at about 60× the tracking cost of motion-only BoT-SORT. The
full-dataset searches below supersede those clips.

**Setup.** All 52 sessions, every frame, scored against the manual boxes (§4). Seeded random search,
20 trials each; trial 0 is the starting configuration. CMC fixed on for both (every trial decodes
real frames). BoT-SORT without ReID (`full`); StrongSORT, which always uses ReID
(`full_strongsort`), starting from its quick-tuned values. Scores are means over the 52 sessions,
per group in G01–G03; they select on the same data, so they compare configurations rather than
estimate held-out quality.

**Summary.**

| | BoT-SORT, no ReID | StrongSORT |
|---|---|---|
| Status | complete (20 trials) | complete (20 trials) |
| Trial 0 (starting values) IDF1 | 0.888 | 0.843 |
| **Best IDF1** | **0.905** (trial 6) | **0.878** (trial 14) |
| Best, per group G01 / G02 / G03 | 0.876 / 0.922 / 0.917 | 0.836 / 0.907 / 0.894 |
| Best: coverage | 0.968 | 0.946 |
| Best: ID switches per subject | 1.04 | 9.26 |
| Best: track ids per subject | 1.40 | 1.77 |
| Median IDF1 over trials (IQR) | 0.815 (0.773–0.876) | 0.825 (0.781–0.856) |
| Trials within 0.01 of the best | 2 (trials 6, 17) | 4 (trials 14, 8, 13, 6) |
| Tracking cost per frame | ~2 ms | ~120–165 ms (ReID) |

**BoT-SORT without ReID, every trial** (`outputs/tuning/tracker_botsort_full.sessions.jsonl`;
the run was restarted with `--fresh` after `det_thresh` was removed from the configuration, and its
trials 0–10 reproduced the earlier run's numbers exactly, confirming `det_thresh` had no effect):

| Trial | IDF1 | G01 | G02 | G03 | Coverage | Switches / subj | Track ids / subj | det_conf | high | low | new | match | max_lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.888 | 0.857 | 0.918 | 0.890 | 0.900 | 0.736 | 1.275 | 0.220 | 0.250 | 0.210 | 0.670 | 0.800 | 17 |
| 1 | 0.697 | 0.656 | 0.738 | 0.702 | 0.839 | 3.911 | 3.652 | 0.208 | 0.503 | 0.232 | 0.607 | 0.518 | 10 |
| 2 | 0.726 | 0.688 | 0.753 | 0.741 | 0.813 | 1.833 | 2.570 | 0.474 | 0.565 | 0.094 | 0.629 | 0.867 | 10 |
| 3 | 0.835 | 0.779 | 0.870 | 0.861 | 0.908 | 2.056 | 1.976 | 0.317 | 0.535 | 0.212 | 0.214 | 0.635 | 31 |
| 4 | 0.881 | 0.849 | 0.895 | 0.902 | 0.922 | 0.975 | 1.498 | 0.254 | 0.544 | 0.147 | 0.543 | 0.949 | 142 |
| 5 | 0.753 | 0.716 | 0.789 | 0.757 | 0.782 | 1.146 | 1.906 | 0.389 | 0.667 | 0.139 | 0.443 | 0.736 | 23 |
| **6** | 0.905 | 0.876 | 0.922 | 0.917 | 0.968 | 1.042 | 1.404 | 0.229 | 0.314 | 0.206 | 0.396 | 0.767 | 25 |
| 7 | 0.858 | 0.829 | 0.875 | 0.872 | 0.886 | 1.192 | 1.487 | 0.433 | 0.368 | 0.088 | 0.638 | 0.854 | 19 |
| 8 | 0.794 | 0.740 | 0.835 | 0.813 | 0.892 | 3.226 | 2.554 | 0.418 | 0.245 | 0.195 | 0.402 | 0.604 | 12 |
| 9 | 0.813 | 0.764 | 0.849 | 0.829 | 0.897 | 3.055 | 2.146 | 0.369 | 0.515 | 0.282 | 0.383 | 0.590 | 128 |
| 10 | 0.719 | 0.676 | 0.758 | 0.726 | 0.727 | 1.297 | 2.043 | 0.482 | 0.674 | 0.165 | 0.510 | 0.725 | 32 |
| 11 | 0.876 | 0.833 | 0.902 | 0.896 | 0.911 | 1.026 | 1.518 | 0.299 | 0.556 | 0.283 | 0.407 | 0.738 | 84 |
| 12 | 0.779 | 0.739 | 0.813 | 0.789 | 0.750 | 0.743 | 1.471 | 0.392 | 0.691 | 0.289 | 0.207 | 0.917 | 138 |
| 13 | 0.811 | 0.768 | 0.833 | 0.834 | 0.811 | 1.170 | 1.738 | 0.489 | 0.601 | 0.281 | 0.440 | 0.900 | 93 |
| 14 | 0.878 | 0.833 | 0.908 | 0.896 | 0.923 | 1.363 | 1.565 | 0.316 | 0.507 | 0.057 | 0.220 | 0.699 | 124 |
| 15 | 0.839 | 0.791 | 0.881 | 0.849 | 0.849 | 0.749 | 1.467 | 0.106 | 0.621 | 0.067 | 0.665 | 0.841 | 40 |
| 16 | 0.818 | 0.768 | 0.861 | 0.828 | 0.844 | 0.974 | 1.710 | 0.272 | 0.644 | 0.106 | 0.329 | 0.935 | 46 |
| 17 | 0.904 | 0.878 | 0.922 | 0.914 | 0.940 | 0.937 | 1.343 | 0.215 | 0.344 | 0.153 | 0.605 | 0.764 | 45 |
| 18 | 0.789 | 0.741 | 0.822 | 0.807 | 0.850 | 1.275 | 1.992 | 0.351 | 0.624 | 0.086 | 0.476 | 0.932 | 27 |
| 19 | 0.733 | 0.691 | 0.778 | 0.734 | 0.834 | 5.848 | 3.352 | 0.464 | 0.205 | 0.141 | 0.408 | 0.519 | 93 |

Best parameters (trial 6): `det_conf 0.229, track_high_thresh 0.314, track_low_thresh 0.206,
new_track_thresh 0.396, match_thresh 0.767, max_lost_frames 25`. Trial 17 (0.904) is a near tie with
a lower `det_conf` (0.215) and longer memory (`max_lost_frames` 45).

**StrongSORT, every trial** (`outputs/tuning/tracker_strongsort_full_strongsort.{csv,json}`; trial
0 re-ran at the quick-tuned values after an earlier trial 0 at BoxMOT's defaults scored 0.715,
because `min_conf` 0.6 drops students, experiment 02, D1):

| Trial | IDF1 | G01 | G02 | G03 | Coverage | Switches / subj | Track ids / subj | det_conf | min_conf | max_cos | max_iou | max_age | n_init |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.843 | 0.804 | 0.854 | 0.873 | 0.992 | 3.878 | 1.947 | 0.220 | 0.215 | 0.325 | 0.858 | 15 | 3 |
| 1 | 0.723 | 0.669 | 0.758 | 0.745 | 0.863 | 8.228 | 5.574 | 0.355 | 0.507 | 0.120 | 0.507 | 21 | 2 |
| 2 | 0.843 | 0.797 | 0.864 | 0.869 | 0.863 | 5.340 | 1.930 | 0.343 | 0.508 | 0.372 | 0.921 | 72 | 3 |
| 3 | 0.816 | 0.772 | 0.831 | 0.847 | 0.996 | 10.708 | 2.959 | 0.101 | 0.188 | 0.117 | 0.828 | 102 | 1 |
| 4 | 0.853 | 0.795 | 0.883 | 0.884 | 0.979 | 5.683 | 2.264 | 0.317 | 0.162 | 0.311 | 0.513 | 23 | 3 |
| 5 | 0.750 | 0.695 | 0.796 | 0.764 | 0.728 | 3.652 | 2.220 | 0.368 | 0.599 | 0.408 | 0.673 | 58 | 3 |
| 6 | 0.872 | 0.821 | 0.902 | 0.896 | 0.962 | 8.058 | 2.038 | 0.374 | 0.168 | 0.444 | 0.675 | 58 | 3 |
| 7 | 0.783 | 0.730 | 0.818 | 0.805 | 0.821 | 7.468 | 3.550 | 0.389 | 0.545 | 0.255 | 0.719 | 41 | 1 |
| 8 | 0.877 | 0.845 | 0.882 | 0.904 | 0.988 | 5.198 | 1.767 | 0.243 | 0.269 | 0.261 | 0.767 | 47 | 3 |
| 9 | 0.865 | 0.826 | 0.870 | 0.900 | 0.991 | 9.676 | 1.979 | 0.257 | 0.142 | 0.214 | 0.780 | 111 | 2 |
| 10 | 0.834 | 0.779 | 0.869 | 0.858 | 0.941 | 5.163 | 2.655 | 0.415 | 0.268 | 0.538 | 0.526 | 19 | 3 |
| 11 | 0.804 | 0.755 | 0.830 | 0.832 | 0.996 | 11.095 | 2.557 | 0.160 | 0.126 | 0.498 | 0.604 | 34 | 2 |
| 12 | 0.803 | 0.743 | 0.844 | 0.827 | 0.929 | 4.887 | 3.293 | 0.179 | 0.436 | 0.390 | 0.634 | 13 | 2 |
| 13 | 0.875 | 0.834 | 0.902 | 0.891 | 0.943 | 10.331 | 1.796 | 0.180 | 0.415 | 0.283 | 0.547 | 128 | 3 |
| **14** | 0.878 | 0.836 | 0.907 | 0.894 | 0.946 | 9.256 | 1.766 | 0.276 | 0.410 | 0.350 | 0.691 | 133 | 3 |
| 15 | 0.850 | 0.796 | 0.887 | 0.871 | 0.877 | 9.073 | 1.868 | 0.498 | 0.349 | 0.330 | 0.841 | 131 | 2 |
| 16 | 0.753 | 0.700 | 0.797 | 0.768 | 0.785 | 4.396 | 2.956 | 0.414 | 0.566 | 0.467 | 0.820 | 31 | 2 |
| 17 | 0.776 | 0.724 | 0.795 | 0.810 | 0.997 | 19.663 | 3.710 | 0.146 | 0.107 | 0.564 | 0.936 | 72 | 1 |
| 18 | 0.802 | 0.750 | 0.832 | 0.828 | 0.815 | 6.947 | 2.218 | 0.492 | 0.545 | 0.174 | 0.938 | 134 | 3 |
| 19 | 0.766 | 0.721 | 0.789 | 0.791 | 0.789 | 3.893 | 2.656 | 0.429 | 0.562 | 0.216 | 0.861 | 37 | 3 |

Best parameters (trial 14): `det_conf 0.276, min_conf 0.410, max_cos_dist 0.350, max_iou_dist
0.691, max_age 133, n_init 3`. The best trials all keep a lost track for a long time (`max_age`
47–133 frames, 9–27 s) and confirm a track only after 3 hits.

**Reading.**

- **BoT-SORT without ReID wins on IDF1** (0.905 against 0.878), in every group, at a small
  fraction of StrongSORT's cost.
- **StrongSORT's identities flicker.** It switches a student's matched track id 4–20 times per
  session across trials (9.3 for its best), against about 1 for BoT-SORT, at similar or higher
  coverage: the box stays on the right person, but the id attached to it changes back and forth.
  IDF1 forgives part of this (the one-to-one mapping keeps the largest share); the feature pass
  does not, since each switch cuts a student's time series.
- **The optimum is a plateau, not a point.** The best two or four trials of each search lie
  within 0.01 of each other with different parameters: the bisection refinement (T2) starts from
  the winners rather than expecting large gains.

### T2. Augmented data: BoT-SORT, BoT-SORT with ReID and CMC, StrongSORT (done, 16 November 2026)

**Setup.** The same 52 sessions on augmented copy 1 (one fixed setting per session copy: colour,
brightness ±35, gamma, sensor noise up to 0.35, horizontal flip with p 0.5, grey occluders with
p 0.5; boxes mirrored where flipped; §5). Beam bisection search, beam 2, one level each (§5).
Each search's trial 0 is its starting point scored on augmented data, which alone measures how much
each tuned tracker degrades on worse footage. **The detection threshold is not searched in T2**: it is
D2's result, kept fixed (`--det-conf-from outputs/tuning/detector_search_augmented.json`), so the
threshold is settled once, by D2, for the rest of FYP-1, and T2 tunes how each tracker uses the
detections it lets through. It is also passed to the trackers as `det_thresh`, although BoT-SORT
and StrongSORT only store that value (BoxMOT 26: CMC masks with `cmc_min_conf` or StrongSORT's
`min_conf`). Run all three at once with
`python -m scripts.parallel_runs tracker-scans --augment 1 --trials 26 34 22 --det-conf-from outputs/tuning/detector_search_augmented.json`.

| Tracker | Profile | Starts from | Trials (one level) | Clean IDF1 of the start | Trial 0 on augmented | Best on augmented | Change vs clean |
|---|---|---|---|---|---|---|---|
| BoT-SORT, no ReID | `full` | its T1 winner | 26 | 0.905 | 0.899 | 0.914 (trial 15) | −0.006 at the start |
| BoT-SORT + ReID + CMC | `full_reid_cmc` | **vanilla BoT-SORT's** T1 winner | 34 | 0.905 (as vanilla) | 0.901 | **0.930** (trial 30) | −0.004 at the start |
| StrongSORT | `full_strongsort` | its T1 winner | 22 | 0.878 | 0.846 | 0.877 (trial 10) | −0.032 at the start |

All three with `detector.conf` fixed at D2's 0.1098 (trial 0 is each start under that threshold on
augmented copy 1). The best of each, mean over the 52 sessions:

| | BoT-SORT | **BoT-SORT + ReID + CMC** | StrongSORT |
|---|---|---|---|
| IDF1 | 0.914 | **0.930** | 0.877 |
| IDF1 per group G01 / G02 / G03 | 0.903 / 0.920 / 0.918 | **0.923 / 0.937 / 0.931** | 0.854 / 0.895 / 0.885 |
| Coverage | 0.963 | 0.958 | 0.970 |
| ID switches per subject | 1.50 | 2.06 | 20.9 |
| Track ids per subject | 1.35 | 1.34 | 1.83 |
| Subjects found | 0.986 | 0.989 | 0.998 |
| Tracking cost per frame | ~2 ms | ~120–165 ms (ReID; embeddings cached for the search) | ~120–165 ms |

Best parameters (the `config_patch` of each `_bisect.json`):

- BoT-SORT + ReID + CMC: `track_high_thresh 0.2852, track_low_thresh 0.1745, new_track_thresh
  0.5479, match_thresh 0.7111, max_lost_frames 96, proximity_thresh 0.7726, appearance_thresh
  0.2857` (ReID and CMC on).
- BoT-SORT: `track_high_thresh 0.2568, track_low_thresh 0.2058, new_track_thresh 0.2979,
  match_thresh 0.8587, max_lost_frames 96` (ReID off, CMC on).
- StrongSORT: `min_conf 0.255, max_cos_dist 0.2249, max_iou_dist 0.8207, max_age 95, n_init 3`.

Readings:

- **The tuned trackers hardly degrade on augmented footage** (−0.004 to −0.006 IDF1 at the
  BoT-SORT starts), once the detection threshold follows D2; StrongSORT degrades more (−0.032).
- **ReID earns its place on degraded footage**: +0.016 IDF1 over the same tracker without it, in
  every group. On clean data (quick clips, T1) it had not beaten tuned motion-only BoT-SORT.
- **Both BoT-SORT winners keep tracks longer** (`max_lost_frames` 96, about 19 s at 5 fps, against
  25 for the T1 winner): with the lower detection threshold, a lost student is worth waiting for.
- **ID switches rise with ReID** (2.06 against 1.50 per subject) while IDF1 is higher and track ids per
  subject are about the same: IDF1 weighs how much of each student's time one identity covers,
  switches count every change. How long the extra switches last was not measured.
- **StrongSORT stays last**, with ten times the ID switches of BoT-SORT.

**How it was run.** All three scans at once with the parallel launcher (`scripts/parallel_runs.py
tracker-scans`), after D2 in the same command: a prepare stage (augmented detections and ReID
embeddings, one decode per session) and the three searches sharing decoded frames in 5 worker
processes. D2 ended at 03:10 and T2 at 09:34: about 6.4 hours for 82 full-dataset trials,
preparation included, where the serial tool needed about an hour per trial. The launcher's results
are identical to `tune_tracker`'s (checked trial by trial with CMC on and off).

```
python -m scripts.tuning.scan_detector --augment --workers 4; if ($?) { python -m scripts.parallel_runs tracker-scans --augment 1 --trials 26 34 22 --workers 5 --gpu-workers 2 --det-conf-from outputs/tuning/detector_search_augmented.json }
```

A clean BoT-SORT + ReID search was started and stopped before its first trial completed (15
November 2026): its only purpose was a starting point, and vanilla BoT-SORT's winner serves.

```
python -m scripts.tuning.tune_tracker --profile full --augment 1 --trials 26
python -m scripts.tuning.tune_tracker --profile full_reid_cmc --augment 1 --trials 34
python -m scripts.tuning.tune_tracker --profile full_strongsort --augment 1 --trials 22
```

Results: `outputs/tuning/tracker_<tracker>_<profile>_aug1-4f270e71_bisect.{csv,json}` and the
per-session `.sessions.jsonl`. The serial equivalents (`tune_tracker --profile ... --augment 1
--det-conf-from ...`) resume the same files.

## 7. Tracker decision

**Final (16 November 2026): BoT-SORT with ReID and CMC, with T2's winning parameters, at
`detector.conf` 0.1098 (D2).** It has the best IDF1 on augmented footage (0.930) and in every group;
T1 made BoT-SORT the family, T2 shows ReID adds 0.016 on degraded footage. The cost is the ReID
model in the frame pass (about 120–165 ms per frame, roughly doubling extraction time; Chapter 5).
`configs/derived.py` applies D2's threshold and the T2 winner's `config_patch` to every config load,
from the results files themselves, so the frame pass of the full run uses exactly these values.

Until 16 November: BoT-SORT without ReID with quick-tuned values was the interim default.

**Detector and tracker thresholds interact.** BoT-SORT drops every box at or below its own
`track_low_thresh` (0.1745 in T2's winner) before association, so in this pipeline the effective
detection floor is 0.1745, not D2's 0.1098 (experiment 02, note under D2). Experiment 09 uses a
detector threshold at the floor (D3, 0.0535) and so re-tunes the tracker for it (T3).

## 8. Engineering issues encountered

Recorded because each cost a failed multi-hour run:

| Issue | Symptom | Fix |
|---|---|---|
| A new YOLO detector was built per session in the tuning script | memory growth | one shared detector |
| Indexing a compressed `.npz` inside a per-frame loop (every access decompresses the whole array, and slices kept each copy alive) | `MemoryError` after ~18 sessions | read each array once, then slice |
| 314 empty PNGs in one session; OpenCV raises on an empty buffer instead of returning nothing | crash at session 43 | zero-byte files dropped when the ZIP is indexed |

## 9. Open

- The downstream criterion in §4 (context-prediction accuracy per tracker) has not been measured;
  it matters if the tracking metrics of two configurations come out close.

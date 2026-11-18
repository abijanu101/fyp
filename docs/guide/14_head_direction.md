# Chapter 14: Head direction from the classroom cameras

Experiment 09 adds a new family of inputs: where each student's head points, a cheap stand-in for
their visual focus of attention (VFoA). This chapter explains every choice behind it:
- why head direction, and why from the classroom cameras;
- why a keypoint pose model rather than a head-pose or gaze network;
- how keypoints become the signals;
- how poses are matched to tracked students;
- the input resolution, chosen by a pilot;
- the window features;
- what is planned next, what it costs, and its limits.

Code: `src/bfmu/head_pose.py` (BFMU frame unit `head_pose`), tests in `tests/test_head_pose.py`, the
pilot in `scripts/analysis/head_pose_pilot.py`. Results: `docs/experiments/09_vfoa.md`.

## 14.1 What is measured, and why

**Visual focus of attention** is what a person is looking at. Measuring it properly needs gaze (where
the eyes point) and the scene's layout (where the board, the teacher, the laptop are). We measure the
first, cheaper step: **the direction of the head**, relative to the body and to the camera.

Why the head:
- **The models need new information, not new transformations.** Experiments 07 and 08 varied the
  model, k, PCA and the class context; nothing moved the held-out ranking by more than ±0.02. The
  95 existing inputs measure body motion, optical flow and SCB behaviour scores; none measures the
  head.
- **The literature measures attention through the head.** Visible engagement is estimated from gaze,
  head pose and facial expression (Goldberg et al., 2019). Classroom synchrony is measured as
  near-synchronous head rotations (Raca, Tormey and Dillenbourg, 2016).
- **On DIPSER itself,** the one outside work found head pose alone (pitch, yaw, roll: 3 numbers)
  carried most of the signal (`docs/misc/benchmarks.md` §3.4).

## 14.2 Why the classroom cameras, not the desk cameras

DIPSER also has a desk camera per student (640×480) and ships precomputed head pose for it
(`docs/misc/benchmarks.md` §4). That would be the easy route, and it was set aside on purpose:

| | Classroom (context) cameras | Desk cameras |
|---|---|---|
| Setting | one camera for the whole room, as in the rest of FYP-1 | one camera per student |
| Head size | small: about 35–50 px wide | large: the face fills the frame |
| Head pose | ours to estimate | precomputed by the dataset authors |
| What a result shows | that a classroom camera can measure it | what head pose can do with ideal input |

FYP-1's claim is about what a cheap classroom setup can measure. A result that needs a camera per
student would not support it. The desk cameras stay available as an upper bound: if the classroom
estimate fails, desk-camera head pose would show whether the signal is missing or only the
resolution.

## 14.3 Why a keypoint pose model

There were four ways to get head direction from a classroom frame:

| Approach | What it needs | Why not, or why |
|---|---|---|
| **Body keypoint pose model** (chosen) | the whole person; reads nose, eyes, ears and shoulders | robust at low resolution because it uses the body around the head; one model gives head and shoulders, so head direction can be read relative to the body; same library as our detector (ultralytics) |
| Head-pose regression network (e.g. 6DRepNet, Hempel et al., 2022) | a tight face crop | trained on large, frontal-ish faces; at 35–50 px the crop holds too little; needs its own face detector; kept as a later variant if the keypoint proxies prove too coarse |
| Face mesh (e.g. MediaPipe) | a face of a usable size | fails on small faces; 478 landmarks is far more than head direction needs |
| Gaze estimation | the eyes at high resolution | the eyes are a few pixels wide in these frames |

The pose model is an ultralytics YOLO pose model, with the 17 COCO person keypoints (the COCO
dataset: Lin et al., 2014; its keypoint annotation was added later). Only seven are used: nose, both
eyes, both ears, both shoulders.

## 14.4 From keypoints to signals

Per matched student and frame (keypoints below a confidence of 0.5 count as missing):

- **`hp.yaw`, head turned left or right.** The nose's horizontal offset from the midpoint of the
  ears, divided by the distance between the ears.
  - If an ear is hidden, the shoulders take their place.
  - 0 means the nose is centred: the head faces along the camera's axis. The sign says which way.
  - Dividing by a width makes it independent of how far the student sits from the camera.
- **`hp.head_height`, head up or down.** The shoulders' midpoint minus the nose's height, divided by
  shoulder width.
  - The cameras look down on the class, so a lowered head (reading notes, a phone, a laptop) brings
    the nose towards the shoulders: a smaller value.
  - It is a pitch proxy, not an angle; slumping lowers it too.
- **`hp.face_conf`, face visible.** The mean confidence of the nose and both eyes: high when the
  face is towards this camera, low when it is turned away or hidden.
- **`hp.ear_asym`, which ear shows.** Left-ear confidence minus right-ear confidence: one ear
  visible means the head is turned.
- **`hp.matched`.** 1 if a pose was matched to the student in this frame, else 0; a missing pose is
  information too (occlusion, turned away).

What these are not:
- **Not angles.** Yaw and head height are ratios. They rank head positions correctly, which is what
  the models need, but "0.3" is not a number of degrees.
- **Camera-relative, not room-relative.** Yaw 0 means "towards this camera", not "towards the
  board". Two students looking at the same board have different yaws if they sit at different
  angles to the camera. §14.8 is the planned fix.
- **Mirror-sensitive.** In a horizontally flipped augmented copy, yaw changes sign. That is correct
  for the flipped image; it is also why copies help, since they show the model both directions.

## 14.5 Getting each student's keypoints: the pose model as the detector

The keypoints must end up on our tracks, which carry the student identity and the labels. There are
two ways, both in the code:

- **The pose model is the person detector** (experiment 09's setting, head-pose `mode: "tracks"`).
  The pose model finds people like any detector, so it replaces YOLO11n. The detector returns each
  person's keypoints with the box (`src/detection/detector.py`); the tracker reports which detection
  updated each track in this frame (BoxMOT's `det_index`) and hands that detection's keypoints to the
  track (`src/tracking/tracker.py`). One network per frame instead of two, and the pairing is exact:
  a track's keypoints are those of the box it was updated with.
- **A second model next to the detector** (`mode: "frame"` or `"crops"`, used by the pilot). The pose
  model runs on its own and its poses are paired with the tracks one-to-one by box overlap (the
  Hungarian assignment on IoU, at least 0.3).

**Why the first, and what it costs.** It saves a network per frame (about 31 ms, about 15% of the
frame pass) and removes the matching step. But the detector feeds everything after it: detection
threshold and tracker were tuned for YOLO11n (D2 in experiment 02, T2 in experiment 03). So the switch is validated
the same way, from data, before any extraction (experiment 09 §1):
- **D3**, the augmented detector scan for the pose model, with a stricter criterion than D2: here a
  missed student is worse than a spurious box, so no student may lose more than 1 point of recall
  against the confidence floor. Its result sets `detector.conf` through `configs/derived_vfoa.py`.
- **T3**, T2's tracker search re-run with the pose model at D3's threshold. Not just a check of T2's
  settings: BoT-SORT ignores every box at or below its own `track_low_thresh` (T2's: 0.1745), which
  would throw away everything D3 keeps below that, so the tracker's thresholds are re-tuned with
  their ranges widened down to the floor. Compared with T2 on IDF1 (0.930) and coverage (0.958).

## 14.6 Input resolution: the pilot

The pose model normally sees the frame at 640 px, which halves the 1280×720 frame: a 40 px head
becomes 20 px. Three inputs were compared on 120 frames from three sessions, on the tracks
experiment 06 cached (`scripts/analysis/head_pose_pilot.py`; table in experiment 09 §3):
- the full frame at 640 (the default);
- the full frame at 1280 (no downscaling);
- upper-body crops of each tracked student, upscaled to 256.

Each with the nano and the small model.

**Choice: the small model on the full frame at 1280.**
- A pose is matched for 99% of tracked students per frame, and yaw is defined for 97%.
- 27 ms per frame, about 15% on top of the frame pass (about 190 ms per frame).
- The crops cost twice as much and leave yaw undefined more often: cutting the student out loses
  the shoulders and the surrounding context the model relies on.
- On overlays the signals behave: turned heads read yaw ±0.3–0.5, students facing forward about 0,
  students looking down at phones or notes a low or negative head height.

## 14.7 Window features

Per student and window (9 columns, all ratios or shares, so comparable across seats):

| Feature | Why |
|---|---|
| `hp.yaw.mean` | where the head mostly points in the window |
| `hp.yaw.std` | how much it moves: scanning the room, turning to a neighbour |
| `hp.yaw.abs` | how far from the camera's axis, either side: turned away |
| `hp.head_height.mean`, `.std` | head up or down, and how much it moves |
| `hp.head_low` | share of frames with the head clearly down (head height below 0.1, about the lowest fifth of frames in the pilot; a first threshold of 0.6 marked almost every frame, so the smoke test caught it before extraction) |
| `hp.face_conf` | how visible the face is |
| `hp.ear_asym` | a second, keypoint-free view of turning |
| `hp.matched` | how often a pose was found: occlusion and turning away |

The window statistics are deliberately simple (mean, SD, a share), the same as the other families:
the experiment asks whether the head carries signal, not how best to summarise it.

## 14.8 Class-relative head direction (planned)

Camera-relative yaw mixes two things: where the student looks, and where they sit. In one window,
though, most of a class looks the same way (at the board, at the teacher, down at a worksheet). So
the class itself is a reference:
- the student's yaw and head height minus the class median in the same window;
- the share of the class whose head points the same way.

This removes the seat effect without knowing the room's layout, and it is the head-rotation
synchrony of Raca et al. It is the first, small step of the class-relative features in guide 15,
which become worth building once head direction exists.

## 14.9 Cost, caching and augmentation

- **The unit reads pixels** (`uses_frames = True`), so its config is part of the frame-pass cache key
  (`src/core/pipeline.py`, `tracking_hash`). Adding it means a new frame pass for every recording:
  detection and tracking re-run with it, unchanged.
- **The re-extraction covers 4 augmented copies.** Head direction depends on camera angle, lighting
  and seat, exactly what the copies vary (colour, noise, a horizontal flip, cut-outs). One copy did
  nothing for the old features (experiment 06), which carried little signal to protect; experiment
  09 tests the dose again (0, 1, 4 copies).
- **The weights** are `models/yolo11s-pose.pt` (the README's setup section has the download line).
  The model loads on first use, so building the pipeline and the tests need no weights.

## 14.10 Limits

- **No focus targets.** "Looking at the board" or "looking at the laptop" needs each camera's room
  layout. Head direction is the input such a mapping would use; the mapping is not built.
- **Proxies, not angles** (§14.4), and camera-relative ones until §14.8.
- **Small heads.** At 35–50 px, keypoints are noisy, especially at the back of the room; the window
  averages over about 20 frames.
- **Everyone is scored.** Teachers or assistants who are tracked get head-pose values too; in
  training they carry no labels and are skipped, as for every other feature.

## 14.11 References

Checked against the publisher or an index record on 18 November 2026.
- Goldberg, P., Sümer, Ö., Stürmer, K., Wagner, W., Göllner, R., Gerjets, P., Kasneci, E. and
  Trautwein, U. Attentive or not? Toward a machine learning approach to assessing students' visible
  engagement in classroom instruction. *Educational Psychology Review*, 2019,
  doi:10.1007/s10648-019-09514-z.
- Raca, M., Tormey, R. and Dillenbourg, P. Sleepers' lag: study on motion and attention. *Journal of
  Learning Analytics* 3(2), 2016, doi:10.18608/jla.2016.32.12.
- Hempel, T., Abdelrahman, A. A. and Al-Hamadi, A. 6D rotation representation for unconstrained head
  pose estimation. *ICIP 2022*, 2496–2500, doi:10.1109/ICIP46576.2022.9897219.
- Lin, T.-Y., Maire, M., Belongie, S., et al. Microsoft COCO: common objects in context. *ECCV 2014*,
  arXiv:1405.0312. (The 17-keypoint person annotation used by the pose model was added to COCO later.)

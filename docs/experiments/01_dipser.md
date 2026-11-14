# DIPSER: dataset work and manual annotation

DIPSER is the classroom dataset this project is built on. This document records how we use it,
the manual annotation we added and why, the camera mapping we reconstructed, and the data issues
found along the way.

## 1. What DIPSER provides

DIPSER records university students across three groups and nine classroom activities
("experiments"). Raw data, as downloaded:

```
data/DIPSER/DIPSEER/group_XX/experiment_YY/
├── context_cam_01.zip ... context_cam_06.zip   room cameras: images/HH_MM_SS_micro.png, ~5 fps
└── subjects/
    ├── metadata.json                            per-student demographics
    └── subject_NN.zip                           labels/labeler_01..05.json, self_labeling.json
```

- **Context cameras** are fixed room cameras (front and rear of the room). Frames are PNG files
  named by wall-clock time; there is no video container and no frame index.
- **Labels are per student (subject).** Each student's attention (1–5) and emotion were rated by
  several external labelers and by the student themselves, as lists of change events with
  wall-clock timestamps.

Scope for Deliverable III: groups 01–03, experiments 01–06 (grid layout), context cameras 01–03
(the front-facing views). That is 54 possible session-cameras, of which **52 exist**:
`group_02/experiment_05` only has `context_cam_01` among the front cameras.

## 2. Why we annotated the context cameras ourselves

### The mismatch

DIPSER's per-student labelling is designed around **per-student capture**: each subject is
recorded individually, so per-student features can be extracted from that student's own stream.
That is a sensible research set-up, but it is not our deployment target. A system that needs a
camera per student is expensive to install, intrusive, and inaccessible to the ordinary classrooms
this project is meant for. Our pipeline uses only the few **room-level context cameras** a real
classroom could plausibly have.

That leaves a gap. The labels say "subject_07 had attention 3 from 10:41:17". The context-camera
footage shows a room of people with no identities. Nothing in DIPSER says *which person in a
context-camera frame is subject_07*.

### Why we need the link

The pipeline itself never needs identities: at inference it works on anonymous tracks and only
outputs classroom-level aggregates. But development needs the link in three places:

1. **Window labels**: to know which students' labels describe a given camera, we need to know
   which subjects that camera can see (`labels.students: "camera"`).
2. **Tracking evaluation**: to score a tracker, we need ground truth for which track is which
   student ([BoxMOT document](02_boxmot.md), §4).
3. **Per-student analysis**, such as the optional student-influence gate (§8.4 of the design),
   which needs per-student labels linked to tracks.

### What we annotated

For every in-scope session and front camera, one **static bounding box per visible subject**,
drawn on a sampled frame. Students sit in fixed seats for a whole session, so a single box per
student per camera locates them for the entire recording.

Stored in `data/DIPSER/abi_labels/<group>/<experiment>.json`:

```json
{
  "context_cam_01": {
    "subject_01": {"x": 1024, "y": 284, "w": 183, "h": 208},
    "subject_02": {"x": 820,  "y": 296, "w": 178, "h": 189}
  }
}
```

Coordinates are pixels in the 1280×720 frame; `x, y` is the top-left corner.

The tool was `scripts/dipser/annotation_helper.py`: it opens a random frame from a camera ZIP,
draws the existing boxes for that camera with their subject ids, and prints a box in the JSON
format above when you right-click two corners. Subject identities were established by comparing
views across cameras and experiments of the same group.

### Coverage

| Group | Experiments | Boxes per camera (cam 01 / 02 / 03) | Subjects with labels covered by ≥1 camera |
|---|---|---|---|
| 01 | 01–05 | 13 / 11 / 14 | 20 / 20 |
| 01 | 06 | 12 / 14 / 11 | 19 / 19 |
| 02 | 01–04 | 13 / 11 / 14 | all |
| 02 | 05 | 13 / — / — | 11 / 18 (only cam 01 exists) |
| 02 | 06 | 12 / 14 / 11 | 18 / 19 |
| 03 | 01–06 | 8 / 13 / 11 | 15 / 15 |

Notes:

- Every labelled student is visible in at least one front camera, except in the two group 02
  sessions above. Cameras overlap, so most students appear in two views.
- A few annotated subjects have **no label archive** in that experiment (for example `subject_10`
  in group 02 experiments 01–02, `subject_13`/`subject_14` in experiments 04–05, `subject_04` in
  group 03 experiment 06). They are real people in the room whose labels DIPSER does not provide.
  The label pipeline simply has no labels for them; their boxes still help tracking evaluation.
- Some files contain out-of-scope entries: `context_cam_04` (20 boxes in group 01 experiment 01,
  empty elsewhere), and group 01 experiments 07–08 (ring layout). The pipeline ignores them.

## 3. Linking subjects to tracks (SID–TID association)

The tracker gives each person a track id (TID) that means nothing outside one run. To link a
track to a subject id (SID), we compare the track's boxes with the static annotated boxes.

### Version 1: nearest neighbour (`notebooks/dipser_tracking.ipynb`)

The first implementation, in the tracking notebook, averaged each track's box over its lifetime
and assigned it to the subject whose annotated box was nearest by Euclidean distance between the
four corner coordinates. Each track was assigned independently.

The flaw: **there was no distance threshold**, so every track got a subject, however far it was
from any annotated box. Teaching assistants and teachers walking around the room were assigned to
whichever seated student was closest on average. Their tracks then counted as that student when
scoring the tracker or linking labels.

### Version 2: per-frame matching with an overlap floor (`src/tracking/association.py`)

The pipeline's implementation (design §4) matches per frame and requires real overlap:

1. **Per frame**: Hungarian assignment between that frame's track boxes and the subject boxes,
   maximising IoU (intersection over union), so one subject can match at most one track per frame.
2. **Distance threshold**: a pair only counts if IoU ≥ `association.min_iou` (0.3). This is the
   distance threshold that was missing (IoU distance = 1 − IoU ≤ 0.7). A box nowhere near any
   seat matches nothing.
3. **Majority vote**: each track gets the subject it matched in most frames.
4. **Passing-person guard**: the winning subject must account for at least
   `association.min_match_share` (50%) of *all* the track's frames. A TA who walks past a seated
   student overlaps that student's box for a few frames, not for most of their track, so they stay
   unassociated.

Step 4 was added after reviewing the notebook's behaviour; a unit test covers it
(`test_walker_passing_a_student_stays_unassociated`).

Tie-breaking is deterministic and documented in the code: equal IoUs resolve by index order, and
vote ties go to the lexicographically first subject id.

## 4. Camera positions: `cameras.yml`

### Why it exists

DIPSER names cameras by number (`context_cam_01` …), not by position. We wrote
`data/DIPSER/abi_labels/cameras.yml` mapping every camera number, per group and per day, to its
physical position (`front_east`, `front_west`, `front_middle`, `rear_*`), determined by looking
at the footage (`scripts/dipser/camera_previews.py` extracts the first frame of every camera for
this purpose).

### The finding: camera numbers are not stable positions

Within the in-scope front cameras, **`context_cam_02` and `context_cam_03` swap positions between
groups and days**:

| Group | Experiments (day) | cam 01 | cam 02 | cam 03 |
|---|---|---|---|---|
| 01 | 01–05 (days 1–2) | front_east | **front_west** | **front_middle** |
| 01 | 06–09 (day 3) | front_east | **front_middle** | **front_west** |
| 02 | 01–05 (days 1–2) | front_east | **front_west** | **front_middle** |
| 02 | 06–09 (day 3) | front_east | **front_middle** | **front_west** |
| 03 | 01–09 (days 1–2) | front_east | **front_middle** | **front_west** |

`context_cam_01` is always front_east. The swap is visible independently in the annotation box
counts: in group 01, camera 02 has 11 boxes and camera 03 has 14 in experiments 01–05, but 14 and
11 in experiment 06.

### Consequences

- **"Camera 02" is not a consistent viewpoint** across sessions. Any feature that depends on the
  viewpoint, such as the spatial occupancy heatmap in Vec (`occ_ij`), means something different
  for front_west and front_middle views.
- **Per-camera-number analyses are confounded.** Grouping results by `context_cam_02` mixes two
  physical views; group results by position (via `cameras.yml`) instead.
- **Leave-one-group-out folds** hold out group 03, whose camera 02 is always front_middle, while
  training mostly sees camera 02 as front_west. Fold-to-fold differences may partly reflect this.
- **Unaffected**: the manual annotations are per session and camera, so they are always in the
  right frame of reference, and tracking is per camera.

The pipeline does not read `cameras.yml` yet. Using camera *position* rather than number is the
natural fix if a viewpoint-dependent feature turns out to matter.


## 5. The attention labels

Each labeler file is a list of change events:

```json
[{"datetime": "10:40:43:047895", "attention": "3", "emotion": "9"},
 {"datetime": "10:45:25:006335", "emotion": "1"}]
```

Only entries with an `attention` key change attention, and a value holds until the next one.
Values are strings. Our parser (`src/labels/attention.py`) casts them to integers and **drops
values outside 1–5**: one self-report (`group_01/experiment_03/subject_18`) contains attention
`7`, probably an emotion code typed into the attention field.

Labels and frames share the same wall clock. Across all 52 sessions, labels start between 0.0 and
2.2 s before the first camera frame, so no offset correction is applied (`labels.offset_s = 0`).

The label analysis (distribution, labeler agreement, threshold choice) will get its own document.

## 6. Data-quality issues found

| Issue | Where | Effect | Handling |
|---|---|---|---|
| Frame rate ~5 fps, irregular; 9–45 gaps per session of up to 6 s | all sessions | naive frame counting misstates time | fps measured from the median interval; gaps measured in missing frames |
| 314 empty (0-byte) PNGs, 28% of the session | `group_03/experiment_03/context_cam_03` | crashed decoding; 28% hole in that session | zero-byte files dropped at indexing |
| Only one front camera | `group_02/experiment_05` | 52 instead of 54 session-cameras | — |
| Attention value 7 | `group_01/experiment_03/subject_18`, self-report | out of scale | dropped |
| Annotated subjects without label archives | several group 02/03 sessions | no labels for some visible people | they count for tracking, not labels |
| `experiments.tsv` lists experiment 06 device use as "required"; the design document says "optional" | `abi_labels/experiments.tsv` | none (not used by the pipeline) | to reconcile |
| `cameras.yml` structure issues (mis-indented group 02 day 3 block, duplicate `context_cam_05` key, camera 03 mapped to front_east in group 01 day 3) | `abi_labels/cameras.yml` | group 02 day 3 unreadable | fixed |

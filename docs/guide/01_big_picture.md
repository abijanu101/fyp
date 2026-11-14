# Chapter 1 — The big picture

Before looking at any code, it helps to hold the whole system in your head at once. This chapter
describes what goes in, what comes out, and the stages in between, without any implementation
detail. Every later chapter zooms into one part of this picture.

## 1.1 The problem in one paragraph

A classroom is filmed by fixed cameras. We want a single curve over time that says how engaged
the class is, like YouTube's "audience retention" graph, without ever producing anything about an
individual student. The curve should also say which kind of activity was going on (a lecture, a
quiz, a design task), because what "engaged" looks like depends on the activity.

## 1.2 What goes in

The input is DIPSER, a dataset of classroom recordings. On disk it looks like this:

```
data/DIPSER/
├── DIPSEER/                          raw data (note the extra E: that is the folder's real name)
│   └── group_01/
│       └── experiment_03/
│           ├── context_cam_01.zip    one camera: images/HH_MM_SS_micro.png, ~5 fps
│           ├── context_cam_02.zip
│           ├── ...
│           └── subjects/
│               ├── metadata.json     age/gender per student (unused)
│               └── subject_07.zip    labels/labeler_01.json ... self_labeling.json
└── abi_labels/                       our own manual annotations
    ├── experiments.tsv               experiment -> layout, device use, activity
    └── group_01/experiment_03.json   one static box per student per camera
```

Three facts about this data shape the whole design:

1. **Frames are PNG files inside ZIP archives**, named by wall-clock time (`10_40_43_442466.png`
   means 10:40:43.442466). There is no video file and no frame number; time is the only index.
2. **Engagement labels are per student, not per camera.** Several labelers rated each student's
   attention on a 1–5 scale. The labels are lists of *change events*: a value holds until the
   labeler changed it.
3. **The experiment folder tells us the activity.** Experiments 01–06 are six different
   activities (news reading, QR scanning, lecture, Canva design, online quiz, student explains).
   That gives a free, exact label for "which activity is running".

Deliverable III uses groups 01–03, experiments 01–06 and cameras 01–03: at most 54 session-cameras,
of which 52 exist on disk.

## 1.3 What comes out

For one camera ZIP, the pipeline writes three files:

- `retention_graph.json`: one record per time window with the predicted activity, an engagement
  probability, an engaged yes/no, and the number of visible people. Nothing about individuals.
- `retention_graph.png`: the smoothed engagement curve, shaded by predicted activity.
- `manifest.json`: everything needed to reproduce the run: frame rate, gaps in the footage, the
  full configuration, and a hash of that configuration.

## 1.4 The journey: nine stages

The design document numbers the stages 1–9. Here they are as a flow, with the package that
implements each:

```
 camera ZIP
     │
     ▼
 1  Ingestion            src/ingestion     decode frames in time order, measure fps and gaps
     │  Frame (time, image)
     ▼
 2  Person detection     src/detection     YOLO finds every person: boxes + confidences
     │  Detections
     ▼
 3  Tracking             src/tracking      BoT-SORT links boxes over time into identities (TIDs)
     │  TrackState per person per frame
     ▼
 4  Tracklets            src/tracking      per-person time series; drop short or broken ones
     │  Tracklet
     ▼
 5  BFMU features        src/bfmu          per-person features: motion, optical flow, behaviour
     │  per-track feature dict per window
     ▼
 6  Window aggregation   src/aggregation   average everyone into one classroom vector, "Vec"
     │  Vec per window                        (+ CFMU cluster features, currently empty)
     ▼
 7  Context gate (SPU)   src/spu           which of the 6 activities is this? (probabilities)
     │  g = [p01 .. p06]
     ▼
 8  Expert mixture       src/experts       one engagement model per activity, weighted by g
     │  engagement probability per window
     ▼
 9  Retention graph      src/output        JSON + PNG + manifest
```

Two side branches only exist during development:

- **Labels** (`src/labels`) turn the labelers' change events into one engaged / not-engaged label
  per window, to train and evaluate the experts.
- **Evaluation** (`src/evaluation`) scores tracking against our manual boxes, the gate against
  the folder structure, and the experts against the labels.

## 1.5 The key idea: windows

Most of the pipeline works per **window**: a fixed slice of time, 8 seconds by default, with 50%
overlap between consecutive windows. A 5-minute session gives about 73 windows.

Everything before stage 6 is about individual people over time. Stage 6 is where people
disappear: all visible people in a window are summarised into one fixed-length vector, *Vec*.
From stage 6 on, the unit of analysis is "the classroom during this window", and no stage after
it ever sees an individual. That is how the privacy guarantee works structurally rather than by
promise: the output stages simply never receive per-person data.

## 1.6 Two kinds of learned component

It is worth being precise about what is "learned" here, because the design is deliberate about it:

- **Pretrained perception models, used frozen**: the YOLO person detector, and the SCB behaviour
  detector (a YOLOv7 trained on classroom images by its authors). We never train these.
- **Classical models trained on DIPSER**: the context gate (Random Forest by default) and the
  engagement experts (logistic regression by default), from scikit-learn.

There is no end-to-end deep network trained on our data. With only 52 session-cameras and three
groups, a deep model would mostly learn the rooms and the students' clothes.

## 1.7 Two passes, not nine stages

The stages above describe *what* happens. For performance, the code groups them into two passes:

```
 ┌──────────── frame pass (slow: minutes per session) ────────────┐
 │ decode → detect → track → pixel-level features → scene features │──► SessionTrack  (cached to disk)
 └─────────────────────────────────────────────────────────────────┘
 ┌──────────── feature pass (fast: about a second per session) ───┐
 │ filter tracklets → window features → Vec → scene windows        │──► SessionFeatures (cached to CSV)
 └─────────────────────────────────────────────────────────────────┘
 ┌──────────── model pass ─────────────────────────────────────────┐
 │ gate → experts → retention graph                                │──► retention_graph.json/.png
 └─────────────────────────────────────────────────────────────────┘
```

The frame pass touches every pixel of every frame and runs neural networks; it is the expensive
part. The feature pass only does arithmetic on the frame pass's output. Splitting them means you
can change the window length, or a feature threshold, and recompute features in seconds instead
of re-running detection for hours. Chapter 8 explains how the caches are keyed so this is safe.

## 1.8 Training versus inference

The same code produces features for training and for inference. That is a deliberate property:
if training used one feature function and inference another, any small difference between them
would silently degrade the deployed model.

- **Training** (`scripts/train_models.py`): run both passes on all 52 sessions, attach window
  labels, train the gate and experts with leave-one-group-out cross-validation, save the models.
- **Inference** (`scripts/run_pipeline.py infer`): run both passes on one ZIP, load the saved
  models, run the model pass, write the graph.

## 1.9 Where to go next

Chapter 2 covers the few Python features the code relies on heavily. If you are comfortable with
dataclasses, abstract base classes, decorators and `python -m`, skim it and move to Chapter 3,
which is the heart of this guide: how the configuration file turns names into running objects.

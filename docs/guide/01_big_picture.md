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

- `retention_graph.json`: one record per time window with the predicted activity, the class's
  predicted attention rating (1–5), each labeler model's rating, an engaged yes/no (rating ≥
  `labels.tau_att`), and the number of visible people. Nothing about individuals. Models trained on
  the window-grid paths write an engagement probability instead (Chapter 7, §7.8).
- `retention_graph.png`: each labeler model's smoothed curve (experts blue, self orange) and the
  final curve in black, on the 1–5 scale, shaded by predicted activity.
- `manifest.json`: everything needed to reproduce the run: frame rate, gaps in the footage, the
  full configuration, and a hash of that configuration.

## 1.4 The journey: nine stages

The design document numbers the stages 1–9. Here they are in order, with the package that
implements each and, in brackets, the config key that controls it (section 1.7 shows how they are
actually grouped and run):

```
 camera ZIP
     │
     ▼
 1  Ingestion            src/ingestion     decode frames in time order, measure fps and gaps
     │  Frame (time, image)                 [ingestion]
     ▼
 2  Person detection     src/detection     YOLO finds every person: boxes + confidences
     │  Detections                          [detector]
     ▼
 3  Tracking             src/tracking      BoT-SORT links boxes over time into identities (TIDs)
     │  TrackState per person per frame     [tracker]
     ▼
 4  Tracklets            src/tracking      per-person time series; drop short or broken ones
     │  Tracklet                            [tracklets]
     ▼
 5  BFMU features        src/bfmu          per-person features: motion, optical flow, behaviour
     │  per-student features per window     [bfmu, window]
     ▼
 6  Window aggregation   src/aggregation   per window: each visible student's features, and
     │  per-student rows + Vec              their summary over the class, "Vec" [aggregation]
     │  (+ CFMU cluster features, currently empty) [cfmu]
     ▼
 7  Context gate (SPU)   src/spu           which of the 6 activities is this? (probabilities),
     │  g = [p01 .. p06]                    from scene features of the whole frame [scene, gate]
     ▼
 8  Expert mixture       src/experts       one mixture per labeler (labeler_01–05, self), each
     │  rating curves per window            weighting its per-activity experts by g; scores each
     │                                      student, averages per window into one curve per
     │                                      labeler, plus their median or majority
     │                                      [expert, mixture, engagement.labelers, output]
     ▼
 9  Retention graph      src/output        JSON + PNG + manifest, window-level only
```

Every bracketed key is a section of `configs/deliverable3.json`. Swapping an implementation (a
different tracker, another feature family, another classifier) or a setting (window length, label
thresholds, early or late aggregation) is a config edit, not a code change. Chapter 3 explains how
names in the config become objects, and Chapter 9 how to use that to run experiments.

Two side branches only exist during development:

- **Labels** (`src/labels`) turn the labelers' change events into training values. By default
  each rating entry gets a window centred on it and a soft value from that labeler's own ratings,
  and evaluation scores every held-out entry (Chapter 7, §7.9) [labels]. The window-grid paths use
  engaged / not-engaged labels per student or per window instead.
- **Evaluation** (`src/evaluation`) scores tracking against our manual boxes, the gate against
  the folder structure, and the experts against the labels.

## 1.5 The key idea: windows

Most of the pipeline works per **window**: a fixed slice of time (`window.length_s`, 4 seconds in
the current config), with 50% overlap between consecutive windows. A 5-minute session gives about
149 windows.

Everything before stage 6 is about individual people over time. Stage 6 cuts each person's time
series into windows: per window, every visible student gets one row of features, and a summary of
all of them, the fixed-length vector *Vec*, describes the classroom as a whole.

The default per-labeler models always score each visible student and average per window. In the
window-grid paths, how the engagement experts (stage 8) see a window is a configuration choice
(`engagement.aggregate`; Chapter 7, §7.5 works through an example):

- **late aggregation** (the configured value): the experts see each visible student's own features (plus
  Vec as context), predict per student, and the window's score is the average over its students;
- **early aggregation** (the design as written): the experts see only Vec, one row per window.

Either way, the output stages only ever receive window-level arrays: the retention graph is built
from one context prediction and window-level ratings (one per labeler model, plus the final one)
per window. That is how the privacy
guarantee works structurally rather than by promise. Per-student predictions in late aggregation
exist only inside the window computation and are never written out. At inference nothing per
student is written at all. During development, `run_pipeline extract` does cache tracks (`.npz`)
and per-student feature rows (`.tracks.csv`) for training, but these never reach the retention
graph.

## 1.6 Two kinds of learned component

It is worth being precise about what is "learned" here, because the design is deliberate about it:

- **Pretrained perception models, used frozen**: the YOLO person detector, and the SCB behaviour
  detector (a YOLOv7 trained on classroom images by its authors). We never train these.
- **Classical models trained on DIPSER**: the context gate (Random Forest by default) and the
  engagement experts (logistic regression by default), from scikit-learn.

There is no end-to-end deep network trained on our data. With only 52 session-cameras and three
groups, a deep model would mostly learn the rooms and the students' clothes.

## 1.7 Two passes, not nine stages

The stages above describe *what* happens. For performance, the code groups them into passes, and
inside the passes the data runs in two lanes: a **track lane** (people, stages 2–6) that feeds the
experts, and a **scene lane** (whole frames) that feeds the context gate. The lanes meet only at the
expert mixture, where the gate's probabilities weight the experts.

```
                 track lane (people)                         scene lane (whole frame)
 ┌──────────── frame pass (slow: minutes per session) ──────────────────────────────────────┐
 │ decode ──┬──► detect → track → pixel-level features                                      │
 │          └────────────────────────────────────────────► scene features per frame        │
 │                                                    ──► SessionTrack (cached to disk)     │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
 ┌──────────── feature pass (fast: about a second per session) ─────────────────────────────┐
 │ filter tracklets → per-student window features → Vec       scene features per window    │
 │                    (per-student rows kept)                                               │
 │                                                    ──► SessionFeatures (cached to CSV)   │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
 ┌──────────── model pass ──────────────────────────────────────────────────────────────────┐
 │ per-student rows (late) or Vec (early)                     scene features → gate → g     │
 │                  └──────────────► experts, weighted by g ◄──────────────────┘            │
 │                                         └──► retention graph (.json, .png)               │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
```

Because the gate has its own lane, tracking errors cannot mislead the activity prediction. A
config option (`gate_uses_vec`) can also hand Vec across to the gate, to test whether track
features help it.

The frame pass touches every pixel of every frame and runs neural networks; it is the expensive
part. The feature pass only does arithmetic on the frame pass's output. Splitting them means you
can change the window length, or a feature threshold, and recompute features in seconds instead
of re-running detection for hours. Chapter 8 explains how the caches are keyed so this is safe.

## 1.8 Training versus inference

The same code produces features for training and for inference. That is a deliberate property:
if training used one feature function and inference another, any small difference between them
would silently degrade the deployed model.

- **Training**: `scripts/run_pipeline.py extract` runs both passes on all 52 sessions and caches
  the features; `scripts/train_models.py` reads that cache, attaches labels, trains the gate and
  experts with leave-one-group-out cross-validation, and saves the models.
- **Inference** (`scripts/run_pipeline.py infer`): run both passes on one ZIP, load the saved
  models, run the model pass, write the graph.

## 1.9 Where to go next

Chapter 2 covers the few Python features the code relies on heavily. If you are comfortable with
dataclasses, abstract base classes, decorators and `python -m`, skim it and move to Chapter 3,
which is the heart of this guide: how the configuration file turns names into running objects.

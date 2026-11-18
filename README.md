# FYP-1: classroom retention graph from DIPSER

Offline pipeline that turns DIPSER context-camera footage into a time-indexed classroom engagement
("retention") graph, with no per-student identity in the output. Implements Technical Design v3
(Deliverable III): detection → tracking → BFMU features → window vector (Vec) → SPU context gate →
one gated mixture of per-experiment experts per labeler → retention graph of predicted 1–5 ratings
(one curve per labeler model, and their median).

New to the code? Read the contributor's guide in [`docs/guide/`](docs/guide/README.md): it builds up
from the big picture and the registry pattern to every stage, with recipes for common changes.

## Setup

```
conda env create -f abi_local.yml        # env "fyp-local": Python 3.13, torch 2.9.0+cu126, boxmot, ultralytics
conda activate fyp-local
```

### Data

DIPSER is expected under `data/DIPSER/DIPSEER/group_XX/experiment_YY/{context_cam_ZZ.zip, subjects/}`.
Our manual box annotations (`data/DIPSER/abi_labels/`) are in the repository. The dataset itself
comes from its FTP server, with the credentials the dataset authors issue, in a `.env` file at the
repository root (`FTP_USER=...`, `FTP_PASSWORD=...`; git-ignored):

```
python scripts/downloading/dipser.py                    # context cameras + per-student labels (filters at the top of the file)
python scripts/downloading/validate_dipser_remote.py    # optional: check the local copy against the server
```

### Model weights (`models/`, git-ignored)

| File | What | How to get it |
|---|---|---|
| `models/yolo11n.pt` | person detector (`detector.weights`) | Ultralytics release asset: `curl -L -o models/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt` |
| `models/yolo11s-pose.pt` | head-pose keypoints (`bfmu` `head_pose`, experiment 09; `yolo11n-pose.pt` for the pilot's nano variant) | Ultralytics release asset: `curl -L -o models/yolo11s-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt` |
| `models/osnet_x0_25_msmt17.pt` | ReID appearance model of the tracker (BoT-SORT with `use_embeddings`, the tuned T2 setting) | BoxMOT's default ReID model, downloaded by BoxMOT itself the first time a tracker with ReID runs from the repo root |

### SCB behaviour checkpoints (`data/SCB-ST-Dataset4/`, git-ignored)

The behaviour models (`bfmu` `scb`) are YOLOv7 checkpoints from the SCB dataset on Hugging Face, and
need the YOLOv7 model code to load:

```
python scripts/downloading/scb.py                       # wintonYF/SCB-Dataset -> data/SCB-ST-Dataset4/ (large; resumable)
git clone https://github.com/WongKinYiu/yolov7 lib/yolov7 && git -C lib/yolov7 checkout a207844
```

SCB checkpoints are third-party pickles; `src/bfmu/yolov7.py` loads them with torch's weights-only
unpickler restricted to an allowlist of model classes, and refuses anything else.

### Trained models (`outputs/models/`)

Trained bundles (`outputs/models/<config hash>.joblib`, 15–30 MB each) are rebuilt from a config with
`python -m scripts.train_models --config <config>`: each experiment document names its config
(`docs/experiments/NN_name.json`) and the result hash, and the same config gives the same hash.
Training needs the feature cache (`outputs/cache/`, git-ignored), which `scripts.run_pipeline
extract` rebuilds from the data above. A download for the bundles themselves is not set up yet.

## Layout

```
src/
  core/         types (stage contracts), registry, config + hashing, pipeline, track store
  datasets/     DipserIndex: in-scope session listing, annotations, label locations
  ingestion/    streaming ZIP frame source, timing audit (fps, gaps)
  detection/    shared YOLO person detector
  tracking/     BoT-SORT / StrongSORT (BoxMOT), tracklets + gap filter, SID-TID association
  bfmu/         per-track features: kalman_geometry, residual_flow, scb (SCB YOLOv7 behaviour models)
  cfmu/         cluster features (null in Deliverable III) + spread maths
  aggregation/  windows, Vec assembly
  spu/          scene features, context gate, oracle/classified context, LOGO split
  experts/      engagement experts, gated mixture, per-labeler models (the default)
  labels/       attention step functions, interpolation, rating entries, window labels
  output/       retention graph JSON/PNG, manifest
  evaluation/   metrics, tracking quality (IDF1 etc.), Krippendorff alpha, stage timer
  tuning/       seeded random search, nested leave-one-group-out search
configs/deliverable3.json   every tunable value; stages are picked by registry name
scripts/                    entry points (below); scripts/data_visualization/ holds dataset inspection tools
tests/                      unit tests (stdlib unittest, no GPU needed)
```

Swap a stage by registering a class in its package and naming it in the config. Stages exchange
only the dataclasses in `src/core/types.py`.

## Workflow

Run from the repo root.

```
python -m unittest discover -s tests -t .              # tests

python -m scripts.tuning.tune_tracker [--profile full]        # detector conf + tracker params vs manual boxes
python -m scripts.tuning.scan_detector                        # does the detector threshold drop students? (cached detections)
python -m scripts.run_pipeline extract [--reuse]       # features for all in-scope sessions (+ augmented copies if augmentation.copies > 0)
python -m scripts.data_visualization.augment_samples                      # augmentation examples and extremes -> outputs/augmentation/
python -m scripts.train_models                         # LOGO evaluation + final models
python -m scripts.reporting.plot_results                         # held-out engagement curves vs labels, per session
python -m scripts.data_visualization.output_viewer group_01 experiment_03 context_cam_01   # play a session with its predictions
python -m scripts.tuning.tune_models [--component gate|expert]   # nested-LOGO hyperparameter search
python -m scripts.tuning.benchmark [--name trackers]            # per-stage timing per tracker variant (resumable)
python -m scripts.run_pipeline infer --zip <camera.zip> --models outputs/models/<hash>.joblib --out <dir>
```

The contributor's guide (`docs/guide/`) explains how all of this works; Chapter 9 covers running
experiments and every config key. `docs/experiments/` records what was tried and found;
`docs/progress/` holds dated progress updates (latest: `docs/progress/2026-11-17.md`).
`docs/misc/benchmarks.md` surveys published results on DIPSER; `docs/misc/targets.md` lays out the
possible attention targets (unit × aggregation × self-report) and where each work sits.
`docs/misc/proposed_metrics.md` holds metrics proposed for review but not computed (currently
`retention_r`, removed from the code).

Any config value can be overridden per run with `--set key.path=<json>`, e.g.
`--set window.length_s=10 labels.tau_att=3`.

Dataset tools live in `scripts/data_visualization/`. These use the pipeline code (run with `-m`):

```
python -m scripts.data_visualization.inspect_labels    # label distribution, agreement, threshold grid
python -m scripts.data_visualization.check_scb         # SCB behaviour detections on a DIPSER sample (§11 gate)
python -m scripts.data_visualization.augment_viewer group_01 experiment_03 context_cam_02   # preview augmentations + transformed boxes
python -m scripts.data_visualization.scb_viewer group_01 experiment_04 context_cam_03   # play a camera with SCB boxes, per-class sliders
```

The rest are standalone (OpenCV/matplotlib only, run as files):

```
python scripts/data_visualization/label_viewer.py group_01 experiment_01 context_cam_01   # video + live labels
python scripts/data_visualization/plot_labels.py group_01 experiment_01 03                # one student's labels
python scripts/data_visualization/plot_labels.py --scan                                   # all plots to outputs/label_plots
python scripts/data_visualization/player.py data/DIPSER/DIPSEER                           # multi-camera player
```

## Caches and reproducibility

`outputs/` holds the following; everything except `cache/` (large, regenerable) is shared in git:

| Path | Contents | Keyed by |
|---|---|---|
| `cache/tracks/<hash>/` | frame pass per session: raw tracklets, pixel signals, scene features | detector, tracker, ingestion, scene, pixel-level BFMU |
| `cache/features/<hash>/` | per session: window rows (Vec + scene, `<session>.csv`) and per-student rows (`<session>.tracks.csv`) | the above + tracklets, BFMU, CFMU, window, aggregation |
| `cache/detections/` | detections at a low confidence floor and ReID embeddings, for tracker tuning | detector |
| `results/<hash>.json`, `models/<hash>.joblib` | evaluation and trained models | full config |
| `plots/<hash>/` | held-out engagement curves (`plot_results`) | full config |
| `tuning/` | search results with ready-to-paste config patches; detector-threshold scan | — |

`run_pipeline extract` reads nothing back by default, so its manifests' stage timings are real
runtime. With `--reuse` (development) it skips finished sessions and loads cached tracks: detection
and tracking then re-run only when their settings change, and window, threshold and BFMU changes
take seconds. Tuning scripts always use their caches. Every result carries the SHA-256 of the full
resolved config (excluding `output_dir`, `tuning` and `benchmark`); results from different hashes are never compared.

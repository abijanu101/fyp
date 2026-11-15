# FYP-1: classroom retention graph from DIPSER

Offline pipeline that turns DIPSER context-camera footage into a time-indexed classroom engagement
("retention") graph, with no per-student identity in the output. Implements Technical Design v3
(Deliverable III): detection → tracking → BFMU features → window vector (Vec) → SPU context gate →
mixture of per-experiment engagement experts → retention graph.

New to the code? Read the contributor's guide in [`docs/guide/`](docs/guide/README.md): it builds up
from the big picture and the registry pattern to every stage, with recipes for common changes.

## Setup

```
conda env create -f abi_local.yml        # env "fyp-local": Python 3.13, torch 2.9.0+cu126, boxmot, ultralytics
conda activate fyp-local
```

Data is expected under `data/DIPSER/` (`DIPSEER/group_XX/experiment_YY/{context_cam_ZZ.zip, subjects/}`
plus the manual annotations in `abi_labels/`). Model weights live in `models/` (git-ignored); the SCB
behaviour checkpoints are read from `data/SCB-ST-Dataset4/` and need the YOLOv7 model code:

```
git clone https://github.com/WongKinYiu/yolov7 lib/yolov7 && git -C lib/yolov7 checkout a207844
```

SCB checkpoints are third-party pickles; `src/bfmu/yolov7.py` loads them with torch's weights-only
unpickler restricted to an allowlist of model classes, and refuses anything else.

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
  experts/      engagement experts, gated mixture
  labels/       attention step functions, window labels
  output/       retention graph JSON/PNG, manifest
  evaluation/   metrics, tracking quality (IDF1 etc.), Krippendorff alpha, stage timer
  tuning/       seeded random search, nested leave-one-group-out search
configs/deliverable3.json   every tunable value; stages are picked by registry name
scripts/                    entry points (below); scripts/dipser/ holds dataset inspection tools
tests/                      unit tests (stdlib unittest, no GPU needed)
```

Swap a stage by registering a class in its package and naming it in the config. Stages exchange
only the dataclasses in `src/core/types.py`.

## Workflow

Run from the repo root.

```
python -m unittest discover -s tests -t .              # tests

python -m scripts.tune_tracker [--profile full]        # detector conf + tracker params vs manual boxes
python -m scripts.scan_detector                        # does the detector threshold drop students? (cached detections)
python -m scripts.run_pipeline extract [--reuse]       # features for all in-scope sessions (+ augmented copies if augmentation.copies > 0)
python -m scripts.augment_samples                      # augmentation examples and extremes -> outputs/augmentation/
python -m scripts.train_models                         # LOGO evaluation + final models
python -m scripts.plot_results                         # held-out engagement curves vs labels, per session
python -m scripts.output_viewer group_01 experiment_03 context_cam_01   # play a session with its predictions
python -m scripts.tune_models [--component gate|expert]   # nested-LOGO hyperparameter search
python -m scripts.benchmark [--name trackers]            # per-stage timing per tracker variant (resumable)
python -m scripts.run_pipeline infer --zip <camera.zip> --models outputs/models/<hash>.joblib --out <dir>
```

The contributor's guide (`docs/guide/`) explains how all of this works; Chapter 9 covers running
experiments and every config key. `docs/experiments/` records what was tried and found;
`docs/misc/benchmarks.md` surveys published results on DIPSER.

Any config value can be overridden per run with `--set key.path=<json>`, e.g.
`--set window.length_s=10 labels.tau_att=3`.

Dataset tools live in `scripts/dipser/`. These use the pipeline code (run with `-m`):

```
python -m scripts.dipser.inspect_labels    # label distribution, agreement, threshold grid
python -m scripts.dipser.check_scb         # SCB behaviour detections on a DIPSER sample (§11 gate)
python -m scripts.dipser.augment_viewer group_01 experiment_03 context_cam_02   # preview augmentations + transformed boxes
python -m scripts.dipser.scb_viewer group_01 experiment_04 context_cam_03   # play a camera with SCB boxes, per-class sliders
```

The rest are standalone (OpenCV/matplotlib only, run as files):

```
python scripts/dipser/label_viewer.py group_01 experiment_01 context_cam_01   # video + live labels
python scripts/dipser/plot_labels.py group_01 experiment_01 03                # one student's labels
python scripts/dipser/plot_labels.py --scan                                   # all plots to outputs/label_plots
python scripts/dipser/player.py data/DIPSER/DIPSEER                           # multi-camera player
```

## Caches and reproducibility

`outputs/` holds the following; everything except `cache/` (large, regenerable) is shared in git:

| Path | Contents | Keyed by |
|---|---|---|
| `tracks/<hash>/` | frame pass per session: raw tracklets, pixel signals, scene features | detector, tracker, ingestion, scene, pixel-level BFMU |
| `features/<hash>/` | per session: window rows (Vec + scene, `<session>.csv`) and per-student rows (`<session>.tracks.csv`) | the above + tracklets, BFMU, CFMU, window, aggregation |
| `cache/detections/` | detections at a low confidence floor and ReID embeddings, for tracker tuning | detector |
| `results/<hash>.json`, `models/<hash>.joblib` | evaluation and trained models | full config |
| `plots/<hash>/` | held-out engagement curves (`plot_results`) | full config |
| `tuning/` | search results with ready-to-paste config patches; detector-threshold scan | — |

`run_pipeline extract` reads nothing back by default, so its manifests' stage timings are real
runtime. With `--reuse` (development) it skips finished sessions and loads cached tracks: detection
and tracking then re-run only when their settings change, and window, threshold and BFMU changes
take seconds. Tuning scripts always use their caches. Every result carries the SHA-256 of the full
resolved config (excluding `output_dir`, `tuning` and `benchmark`); results from different hashes are never compared.

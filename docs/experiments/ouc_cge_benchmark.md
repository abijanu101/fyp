# OUC-CGE: a second dataset for the pipeline (benchmark)

**Status (26 November 2026):** data audited, floor and shortcut baselines computed (no GPU needed).
Pipeline adapters written and smoke-tested on 5 clips (CPU, without ReID and SCB). **The pipeline has
not been run on the full dataset yet: no pipeline numbers exist.** The full run is in
[`docs/ouc_cge/RUN_REQUEST.md`](../ouc_cge/RUN_REQUEST.md).

Why a second dataset: DIPSER labels set a low ceiling (experiment 06). OUC-CGE has a different label
(one group-level engagement level per clip, low / mid / high) and a published reference, so it tests
whether the same features carry engagement signal where the labels are cleaner. It also follows the
team's motion base paper (DualEngage), which reports on OUC-CGE.

Structure follows [`docs/misc/benchmarks.md`](../misc/benchmarks.md): the floor, then what others
report, then what can fairly be compared.

## 1. The dataset as downloaded

Lu et al., *A Video Dataset for Classroom Group Engagement Recognition*, Scientific Data 2025.
Videos from OSF (the links are in the download script); split files from the authors' GitHub repo
(`OUC-CGE-main/{train,val,test}.csv`, official per-clip 8:1:1 split). Audit:
`python -m scripts.ouc_cge.audit` → [`outputs/ouc_cge/data/`](../../outputs/ouc_cge/data/)
(`audit.json`, `clips.csv`: one row per clip).

| | train | val | test | total |
|---|---|---|---|---|
| low | 2,396 | 307 | 325 | 3,028 |
| mid | 1,653 | 199 | 189 | 2,041 |
| high | 2,115 | 264 | 257 | 2,636 |
| **total** | 6,164 | 770 | 771 | **7,705** |

- **All 7,705 split clips are on disk** (DualEngage could retrieve about 6,700). No video outside the splits.
- `test.csv` shipped with a placeholder row (`videos/anonymity/a.mp4`) that matches no clip; it is skipped
  (the code skips any row that is not a clip; the local copy was also cleaned, original kept as `test_original.csv`).
- **116 clips do not decode, all of them high** (train 102, val 7, test 7): 112 in the archive
  `high-2636(1001-2000).7z`, whose files report 4 frames (about 0.13 s) despite 1.4–8 MB each, and 4 in
  `high-2636(2501-2636)` that do not open ("moov atom not found"). The extracted bytes match the
  archive's CRC32, so the defect is in the published archive, not the download. Excluded everywhere
  (`ouc_cge.min_duration_s` = 1 s drops them).
- One stray `7.mp4` in a low archive (not `viewN.mp4`) is ignored.
- **Clips are shorter than the paper's "10 s"**: median 5.4 s (5th–95th percentile 4.0–10.7 s; 75% under
  9 s). Median length is the same per level (5.4 / 5.4 / 5.5 s), but the tails are not:

| length | clips | low | mid | high |
|---|---|---|---|---|
| < 3 s | 341 | **305** | 0 | 36 |
| 3–5 s | 166 | 12 | 4 | 150 |
| 5–9 s | 5,217 | 2,202 | 1,360 | 1,655 |
| 9–12 s | 1,685 | 427 | 672 | 586 |
| ≥ 12 s | 180 | 82 | 5 | 93 |

- Formats: 30 fps for 7,204 decodable clips, 22–29 fps for most others, 9 under 20 fps (two oddities:
  `high_view1394` 180 s at 0.73 fps, `high_view1898` 47 s at 1.8 fps). Resolution 1920×1080 for 6,774;
  the rest 1280×720, 854×480, 1064×598 or 1092×614 (the paper describes 1280×720 cameras).

**No session or camera id is published.** Clips from one recording can sit in both train and test, so
every number on the official split is likely optimistic (§3).

## 2. The floor: models that never look at the video

`python -m scripts.ouc_cge.floor` → `outputs/ouc_cge/data/floor.json`. Fitted on train, scored on all
split rows (the 116 undecodable clips included: this is the label distribution only).

| Baseline | Val acc | Val macro-F1 | Test acc | Test bal. acc | **Test macro-F1** |
|---|---|---|---|---|---|
| Majority ("low" every time) | 0.399 | 0.190 | 0.422 | 0.333 | **0.198** |
| Uniform random (mean of 100 seeds) | 0.333 | 0.331 | 0.335 | 0.335 | 0.331 |
| Stratified random (100 seeds) | 0.341 | 0.332 | 0.344 | 0.333 | 0.333 |

## 3. The shortcut check: file properties only

`python -m scripts.ouc_cge.metadata_baseline` → `outputs/ouc_cge/data/metadata_baseline.json`.
Decodable clips only (train 6,062, val 763, test 764). Features are properties of the file, not of
what it shows: length, fps, width and height, and bytes per second.

| Inputs | Model | Val macro-F1 | Test acc | **Test macro-F1** |
|---|---|---|---|---|
| length | random forest | 0.477 | 0.465 | 0.456 |
| fps, width, height | random forest | 0.310 | 0.342 | 0.293 |
| length + format | random forest | 0.501 | 0.488 | 0.479 |
| bytes per second | random forest | 0.433 | 0.408 | 0.407 |
| **all of the above** | random forest | 0.586 | 0.577 | **0.572** |
| all of the above | logistic regression | 0.367 | 0.418 | 0.350 |

**Reading.** A random forest on file properties alone reaches test macro-F1 0.57, against 0.20–0.33 for
the floor. Length and format identify the recording (camera, session, encoder settings), and the
official split puts clips of the same recordings in train and test, so the label travels with the
recording. Bytes per second is partly a real signal (compressed size grows with motion) and partly
the camera. **The bar for the pipeline on the official split is therefore about 0.57, not 0.20**;
a pixel model that only reaches 0.5–0.6 has not shown that it reads engagement. The logistic
regression stays near the floor because the relation is not linear (it needs thresholds on length
and format, which a forest finds).

## 4. Published results on OUC-CGE

| Work | Protocol | Reported | Comparable? |
|---|---|---|---|
| Lu et al. 2025 (dataset paper), SlowFast and other video models | official split | *to copy from the paper's results table* | Same split as ours; same leakage (§3) |
| DualEngage (Kayenat et al., arXiv 2604.10078) | stratified 5-fold over ~6,700 clips | accuracy 0.9621 ± 0.0161, macro-F1 0.9530 ± 0.0204 | No: different clip set, folds over clips (leaks across recordings); its own ablation reports the motion stream alone at macro-F1 0.51 |

Not rerun here: the authors' repo (`OUC-CGE-main`) ships SlowFast code and configs but no trained weights,
so a rerun means training from Kinetics weights, which needs a larger GPU than the 4 GB laptop (see the run request).

## 5. The pipeline on OUC-CGE (adapters)

Only new files; no existing stage or contract changed.

| Piece | File | What it does |
|---|---|---|
| Dataset index | `src/datasets/ouc_cge.py` | split files → clips (`<level>_<viewN>`), recursive video lookup (nested archive folders, `Low` capitalised, sample folders ignored); SessionKey = (`ouc_cge`, level, viewN) |
| Video frame source | `src/ingestion/video.py` (registry `"video"`) | reads mp4 with OpenCV; `target_fps` 5 subsamples to DIPSER's rate, so the tuned tracker and BFMU settings (in frames) keep their meaning; skipped frames are grabbed, not decoded |
| Config | `configs/ouc_cge.json` | `deliverable3.json` with the dataset, `ingestion.name` "video" + `target_fps` 5, and an `ouc_cge` block (models, feature families, arms, `min_duration_s`) |
| Extraction | `scripts/ouc_cge/extract.py` | the repo's `Pipeline.track` + `window_features`, **one window per clip** (the label unit); caches like `run_pipeline extract`; `--limit`, `--shard i/n`, `--reuse`, `--max-hours`; failures logged, not fatal |
| Evaluation | `scripts/ouc_cge/benchmark.py` | logistic regression and random forest (class-balanced, `src/core/classifiers.py`), fitted on train, scored on val and test; arms: duration, motion, behaviour, + layout, + scene; majority row included |
| Data scripts | `scripts/ouc_cge/{audit,floor,metadata_baseline}.py` | §1–3; no torch needed |
| Tests | `tests/test_ouc_cge.py` | split parsing (placeholder, label/folder mismatch), lookup, video source (rate, stride, cap) |

Not used on OUC-CGE: the context gate and the expert mixture (OUC-CGE is all group work: no activities),
the per-labeler models and the label pipeline (one label per clip, not per student over time).
Synchrony features are not on `main` (CFMU is null; class-relative features are on the `clustering`
branch), so the baseline here is motion + behaviour + layout, not yet the full
motion + behaviour + synchrony baseline.

**Smoke test (CPU, 5 clips, ReID and SCB off because their downloads are blocked in that environment):**
every stage runs; 11–51 frames per clip at 5.00 fps; 0–7 tracklets; 2.4–16.8 s per clip on CPU.
`low_view685` (2.2 s) keeps no tracklet: `tracklets.min_visible_s` is 3 s.

## 6. Open decisions (placeholders)

1. **Short clips.** Report both `min_duration_s` 1 (all decodable clips) and 3 (drops the 341 clips the
   tracker cannot use, 305 of them low). Alternative: lower `tracklets.min_visible_s` for OUC-CGE.
2. **DIPSER-tuned settings** (detector threshold, tracker, flow) are used unchanged on 1080p footage of
   round tables. Hand-check a few clips with the viewers before reading the full results.
3. **Window = whole clip.** A 4-s grid (`ouc_cge.window` "grid") would give several rows per clip and
   needs pooling; not implemented in the benchmark script.
4. **Session-grouped split.** Impossible without the authors' metadata; ask them. Meanwhile, every
   result is read against §3, not only §2.

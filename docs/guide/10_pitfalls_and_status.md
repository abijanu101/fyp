# Chapter 10 — Pitfalls and status

This chapter collects the things that have actually gone wrong, so you do not rediscover them,
and records where the project stands. The status section will go stale; update it when you
change something.

## 10.1 Environment quirks (`fyp-local`)

**Import Pillow before torchvision.** In this environment (torchvision 0.24.0+cu126, Pillow
13.0), importing torchvision first makes Pillow's compiled module fail with
`ImportError: DLL load failed while importing _imaging: The operating system cannot run %1`.
Importing Pillow first works. ultralytics happens to import Pillow first; YOLOv7's model code
does not. `src/bfmu/yolov7.py` imports `PIL.Image` at the very top for this reason. If you write a
new entry point that imports torchvision directly, import `PIL.Image` before it.

**YOLOv7 needs `lib/yolov7` and `seaborn`.** The SCB checkpoints store YOLOv7's own classes, so
the official repo must be cloned into `lib/yolov7` (pinned commit `a207844`), and its
`utils/plots.py` imports `seaborn` at load time. Its other requirements (`albumentations`,
`pafy`, `pkg_resources`) are only imported by code paths we never call.

**Install with `--no-deps`.** The environment pins `torch==2.9.0` for the GTX 1080. A plain
`pip install` of something that depends on torch can try to "upgrade" it. Install extras with
`--no-deps` and add them to `abi_local.yml`.

**Run scripts with `python -m`** from the repo root (Chapter 2).

## 10.2 Loading third-party checkpoints safely

A PyTorch `.pt` file is a Python pickle. Loading it the old way (`torch.load(weights_only=False)`)
can execute arbitrary code embedded in the file. `src/bfmu/yolov7.py` loads the SCB checkpoints
safely:

1. **Scan without loading**: `torch.serialization.get_unsafe_globals_in_checkpoint` lists every
   class the file references (17 for the SCB checkpoints: YOLOv7 layers, standard torch layers,
   numpy arrays).
2. **Refuse anything unexpected**: if any reference falls outside the `ALLOWED` prefixes, raise
   before loading.
3. **Load restricted**: `torch.load(weights_only=True)` inside `safe_globals([...])`, which only
   reconstructs the listed classes. numpy 2 also needs its dtype classes (`Float64DType` and
   similar) allowed explicitly, because they never appear by name in the file.

A test saves a checkpoint containing a disallowed type and checks that loading raises.

## 10.3 Data quirks (DIPSER)

| Quirk | Where | How the code handles it |
|---|---|---|
| Frame rate ~5 fps with gaps up to 6 s | every session | measured fps; gap filter counts missing frames, not wall-clock time |
| 314 empty (0-byte) PNGs, 28% of the session | `group_03/experiment_03/context_cam_03` | dropped when the ZIP is indexed |
| Only one camera for one session | `group_02/experiment_05` | 52 in-scope sessions instead of 54 |
| Attention value `7` in a self-report | `group_01/experiment_03/subject_18` | values outside 1–5 are dropped |
| Labels start up to 2.2 s before the video | all sessions | negligible; `labels.offset_s` exists if needed |
| TAs and teachers walk between students | many sessions | SID association requires the matched student to cover ≥50% of a track's frames |
| `experiments.tsv` says experiment 06 device use is "required" | annotation file | the design document says "optional"; not used by the pipeline |

## 10.4 Code pitfalls that have bitten us

**Indexing a compressed `.npz` repeatedly.** Every `z["key"]` decompresses the whole array again.
Inside a loop that also keeps slices alive, this exhausted 16 GB of RAM. Read every array once
into a dict, then slice (Chapter 8).

**Building expensive objects per session.** An early version of the tracker-tuning script built a
new YOLO detector for every session. Build once, share.

**Forgetting to import a new module.** Its registration never runs; the registry error lists what
*is* registered (Chapter 3).

**Wall-clock gaps versus frame gaps.** Splitting tracks on elapsed time cut every track at every
camera dropout. Use frame positions (`np.searchsorted(frame_t, tr.t)`).

**Replacing NaN with 0 early.** NaN means "no data"; 0 is a real measurement. Let the models'
imputer handle NaN with training statistics.

**Selecting hyperparameters on the test group.** Use `nested_logo_search` and report its nested
estimate.

**Shortcuts that change what is measured.** Caching and blank frames make tuning affordable, but
each is a claim that the fast path equals the real one. Verify it (`tune_tracker --check`), and
prefer the slower honest path where a shortcut would be a single point of failure: the
full-dataset tracker experiments keep CMC on, so every trial decodes real frames.

**Inline comments in `.gitignore`.** Git reads `lib/yolov7/  # comment` as one pattern that matches
nothing. Put comments on their own line.

## 10.5 Status

*Last updated: November 2026. Update this section when you change something it describes.*

### Built and verified on real data

- Ingestion, timing audit, empty-frame handling.
- Person detection; BoT-SORT and StrongSORT tracking with Kalman velocity access; tracklet
  filtering.
- SID–TID association with the passing-person guard; tracking metrics against the manual boxes.
- BFMU: Kalman/geometry motion, residual flow, SCB (soft and flags modes).
- Two-pass pipeline with the two-level cache; feature pass verified against a cached track; one
  full session extracted fresh with per-stage timing (Chapter 5).
- Tracker tuning: quick profiles done (BoT-SORT 0.908 → 0.950, StrongSORT 0.851 → 0.932); ReID
  embedding cache and fast paths verified with `--check`; resumable full-dataset runs.
- Label parsing, label viewers and plots, label inspection; SCB sanity check and viewer.

### Running

- **Three full-dataset tracker experiments** (`tune_tracker`, 20 trials each, CMC on everywhere):
  BoT-SORT without ReID (`full`), StrongSORT (`full_strongsort`), BoT-SORT with ReID
  (`full_reid_cmc`). Their outcome sets the default tracker. Results go in
  `docs/experiments/02_boxmot.md`, §6–7.

### Built and unit-tested, not yet run on all data

- Context gate, engagement experts, mixture, `train_models`, `tune_models`, `infer`.
- `scripts/benchmark.py` (per-stage timing per tracker variant).

These wait on the full extraction of all 52 sessions, which is best run once the tracker is chosen
(a different tracker means re-running the frame pass, about 4–5 hours).

### Decisions taken (and to explain in the write-up)

- **Tracker choice pending** the three full-dataset experiments. On the quick clips, ReID made
  tracking more forgiving of bad settings but did not beat a tuned motion-only tracker, at about
  60× the per-frame tracking cost.
- **CMC on for every full-dataset tracker trial**, for reliability: no full-dataset result depends
  on the blank-frame shortcut.
- **SCB in soft mode.** On DIPSER, `read` fires on anyone looking down at a desk or device and
  `hand_raising` on hand-to-face poses (`docs/experiments/03_scb.md`). Raw scores are passed to the
  experts instead of decisions; a with/without-SCB comparison will decide whether it stays.
- **Region-of-interest features (visual focus of attention): out of scope** for now.
- **Krippendorff's alpha on window-midpoint values** as the labeler-agreement measure.

### Open

1. **Label target.** Labeler agreement is near chance (ordinal α between −0.08 and 0.13 per
   experiment); labelers use the scale very differently (one sits at 3 for 88% of the time,
   another at 2 for 73%); the median removes almost all 4s and 5s. The design's thresholds give
   0.5% engaged windows. Provisional thresholds (`tau_att=3`, `tau_share=0.6`, passed with `--set`)
   until label design is revisited.
2. **Performance.** About 190 ms per frame (~43 minutes per 45-minute session) against a 3-minute
   target, measured under CPU load. Largest costs: residual flow (68 ms), SCB (46 ms), PNG decoding
   (42 ms), detection (31 ms); ReID tracking would add 120–165 ms. The 50-student stress test (§11)
   is not done.
3. **The first end-to-end result.** Full extraction, training (with the feature-family ablation,
   not yet written), and one inference run, which gives the first retention graph from trained
   models.
4. **Possible new feature family.** A frozen pretrained image encoder (for example DINOv2-small
   or ResNet18) on student crops, and possibly whole frames for the gate, as an alternative to
   SCB's fixed classes.

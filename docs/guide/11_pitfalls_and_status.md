# Chapter 11 — Pitfalls and status

This chapter collects the things that have actually gone wrong, so you do not rediscover them,
and records where the project stands. The status section will go stale; update it when you
change something.

## 11.1 Environment quirks (`fyp-local`)

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

## 11.2 Loading third-party checkpoints safely

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

## 11.3 Data quirks (DIPSER)

| Quirk | Where | How the code handles it |
|---|---|---|
| Frame rate ~5 fps with gaps up to 6 s | every session | measured fps; gap filter counts missing frames, not wall-clock time |
| 314 empty (0-byte) PNGs, 28% of the session | `group_03/experiment_03/context_cam_03` | dropped when the ZIP is indexed |
| Only one camera for one session | `group_02/experiment_05` | 52 in-scope sessions instead of 54 |
| Attention value `7` in a self-report | `group_01/experiment_03/subject_18` | values outside 1–5 are dropped |
| Labels start up to 2.2 s before the video | all sessions | negligible; `labels.offset_s` exists if needed |
| TAs and teachers walk between students | many sessions | SID association requires the matched student to cover ≥50% of a track's frames |
| `experiments.tsv` says experiment 06 device use is "required" | annotation file | the design document says "optional"; not used by the pipeline |

## 11.4 Code pitfalls that have bitten us

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

## 11.5 Status

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

- **Three full-dataset tracker experiments** (`tune_tracker`, 20 trials each, CMC on everywhere,
  resumable). As of 15 November 2026: StrongSORT (`full_strongsort`) is **complete** (best IDF1
  0.878, with frequent ID switches); BoT-SORT without ReID (`full`) is at 17 of 20 (best 0.905);
  BoT-SORT with ReID (`full_reid_cmc`) has started. Their outcome sets the default tracker. Results
  in `docs/experiments/02_boxmot.md`, §6–7.

### Built, not yet run on all data

- Context gate, engagement experts, mixture and both aggregation modes are unit-tested.
  `train_models`, `tune_models` and `infer` are scripts and have no unit tests; they were
  smoke-tested on six 100-frame sessions (`tmp/agg_smoke`): extract and train in both modes, infer
  with a late-mode model. Then run end to end on six full-length sessions (extract, train, held-out
  plots, inference on an unseen session; `docs/experiments/05_six_session_run.md`).
- `scripts/benchmark.py` (per-stage timing per tracker variant).
- Data augmentation (augmented training copies per session, Chapter 8 §8.4): verified end to end
  on six 100-frame sessions with one copy each; off by default (`augmentation.copies: 0`). Whether
  it helps is untested: it needs the full extraction plus copies (each copy is another frame pass).

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
- **Late aggregation is the default; early is a config switch away** (`engagement.aggregate`), so
  the two can be compared on the same features and labels. Late aggregation departs
  from §8 as written; §8.4 already anticipates per-student training signal through SID–TID
  association. Each mode has its own engaged cut-off (`mixture.tau` early, `labels.tau_share`
  late) and the hand-weighted baseline reads students' own features in late mode (§7.5).
  The comparison against early still needs the full extraction. Known limitation: at inference,
  people who are not students (teachers, TAs) are scored and averaged in like students; in
  training they are unlabelled and skipped.
- **Region-of-interest features (visual focus of attention): out of scope** for now.
- **Krippendorff's alpha on window-midpoint values** as the labeler-agreement measure.

### Open

1. **Label target.** Labeler agreement is near chance (ordinal α between −0.08 and 0.13 per
   experiment); labelers use the scale very differently (one sits at 3 for 88% of the time,
   another at 2 for 73%); the median removes almost all 4s and 5s. The design's thresholds give
   0.5% engaged windows. Provisional thresholds (`tau_att=3`, `tau_share=0.6`, passed with `--set`)
   until label design is revisited.
2. **Performance.** About 185 ms per frame (~40 minutes per 45-minute session) against a 3-minute
   target, measured on six full-length sessions under load (`docs/experiments/05_six_session_run.md`).
   Largest costs: residual flow (56 ms), PNG decoding (49 ms), SCB (47 ms), detection (29 ms); ReID
   tracking would add 120–165 ms. The 50-student stress test (§11) is not done.
3. **The first real end-to-end result.** A six-session full-length run works end to end
   (`docs/experiments/05_six_session_run.md`), but with one session per activity it cannot evaluate
   the gate or the per-activity experts. Needed: the full extraction, training in both aggregation
   modes, and the feature-family ablation (not yet written).
4. **Possible new feature family.** A frozen pretrained image encoder (for example DINOv2-small
   or ResNet18) on student crops, and possibly whole frames for the gate, as an alternative to
   SCB's fixed classes.

5. **Improving the engagement scores.** On the six-session run the features carry consistent
   within-session signal (class motion against the labelled share, ρ ≈ −0.3) that the trained
   models do not use; leaving out the Vec context already helps. Candidate changes, in order:
   within-session metrics, per-session feature normalisation, a continuous target, per-labeler
   label normalisation, signed baselines, pooled experts, temporal smoothing
   (`docs/experiments/05_six_session_run.md`, §6–7).

## 11.6 Settings that do nothing: decisions to make

A config key that changes nothing still changes the config hash (and some change the tracking
hash), so two runs that behave identically get different results files, and a reader of the config
believes a knob is working when it is not. Each item below is waiting for a decision; tick it and
record the decision when it is made. Found by tracing every key of `configs/deliverable3.json`
through the code (Chapter 9, §9.6–9.7).

**☐ 1. `context_mode` (`"classified"`): unused.**
- Purpose: choose whether the experts are weighted by the gate's predicted activity or by the true
  one (oracle).
- Reality: nothing reads it. `train_models` always reports both, and `Pipeline.predict` always uses
  the gate (`src/core/pipeline.py`, `predict`). The string appears only in an error message
  (`src/spu/context.py`).
- Cost of keeping: changing it changes only the config hash.
- Options: (a) remove it; (b) wire it into `run_pipeline infer`, so a retention graph can be drawn
  with the true activity on DIPSER sessions (useful to show expert quality without gate errors).
- Recommendation: (a), unless oracle-context graphs are wanted, then (b).

**☐ 2. `hand_weighted.phone_key` / `phone_sign`: no effect.**
- Purpose: give the label-free baseline a phone-use term whose sign depends on the activity:
  positive where devices are part of the task (01, 04, 05), negative where they are not (02, 03),
  zero for 06.
- Reality: `phone_key` names `vec.scb.using_phone.mean`, but the configured SCB checkpoint has no
  phone class (classes: hand_raising, read, write), so the column never exists and
  `HandWeightedExpert` silently skips it (`src/experts/expert.py`).
- Evidence for a substitute: on DIPSER, SCB's `read` fires on students looking at phones, tablets
  and laptops (15 of about 17 students tagged `read` in a Canva session;
  `docs/experiments/03_scb.md`, §4). `read` is the de facto device-use signal.
- Options: (a) remove both keys; (b) point `phone_key` at `vec.scb.read.mean`, so the per-activity
  signs apply to the signal that actually tracks device use (it would then replace `read`'s fixed
  0.5 weight); (c) keep as is for a future SCB model with a phone class.
- Recommendation: (b) if the baseline should encode the activity-dependent meaning of device use,
  (a) if it should stay a minimal floor. (c) keeps a silent no-op.
- Related: on the six-session run, students with higher SCB `read` and `write` scores were rated
  *less* attentive (single-feature AUC 0.36–0.39), while the baseline weights both positively
  (`docs/experiments/05_six_session_run.md`, §6.3). Decide the weights and the phone term together.

**☐ 3. `mixture.tau` (0.5): unused in late aggregation only.**
- Purpose: the engaged cut-off for early aggregation, where the window score is a probability.
- Reality: with `engagement.aggregate = "late"` (the default) the cut-off is `labels.tau_share`,
  because the late score estimates a share of engaged students (§7.5).
- Recommendation: keep. Removing it breaks early aggregation, which is kept for comparison.
  Chapter 9 marks it early-only.

**☐ 4. Settings that are inactive in the current configuration, but not dead.**
`tracker.options.proximity_thresh` and `appearance_thresh` act only when `use_embeddings` is
true (the `full_reid_cmc` experiment); `scb.models[].conf` acts only in `scb.mode = "flags"`.
Both are in the tracking hash, so changing them forces a needless frame pass while inactive.
- Options: (a) keep, documented (Chapter 9 does); (b) drop them from the config while inactive and
  restore them with the mode that needs them.
- Recommendation: (a): they are the starting values for the modes the experiments still compare.

**☑ 5. `tracker.options.det_thresh`: removed.** BoxMOT 26 stores it but neither BoT-SORT nor
StrongSORT reads it. Removed from the config and from `tune_tracker`; the detector's own threshold
(`detector.conf`) is the only confidence filter, and it was verified not to drop any student
(`docs/experiments/02_boxmot.md`, §3).

**☑ 7. On-the-fly augmentation.** Augmented copies are no longer written to disk at all (the
materialised datasets and `scripts/augment_dataset.py` were removed on 15 November 2026): frames are
augmented as they are read, and seeding makes every read of a copy identical. For the final
training run, raise `augmentation.copies` as far as extraction time allows; no disk is needed.

**☑ 6. StrongSORT scan restarted with its quick-tuned baseline, and complete.** Migrated (trials
1–5 kept, trial 0 re-ran at 0.843), finished at a best IDF1 of 0.878
(`docs/experiments/02_boxmot.md`, T1). The BoT-SORT scan was restarted with `--fresh` instead of
migrated; its recomputed trials matched the old ones exactly.

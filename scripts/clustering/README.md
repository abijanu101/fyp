# Clustering features on the full dataset (GPU machine)

Every command needed to run the clustering screens of guide 15 §15.9 on all of DIPSER, from a fresh
machine to the one file to bring back. Run everything from the repository root, in the
**Anaconda Prompt** (Windows) or a shell with the environment active. Each script's docstring has
its full options (`python -m scripts.clustering.<name> --help`).

**Two kinds of work, two kinds of hardware.**
- **Extraction (step 3)** runs detection, tracking, head pose and SCB on every frame. This is the
  GPU job, and it takes almost all of the time.
- **The clustering scripts (steps 4–6)** only read the caches and run on the **CPU**. `--workers N`
  runs N sessions at once. Use about the number of CPU cores minus 2 (8 below).

## 1. Set up the machine (once)

```
git clone https://github.com/abijanu101/fyp.git
cd fyp
git checkout clustering                                # the branch with scripts/clustering
conda env create -f abi_local.yml
conda activate fyp-local
pip install python-dotenv                             # the downloader needs it; abi_local.yml lacks it
python -m scripts.setup                               # every model the pipeline needs
python -m scripts.setup --check                       # all present?
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must print True
```

Create `.env` at the repository root with the DIPSER FTP credentials (`FTP_USER=...`,
`FTP_PASSWORD=...`). It is git-ignored; never commit it.

## 2. Download the data

```
python scripts/downloading/dipser.py
```

The default scope is the pipeline's: groups 01–03, experiments 01–06, context cameras 01–03
(54 camera ZIPs, 52 with frames). Each subject archive (about 1 GB, 20 per experiment) is
downloaded only to extract its labels, then deleted. The run is resumable: re-running skips what is
done. For a smaller test, narrow the scope, e.g.
`python scripts/downloading/dipser.py --groups group_01 group_02 --experiments experiment_03`.

## 3. Extract features with the experiment 09 config (GPU, the long step)

```
python -m scripts.parallel_runs extract --config docs/experiments/09_vfoa.json --workers 3
```

- **What runs:** every recording, then its 4 augmented copies, 5 passes of 52 recordings in all.
  The copies are needed by the label scripts (step 6), which train as `train_models` does.
- **Workers:** each worker holds its own detector, ReID and SCB models on the GPU. Use 3 with
  16 GB or more of VRAM, 2 with 8–12 GB. If you see a CUDA out-of-memory error, use fewer.
- **Resumable:** a finished recording is skipped. If the run stops, run the same command again.
- **Time:** about real time per recording per worker on a GPU (experiment 04 measured 185 ms per
  frame). On this laptop's CPU it was 61 minutes for one 5-minute recording.
- **Output:** `outputs/cache/tracks/<hash>/` and `outputs/cache/features/<hash>/`. These are
  git-ignored and large.

Optional: the experiment 09 baseline itself, which is the `null` arm with the configured gate. It
also builds the label-window caches that step 6 needs, so step 6 starts faster:

```
python -m scripts.train_models --config docs/experiments/09_vfoa.json
```

## 4. What to cluster on (label-free, CPU)

```
python -m scripts.clustering.signal_screen --workers 8
python -m scripts.clustering.event_check
python -m scripts.clustering.descriptor_subset
python -m scripts.clustering.stability --workers 8
```

- **Order matters:**
  - `descriptor_subset` reads `signal_screen`'s descriptor table to drop seat-like descriptors.
  - `stability` reads the chosen `subset`.
- **To go faster** at the cost of precision, add `--max-windows 60` to `signal_screen` and
  `stability`. That uses 60 evenly spaced windows per session (pairs of neighbours, for the
  temporal score).
- **Before step 5, read the stability output** (`stability.csv` or `stability.png`):
  - Pick the descriptor threshold (`desc:subset`, agglomerative) and the synchrony threshold
    (`sync:<channel>`, agglomerative) with the highest temporal ARI that is still `usable`.
  - Pick the channels that passed synchrony in `signal_screen_channels.csv` (`pass_sync`).
  - On the one local recording these were τ 1.0 / 0.5 and `rf.total hp.yaw kg.speed scb.hand_raising`.

## 5. Compute the candidate features (CPU, stored as JSON)

```
python -m scripts.clustering.candidates --workers 8 --channels rf.total hp.yaw kg.speed scb.hand_raising --tau-desc 1.0 --tau-sync 0.5
```

- **Where:** `outputs/experiments/10_clustering/candidates/<feature hash>-<settings hash>/`
  - one `<stem>.json` per session, originals and augmented copies;
  - the settings in `params.json`.
- **File layout:** the feature names are listed once (`per_student_features`, `per_window_features`).
  Each window centre then holds its window values and one value list per visible student, in that
  order. Values have 6 significant digits; NaN is `null`. About 4 MB per session.
- **Other formats:** `--format csv` writes flat tables, and `--format both` writes both. The
  scripts read either, JSON first.
- **First run:** this builds the label rows first, which caches `<stem>.labelwin.csv` for every
  session. It is resumable: finished sessions are skipped unless `--force`.
- **Other thresholds:** changing τ or the channels writes to a new folder, and the old one stays.
  The next steps use the newest folder unless given `--candidates <folder>`.

## 6. Do the features predict attention? (labels, CPU)

Two train/test splits, both run by default (`--split both`) and reported separately:

| Split | Train | Test | Folds | Tests generalisation to |
|---|---|---|---|---|
| `activity` (main) | 4 of the 6 activities (experiments 01–06), every group | the other 2 activities | all 15 pairs, averaged | new activities. The same students and seats are in train and test, because every group records every activity |
| `group` | 2 groups | the third group | 3 | new students (the protocol of `train_models` and experiments 06–09) |

The gap between the two shows how much of a score comes from knowing the students. `--split activity`
or `--split group` runs only one.

```
python -m scripts.clustering.feature_screen
python -m scripts.clustering.importance --gate
python -m scripts.clustering.ablation
python -m scripts.clustering.ablation --best agglomerative
python -m scripts.clustering.ablation --best agglomerative --no-window
```

- **`feature_screen`:** within-student correlation and mutual information, training rows only,
  per split. Plots: `feature_screen_activity.png`, `feature_screen_group.png`.
- **`importance`:** held-out permutation importance per split, with the true activity as context.
  `--context classified` uses the gate instead, on the group split only. `--gate` also tests the
  window features in the activity gate, on the group split only: a gate can't be tested on
  activities it never saw.
- **`ablation`:** every arm on both splits.
  - **Context:** the activity split always uses the true activity, because the gate can't name an
    activity it never saw. With experiment 09's shared expert this doesn't change the ratings. The
    group split uses the configured gate.
  - **`--best`:** the first run chooses `sync+best` on the held-out folds. Read which arm it picked
    for each split, then rerun with `--best <that arm>`, so the cited result doesn't choose on the
    test data.
  - **`--no-window`** checks whether the window-level features only act as session identifiers.
  - **Reading the verdict:** the 15 activity folds overlap, since each activity is tested in 5 of
    them. So their spread understates the uncertainty: treat the activity verdict as indicative,
    and the group verdict (independent folds) as the stricter one.
- **Expert model:** the scripts use the config's expert (logistic regression in experiment 09). To
  try another, add the same `--set` you would give `train_models`. The activity split needs one
  expert shared by all activities (`mixture.expert_scope=shared`, as in experiment 09); the scripts
  stop with a message otherwise.

## 7. Bring the results back

```
python -m scripts.clustering.report
```

This writes `outputs/experiments/10_clustering/clustering_report.json`: every screen's key numbers
in one file. Copy back over AnyDesk:
- `clustering_report.json`;
- every `.csv`, `.json` and `.png` directly in `outputs/experiments/10_clustering/`;
- the `ablation/` folder.

The `candidates/` folder is about 1 GB. Copy it only if you want to recompute something locally.
The caches in `outputs/cache/` stay on the GPU machine.

## Checks if something stops

| Message | Cause | Fix |
|---|---|---|
| `no cached sessions in outputs/cache/...` | step 3 not run, or a different config | run step 3 with `--config docs/experiments/09_vfoa.json` |
| `augmented copy 1 not extracted` | step 3 stopped before the copies | rerun step 3 (it resumes) |
| `leave-one-group-out needs at least two groups` | only one group downloaded or extracted | download and extract all three groups |
| `the activity split ... needs at least 3` | too few activities extracted | extract experiments 01–06 |
| `--split activity needs mixture.expert_scope=shared` | the config uses per-activity experts | `--set mixture.expert_scope=shared`, or `--split group` |
| `no candidate tables` | step 5 not run | run step 5 |
| CUDA out of memory in step 3 | too many workers for the GPU | `--workers 2` or `1` |

# Experiment 10 on the full dataset (GPU machine)

Every command needed to run experiment 10, the clustering features of guide 15 §15.9, on all of
DIPSER: from a fresh machine to the one file to bring back. Run everything from the repository
root, in the **Anaconda Prompt** (Windows) or a shell with the environment active. Each script's
docstring has its full options (`python -m scripts.clustering.<name> --help`).

**Two kinds of work, two kinds of hardware.**
- **Extraction (step 3)** runs detection, tracking, head pose and SCB on every frame. This is the
  GPU job, and it takes almost all of the time. It can be skipped if experiment 09's caches can be
  copied over (step 3).
- **The clustering scripts (steps 4–7)** only read the caches and run on the **CPU**. `--workers N`
  runs N sessions at once. Use about the number of CPU cores minus 2 (8 below).

## All commands, in order

```
:: 1. Set up (once). If the repository is already cloned, skip the clone and the cd, and run
::    git fetch origin, then git checkout clustering, then git pull
git clone https://github.com/abijanu101/fyp.git
cd fyp
git checkout clustering
conda env create -f abi_local.yml
conda activate fyp-local
pip install python-dotenv
python -m scripts.setup
python -c "import torch; print(torch.cuda.is_available())"

:: 2. Data (skip if data\DIPSER\DIPSEER already holds groups 01-03, experiments 01-06)
python scripts/downloading/dipser.py

:: 3. Features on the GPU (skip if experiment 09's caches were copied in; see step 3)
python -m scripts.parallel_runs extract --config docs/experiments/09_vfoa.json --workers 3

:: 4. What to cluster on (no labels)
python -m scripts.clustering.signal_screen --workers 8
python -m scripts.clustering.event_check
python -m scripts.clustering.descriptor_subset
python -m scripts.clustering.stability --workers 8
::    now read stability.csv and signal_screen_channels.csv, and set the values in step 5

:: 5. Candidate features (stored as JSON)
python -m scripts.clustering.candidates --workers 8 --channels rf.total hp.yaw kg.speed scb.hand_raising --tau-desc 1.0 --tau-sync 0.5

:: 6. Do they predict attention? (both splits: 4 activities train / 2 test, and leave one group out)
python -m scripts.clustering.feature_screen
python -m scripts.clustering.importance --gate
python -m scripts.clustering.ablation
::    read which arm "sync+best" picked, then freeze it (agglomerative shown; use the one it printed)
python -m scripts.clustering.ablation --best agglomerative
python -m scripts.clustering.ablation --best agglomerative --no-window

:: 7. One file with every result, to copy back
python -m scripts.clustering.report
```

Lines starting with `::` are comments for the Anaconda Prompt (cmd). In PowerShell, skip them or
change `::` to `#`.

Expected time: step 3 about 17 h on 3 workers (zero if the caches are copied); steps 4–5 about 2–3 h;
step 6 about 5–8 h; step 7 under a minute. Steps 4–7 run on the CPU and can be left unattended.

## 1. Set up the machine (once)

- **New machine:** the clone, `conda env create` and `scripts.setup` lines above.
- **Repository already cloned:** run `git fetch origin`, `git checkout clustering` and `git pull`.
  `git log --oneline -1` should show the newest commit of the `clustering` branch.
- **An existing environment that already runs the pipeline** (a conda env or venv) can be used
  instead of creating `fyp-local`. It needs the packages in `abi_local.yml` plus `python-dotenv`.
- **The GPU check** must print `True`. `python -m scripts.setup --check` lists any missing model.
- **`.env`:** create it at the repository root with the DIPSER FTP credentials (`FTP_USER=...`,
  `FTP_PASSWORD=...`). It is git-ignored; never commit it. It's only needed for step 2.

## 2. Download the data

The default scope is the pipeline's: groups 01–03, experiments 01–06, context cameras 01–03
(54 camera ZIPs, 52 with frames). Each subject archive (about 1 GB, 20 per experiment) is
downloaded only to extract its labels, then deleted. The run is resumable: re-running skips what is
done. To narrow the scope, for example for a test:
`python scripts/downloading/dipser.py --groups group_01 group_02 --experiments experiment_03`.

## 3. Extract features with the experiment 09 config (GPU, the long step)

- **Skip this step if experiment 09's caches exist.** Experiment 10 uses experiment 09's config
  unchanged, so experiment 09's extraction produced exactly the folders it reads:
  `outputs/cache/tracks/2a11240a2d42/` and `outputs/cache/features/4c761244568f/`. Copy both from
  the computer that ran experiment 09 into `outputs\cache\tracks\` and `outputs\cache\features\`
  here, keeping the folder names. Running the extraction command afterwards then just reports every
  recording as done.
- **What runs otherwise:** every recording, then its 4 augmented copies, 5 passes of 52 recordings
  in all. The copies are needed by step 6, which trains as `train_models` does.
- **Workers:** each worker holds its own detector, ReID and SCB models on the GPU. Use 3 with
  16 GB or more of VRAM, 2 with 8–12 GB. If you see a CUDA out-of-memory error, use fewer.
- **Resumable:** a finished recording is skipped. If the run stops, run the same command again.
- **Time:** experiment 09 measured about 17 h for all 260 recordings on 3 workers, about 0.8 min
  of extraction per minute of footage. On a laptop CPU it was 61 minutes for one 5-minute
  recording.
- **Output:** the two cache folders above, git-ignored, about 10–15 GB.
- **Not needed:** `train_models` on the experiment 09 config. Its result is already in experiment
  09, and experiment 10's `null` arm is the same model, recomputed on experiment 10's folds.

## 4. What to cluster on (label-free, CPU)

- **Order matters:**
  - `descriptor_subset` reads `signal_screen`'s descriptor table to drop seat-like descriptors.
  - `stability` reads the chosen `subset`.
- **Start early:** these four only need the original recordings, not the augmented copies.
  `parallel_runs` extracts the originals first, so step 4 can start while step 3 is still
  extracting copies.
- **To go faster** at the cost of precision, add `--max-windows 60` to `signal_screen` and
  `stability`. That uses 60 evenly spaced windows per session (pairs of neighbours, for the
  temporal score).
- **Before step 5, read the outputs** in `outputs/experiments/10_clustering/`:
  - In `stability.csv` (or `stability.png`), pick the descriptor threshold (`desc:subset`,
    agglomerative) and the synchrony threshold (`sync:<channel>`, agglomerative) with the highest
    temporal ARI that is still `usable`.
  - In `signal_screen_channels.csv`, pick the channels with `pass_sync` true.
  - Put them into step 5's `--tau-desc`, `--tau-sync` and `--channels`. On the one local recording
    these were τ 1.0 / 0.5 and `rf.total hp.yaw kg.speed scb.hand_raising`.

## 5. Compute the candidate features (CPU, stored as JSON)

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
  Step 6 uses the newest folder unless given `--candidates <folder>`.

## 6. Do the features predict attention? (labels, CPU)

Two train/test splits, both run by default (`--split both`) and reported separately:

| Split | Train | Test | Folds | Tests generalisation to |
|---|---|---|---|---|
| `activity` (main) | 4 of the 6 activities (experiments 01–06), every group | the other 2 activities | all 15 pairs, averaged | new activities. The same students and seats are in train and test, because every group records every activity |
| `group` | 2 groups | the third group | 3 | new students (the protocol of `train_models` and experiments 06–09) |

The gap between the two shows how much of a score comes from knowing the students. `--split activity`
or `--split group` runs only one.

- **`feature_screen`:** within-student correlation and mutual information, training rows only,
  per split. Plots: `feature_screen_activity.png`, `feature_screen_group.png`.
- **`importance`:** held-out permutation importance per split, with the true activity as context.
  `--context classified` uses the gate instead, on the group split only. `--gate` also tests the
  window features in the activity gate, on the group split only: a gate can't be tested on
  activities it never saw.
- **`ablation`:** the main test. It compares these arms on both splits:
  - `null`: experiment 09's model, with no class-relative features (the baseline);
  - `normalised`: `null`'s inputs with each student's features standardised against their own
    session average and spread. It tests whether the existing features carry more signal once the
    seat is removed;
  - `sync`, `agglomerative`, `hdbscan`, `prototype`, `sync+best`: the class-relative arms of
    guide 15 §15.6.
- **Running `ablation` three times:**
  1. The first run chooses `sync+best` on the held-out folds and prints which arm it picked, for
     each split.
  2. The second run freezes that choice with `--best <that arm>`, so the cited result doesn't
     choose on the test data.
  3. The third, with `--no-window`, checks whether the window-level features only act as session
     identifiers.

  To save time on runs 2 and 3, add `--arms null normalised sync sync+best`, which skips the arms
  the first run already scored.
- **Context:** the activity split always uses the true activity, because the gate can't name an
  activity it never saw. With experiment 09's shared expert this doesn't change the ratings. The
  group split uses the configured gate.
- **Reading the verdict:** the 15 activity folds overlap, since each activity is tested in 5 of
  them. So their spread understates the uncertainty: treat the activity verdict as indicative,
  and the group verdict (independent folds) as the stricter one.
- **Model and inputs:** the scripts follow the config exactly as `train_models` does. That covers
  the expert (logistic regression in experiment 09), the base inputs (08's per-student features
  plus head pose, no class context), the training rebalancing (`labels.train_keep`, off) and the
  rating scale (`labels.collapse`, off). With the commands above, every arm starts from experiment
  09's default model, which is what experiment 10 compares against. Experiment 09 itself is not
  rerun.

### Optional: other experiment 09 variants

Not part of the plan; only if a specific question needs it. Experiment 09 also tried variants of
its model: without head pose, on the LOW / MID / HIGH scale, and with fewer 2s and 3s in training.
To test the clustering features on top of one of those, add that variant's `--set` to `ablation`.
For example, to answer "do the clustering features still help without head pose?", in the
Anaconda Prompt:

```
python -m scripts.clustering.ablation --best agglomerative --set "engagement.columns={\"exclude\": [\"trk.hp.\", \"vec.hp.\", \"rel.\"]}"
```

- **What survives:** each `ablation` run's full results go to `ablation/<config hash>-<settings
  hash>.json`. The variant has a different config hash, so the main runs' files are kept.
- **What gets overwritten:** the summary tables `ablation.csv`, `ablation.json` and
  `ablation_verdict.*` hold the latest run only. Copy the `ablation/` folder before running a
  variant, or run variants after `report`.
- **Same candidates:** keep the default `--out`, so the variant uses the same candidates and
  descriptor subset as the main run.
- **Expert:** the activity split needs one expert shared by all activities
  (`mixture.expert_scope=shared`, as in experiment 09); the scripts stop with a message otherwise.

## 7. Bring the results back

`report` writes `outputs/experiments/10_clustering/clustering_report.json`, with every screen's key
numbers in one file. Copy back over AnyDesk:
- `clustering_report.json`;
- every `.csv`, `.json` and `.png` directly in `outputs/experiments/10_clustering/`;
- the `ablation/` folder.

The `candidates/` folder is about 1–3 GB. Copy it only if you want to recompute something locally.
The caches in `outputs/cache/` stay on the GPU machine.

## Checks if something stops

| Message | Cause | Fix |
|---|---|---|
| `no cached sessions in outputs/cache/...` | step 3 not run or caches not copied, or a different config | run step 3 with `--config docs/experiments/09_vfoa.json`, or copy experiment 09's caches |
| `augmented copy 1 not extracted` | step 3 stopped before the copies | rerun step 3 (it resumes) |
| `leave-one-group-out needs at least two groups` | only one group downloaded or extracted | download and extract all three groups |
| `the activity split ... needs at least 3` | too few activities extracted | extract experiments 01–06 |
| `--split activity needs mixture.expert_scope=shared` | the config uses per-activity experts | `--set mixture.expert_scope=shared`, or `--split group` |
| `no candidate tables` | step 5 not run | run step 5 |
| CUDA out of memory in step 3 | too many workers for the GPU | `--workers 2` or `1` |

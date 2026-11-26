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

## The one command

With the environment set up and the extraction caches in place (steps 1–3), all of experiment 10 is:

```
python -m scripts.clustering.run_all
```

- **What it does:** steps 4–7 in order. It reads the channels and the two τ from the outputs by the
  rules below, and saves them in `run_choices.json`.
- **Logs:** each step's output goes to the console and to `outputs/experiments/10_clustering/logs/<step>.log`.
- **If it stops** (a crash, a closed window, a reboot): run the same command again. Finished steps
  are skipped, and the interrupted step resumes from the recordings or folds it had finished.
- **When an earlier step reruns** (`--redo stability`, other `--channels`, …), every step after it
  reruns too, reusing its own saved results wherever nothing changed.
- **Workers:** `--workers` for the per-recording steps (default: CPU threads minus 2), and
  `--label-workers` for the label steps (default 3; each holds the label rows, about 0.5 GB).
- **Other options:** `--split group` runs one split only; `--skip importance` leaves out the slowest
  optional step; `--dry-run` prints the commands without running them.
- **Overrides:** `--channels`, `--tau-desc` and `--tau-sync` replace the automatic choices, for
  example on a sample too small for the per-recording test.

The individual commands follow, for running or rerunning one step by hand.

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

:: 0. Before a long run: every step on a small real sample, with real labels (see "Testing first")
set FYP_SAMPLE_TEST=1
python -m unittest tests.test_clustering_sample -v

:: 4. What to cluster on (no labels)
python -m scripts.clustering.signal_screen --workers 8
::    channels: the rows of signal_screen_channels.csv with pass_sync True (and check pass_sync_low_motion)
python -m scripts.clustering.event_check --workers 8 --channels <channels>
python -m scripts.clustering.descriptor_subset
python -m scripts.clustering.stability --workers 8 --channels <channels>
::    tau-desc and tau-sync: printed at the end of stability ("choice of tau")

:: 5. Candidate features (stored as JSON)
python -m scripts.clustering.candidates --workers 8 --channels <channels> --tau-desc <tau-desc> --tau-sync <tau-sync>

:: 6. Do they predict attention? (both splits: leave one group out, and 4 activities train / 2 test)
python -m scripts.clustering.feature_screen --workers 3
python -m scripts.clustering.importance --gate --repeats 2 --workers 3
python -m scripts.clustering.ablation --workers 3

:: 7. One file with every result, to copy back
python -m scripts.clustering.report
```

`scripts/experiments/10_clustering/run.sh` runs steps 4–7 in order and reads the channels and the two
τ from the outputs by the rules below (bash: Git Bash or WSL).

Lines starting with `::` are comments for the Anaconda Prompt (cmd). In PowerShell, skip them or
change `::` to `#`.

Expected time on a 6-core CPU: step 3 about 17 h on 3 GPU workers (zero if the caches are copied);
step 4 about 1–1.5 h; step 5 about 2.5–3 h (synchrony over 24 s of context); step 6 about 6–8 h
(`ablation` 11 arms, about 4–5 h; `feature_screen` about 1.5 h; `importance --repeats 2` about 2.5 h);
step 7 under a minute. Every printed line starts with the time. The label rows are built once and
reused from `outputs/cache/label_rows/` (about 10 minutes saved per step).

## Parallel runs, resuming and failures

- **Workers:** every step except `descriptor_subset` and `report` takes `--workers N`.
  - The per-recording steps (`signal_screen`, `event_check`, `stability`, `candidates`) are light:
    use the number of CPU threads minus 2.
  - The label steps (`feature_screen`, `importance`, `ablation`) run their folds in parallel, and
    each worker holds a copy of the label rows (about 0.5 GB on the full data). So use 2–4 workers
    on 16 GB of RAM.
- **Resuming:** every step saves each finished recording or fold as it completes:
  - `candidates` writes its JSON files;
  - the others save under `outputs/experiments/10_clustering/partial/<step>/<settings hash>/`.

  Rerunning the same command after a crash, a closed window or a reboot loads what's done and only
  computes the rest. Changed settings get a new folder, so nothing stale is reused. To force a step
  to recompute, delete its folder under `partial/`. The label rows are built once and reused from
  `outputs/cache/label_rows/`.
- **Failures:** if one recording or fold fails, the others still finish and are saved. The failures
  are then listed with their full error messages, and the step stops with an error. Fix the cause
  and rerun: only the failed jobs run again.
- **Logs:** every printed line starts with the time.

## Testing first

`tests/test_clustering_sample.py` runs steps 4–7 end to end, with real labels, on a small real sample
(at least 2 groups and 3 activities, with augmented copies), in a scratch folder
(`tmp/exp10_sample_test`). Experiment 10's first full run crashed twice in steps 5–6, because those
steps first met real labels there; run this before any long run. It is skipped by the normal test
run, and needs `FYP_SAMPLE_TEST=1` (PowerShell: `$env:FYP_SAMPLE_TEST = "1"`). With one augmented copy
per recording, extract the sample with `--set augmentation.copies=1` and run the test with
`FYP_SAMPLE_COPIES=1` (the default).

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
  09, and experiment 10's `base` arm is the same model, recomputed on experiment 10's folds.

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
- **How a channel passes** (`signal_screen_channels.csv`): each recording counts once. Its mean
  excess synchrony (real minus shifted) is tested across recordings, Benjamini–Hochberg across the
  channels, `pass_sync` = q < 0.05. `pass_sync_low_motion` repeats this in each recording's calmest
  third of windows: a channel that passes only when the whole picture moves may be driven by the
  camera, not the students. `event_check_channels` does the same for the event reactions, with and
  without camera-like events.
- **The channels for step 5:** the rows with `pass_sync` True. Note any that fail
  `pass_sync_low_motion`; they are kept, but their synchrony may partly be the camera.
- **τ** (`stability`, printed at the end): among usable agglomerative settings, the highest
  **temporal excess**, which is how much more a grouping persists into the next window than into a
  window 20 s away. The raw temporal ARI rises by itself as clusters get coarser (everyone in one
  cluster scores 1.0), so it is not used. If the chosen τ is the largest tested, the script says so:
  extend `--taus-desc` or `--taus-sync` and rerun `stability`.

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
- **Synchrony over 24 s:** synchrony, lag and synchrony clustering are computed over the 24 s ending
  with each rating window (`--context-s`), not the 4 s window. Two students must share at least half
  those frames (`--min-common-share`), and the lag is searched over ±3 s (`--max-lag-s`).
- **Standardisation floor:** each student's descriptors are standardised against their own session,
  but their SD is never taken below a quarter of the typical student's. Otherwise a very still
  student's jitter would become large z-scores.

## 6. Do the features predict attention? (labels, CPU)

Two train/test splits, both run by default (`--split both`) and reported separately:

| Split | Train | Test | Folds | Tests generalisation to |
|---|---|---|---|---|
| `activity` (main) | 4 of the 6 activities (experiments 01–06), every group | the other 2 activities | all 15 pairs, averaged | new activities. The same students and seats are in train and test, because every group records every activity |
| `group` | 2 groups | the third group | 3 | new students (the protocol of `train_models` and experiments 06–09) |

The gap between the two shows how much of a score comes from knowing the students. `--split activity`
or `--split group` runs only one.

- **`feature_screen`:** within-student correlation with the ratings, training rows only, per split.
  Each correlation gets a p-value from a block null: each student's ratings are circularly shifted
  along their own time order, 200 times, which keeps their autocorrelation. Benjamini–Hochberg then
  corrects across the candidates (q). Mutual information is no longer reported (its estimator broke
  on tied values). `feature_screen_missingness.json` checks whether the NaN pattern alone predicts
  the recording. Plots: `feature_screen_activity.png`, `feature_screen_group.png`.
- **`importance`:** held-out permutation importance per split, with the true activity as context.
  `--context classified` uses the gate instead, on the group split only. `--gate` also tests the
  window features in the activity gate, on the group split only: a gate can't be tested on
  activities it never saw.
- **`ablation`:** the main test, **run once**. Every arm is fixed in advance, and nothing is chosen
  on test results:
  - `base`: experiment 09's model, with no class-relative features (the baseline; called `null` in the
    first full run, renamed because pandas reads the word "null" in a CSV as missing);
  - `normalised`: `base`'s inputs with each student's features standardised against their own session;
  - `sync`, `agglomerative`, `hdbscan`, `prototype`: per-student features of one family each;
  - `sync+agglomerative`, `sync+hdbscan`, `sync+prototype`, `normalised+sync`,
    `normalised+sync+agglomerative`: fixed combinations;
  - `window`: the window-level features (number of groups, fragmentation, class synchrony, …),
    centred per recording, as their own arm. The per-student arms never include them.
- **The test:** every held-out entry's prediction is saved, and each arm is compared with `base` and
  `sync` as a paired difference (Spearman and MAE), with a 95% interval from a bootstrap over
  recordings (`ablation_intervals`).
  - **Two primary tests**, both fixed in advance, on the group split, Spearman, against `base`:
    1. **Experts:** `sync+agglomerative` (`--primary`). Do the class-relative features help on the
       experts' ratings?
    2. **Self-report:** `normalised` (`--primary-self`). Does removing between-student (seat)
       differences recover the students' own reports? The first full run found this in all 18 folds;
       this test confirms it or not.

    Because there are two, each is "supported" only if its Bonferroni-adjusted 98.75% interval is
    above 0. That keeps a 5% chance of any false claim across both.
  - **Everything else is exploratory** (95% intervals, uncorrected): report it, but don't claim it.
- **The group split is primary** for any claim about the features: it's the only one with unseen
  students.
- **Context:** the activity split always uses the true activity, because the gate can't name an
  activity it never saw. With experiment 09's shared expert this doesn't change the ratings. The
  group split uses the configured gate.
- **The activity split:** its 15 folds overlap, since each activity is tested in 5 of them. Its
  intervals average each entry's predictions over the folds that test it, and resample recordings,
  so they don't treat the folds as independent. It still answers only "new activities".
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
python -m scripts.clustering.ablation --set "engagement.columns={\"exclude\": [\"trk.hp.\", \"vec.hp.\", \"rel.\"]}"
```

- **What survives:** each `ablation` run's full results go to `ablation/<config hash>-<settings
  hash>.json`. The variant has a different config hash, so the main runs' files are kept.
- **What gets overwritten:** the summary tables `ablation.*`, `ablation_intervals.*` and
  `ablation_predictions.csv.gz` hold the latest run only. Copy the `ablation/` folder before running a
  variant, or run variants after `report`.
- **Same candidates:** keep the default `--out`, so the variant uses the same candidates and
  descriptor subset as the main run.
- **Expert:** the activity split needs one expert shared by all activities
  (`mixture.expert_scope=shared`, as in experiment 09); the scripts stop with a message otherwise.

## 7. Bring the results back

`report` writes `outputs/experiments/10_clustering/clustering_report.json`, with every screen's key
numbers in one file. It also checks that the latest `ablation` run covers every split and arm it was
asked for, with both primary tests. It prints `ABLATION INCOMPLETE: …` if not, instead of a silent
gap. Copy back over AnyDesk:
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

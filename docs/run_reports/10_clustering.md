# Experiment 10 (clustering): full-dataset run, 25–26 November 2026

Steps 4–7 of [scripts/clustering/README.md](../../scripts/clustering/README.md), run once on all of
DIPSER, by someone other than the experiment's owner. This report records how the run went, so the
results can be reviewed. It does not analyse them.

**For review, in short:**
1. **Code fixes:** two crashes were fixed in shared code, `src/cfmu/classrel.py` and
   `scripts/clustering/common.py`. Neither fix changes the values of windows that did not crash.
   Both are uncommitted (§4).
2. **The run differs from the guide in one place:** the two splits chose different best arms
   (activity: hdbscan, group: agglomerative), so each split was frozen on its own choice (§2).
3. **Both thresholds are at the top of the grid that was tested:** τ-desc 1.5 and τ-sync 0.9 (§2).
4. **Run time:** about 15 h of computing, about 30 h of elapsed time. Most of the difference is
   waiting for the fixes (§3).

## 1. What was run

| | |
|---|---|
| Machine | AMD Ryzen 5 3600 (6 cores, 12 threads), 16 GB RAM, Windows 11; CPU only |
| Environment | conda `fyp-local`, launched from WSL bash |
| Workers | 10 (threads minus 2) for `signal_screen`, `stability` and `candidates` |
| Config | `docs/experiments/09_vfoa.json` unchanged; config hash `b1c2d485d3db` |
| Caches | experiment 09's: `outputs/cache/tracks/2a11240a2d42/` and `outputs/cache/features/4c761244568f/`, 260 recordings (52 originals, 208 augmented copies). Step 3 (extraction) was not rerun |
| Runner | `scripts/experiments/10_clustering/run.sh` (run from `tmp/exp10/` at the time). It runs the guide's commands in order, skips finished steps, and reads the choices between steps from the outputs with `pick.py` (§2). Logs: `outputs/experiments/10_clustering/logs/<step>.log` |
| Outputs | `outputs/experiments/10_clustering/`, 1.2 GB in all, mostly `candidates/` |

## 2. Choices made between steps

The guide leaves these to the person running it. The runner picked them by fixed rules, written
before the run, so nothing was chosen by looking at results.

| Choice | Rule | Value |
|---|---|---|
| Channels | `signal_screen_channels.csv`, `pass_sync` true | `rf.total scb.read scb.write scb.m0.obj scb.hand_raising kg.speed hp.face_conf` (7) |
| Descriptor subset | `descriptor_subset` (no choice) | `trk.kg.events trk.rf.mag_01 trk.scb.write trk.scb.write.max trk.rf.mag_21 trk.scb.m0.obj.max trk.scb.hand_raising trk.kg.energy trk.rf.mag_22 trk.rf.mag_20` |
| τ-desc | `stability.csv`: `desc:subset`, agglomerative, usable, highest temporal ARI | **1.5** (temporal ARI 0.323; 1.0 gave 0.281, 0.8 gave 0.261) |
| τ-sync | `stability.csv`: `sync:*`, agglomerative, usable, highest temporal ARI averaged over the channels | **0.9** (0.184; 0.7 gave 0.178, 0.5 gave 0.174) |
| Best arm for `sync+best` | what the first `ablation` run chose, per split | activity: **hdbscan**; group: **agglomerative** |

**Points to check:**
- **Both τ are the largest value tested** (the grid is 0.4–1.5 for τ-desc and 0.3–0.9 for τ-sync).
  The temporal ARI was still rising there, so a larger τ might score higher still. For τ-sync,
  the differences between settings are small (0.174 to 0.184).
- **Larger τ means fewer, bigger clusters.** At τ-desc 1.5 that is 0.25 clusters per student; at
  1.0, 0.41.
- **The best arms differ by split.** The guide assumes one best arm and freezes it for runs 2 and 3.
  Instead, each split was run on its own (`--split activity --best hdbscan` and
  `--split group --best agglomerative`), each with and without the window features. That is four
  frozen runs instead of two. Each run's full result is in `ablation/<config hash>-<settings
  hash>.json`, and `report` collects all of them.

## 3. Time

### Computing time per step

| Step | What it does | Computing time | Guide's estimate |
|---|---|---|---|
| `signal_screen` | label-free screens 1–3, originals only | 11 min | |
| `event_check` | screen 4 | < 1 min | |
| `descriptor_subset` | screen 5 | < 1 min | |
| `stability` | screen 6, every space × method × τ | 25 min | |
| `candidates` | builds the label rows (caching `<stem>.labelwin.csv`), then the candidate features for 260 sessions | 1.1 h: 52 min for 195 sessions until crash 1, then 12 min for the remaining 65 | steps 4–5 together: 2–3 h (measured: ~1.7 h) |
| `feature_screen` | 18 folds (15 activity + 3 group), about 7 min each | 2.4 h, plus ~55 min in the crashed first attempt | |
| `importance --gate` | permutation importance: 39 feature groups × 5 repeats on 18 folds (5.6 h), then the gate check (16 min) | **5.9 h** | |
| `ablation`, first run | 7 arms, both splits | 2.7 h | |
| `ablation --split activity --best hdbscan`, 4 arms | | 19 min | |
| `… --no-window` | | 17 min | |
| `ablation --split group --best agglomerative`, 4 arms | | 11 min | |
| `… --no-window` | | 11 min | |
| `report` | | < 1 min | |
| **Total** | | **about 14.6 h**, including the crashed attempts (step 6 alone: ~13 h) | steps 4–7: 7–11 h |

**Notes:**
- **Steps 4–5 were faster than the guide's estimate; step 6 took longer.** `importance` takes half of step 6's time.
  It has to recompute every fold's model and score it many times over, which the guide's estimate
  did not account for.
- **What makes the first `ablation` run expensive:** it took 2.7 h. The frozen runs (4 arms, one
  split) took 11–19 minutes each, so most of the first run's time goes to the hdbscan and
  prototype arms and to choosing `sync+best`.
- **Building the label rows** comes first in every labelled step and takes about 10 min each time.
  Of the computing time above, more than an hour is that rebuild, repeated in each of the 11 labelled
  runs, counting the crashed attempts.
- **Workers:** only `signal_screen`, `stability` and `candidates` run in parallel. Steps 6–7 use
  one process, so a machine with more cores would not speed up most of the run.

### Elapsed time

| From | To | Duration | |
|---|---|---|---|
| 25 Nov 01:47 | 02:24 | 37 min | label-free screens (step 4) |
| 02:24 | 03:16 | 52 min | `candidates`: 195 of 260 sessions written, then crash 1 (§4) |
| 03:16 | 13:54 | 10.6 h | idle: overnight, then the fix |
| 13:54 | 14:06 | 12 min | `candidates` resumed: the remaining 65 sessions |
| 14:06 | ~15:00 | ~55 min | `feature_screen`: label rows, then crash 2 in its first fold |
| ~15:00 | 16:24 | ~1.4 h | idle: fix and check |
| 16:24 | 26 Nov 03:22 | 11.0 h | `feature_screen`, `importance`, first `ablation` |
| 03:22 | 05:50 | 2.5 h | idle: runner error R1 (§4) |
| 05:50 | 06:27 | 37 min | frozen activity runs, then `report`, which ran early because of runner error R2 |
| 06:27 | 07:00 | 33 min | idle: runner error R2 |
| 07:00 | 07:22 | 22 min | frozen group runs, then `report` again |
| **25 Nov 01:47** | **26 Nov 07:22** | **29.6 h** | **of which running ~14.6 h, idle ~15.0 h** |

Sources: log and output file timestamps, and the times of the runner's notes on each crash and fix
(Appendix A). Times marked ~ are estimates: the runner logs no start or end times (see §5).

## 4. Errors

### In the shared code: fixed, uncommitted, to review

| # | Where it stopped | Cause | Fix | Effect on results |
|---|---|---|---|---|
| 1 | `candidates`, shown after 60 sessions (195 were written; §5): `ValueError: operands could not be broadcast together with shapes (2,) (0,)` in `classrel._corr` | `lag_behind_class` shifts each student's series against the class's by up to `max_lag` frames (1 s at the camera's frame rate). A student seen for only a couple of frames in a window gives a series shorter than the shift, and the two slices came out different lengths | [src/cfmu/classrel.py](../../src/cfmu/classrel.py) `lag_behind_class`: shifts limited to `|k| < T`, the window length | none for normal windows (T > `max_lag`); the very short ones now get a value instead of crashing. The 195 sessions written before the fix had no such windows, or they would have crashed too |
| 2 | `feature_screen`: `ValueError: operands could not be broadcast together with shapes (243883,) (227601,)` | `common.attach` joins feature columns onto the label rows on (stem, window centre rounded to 1 ms, track). Some rating windows have centres 0.4 ms apart, from two labelers' entries at the same moment, so they share a key and the left join doubled those rows. All 260 label-window caches contain such pairs, 45,153 extra rows in all. It showed up in the prototype columns joined per fold | [scripts/clustering/common.py](../../scripts/clustering/common.py) `attach`: keep one row per key on the feature side. The two windows cover the same frames (0.4 ms is less than a frame), so their features are identical | none: no output had been written with the doubled rows. Checked on the real data: 227,601 label rows before and after both joins. `importance` and `ablation` use the same join |
| — | `signal_screen` printed numpy `RuntimeWarning: invalid value encountered in divide` | `lag1_autocorr` checks the whole series for zero spread, but correlates `x[:-1]` with `x[1:]`, and one of those can be flat (e.g. `1 1 1 1 2`). The result is NaN either way, and NaN is handled downstream | warning silenced in [src/cfmu/classrel.py](../../src/cfmu/classrel.py) `lag1_autocorr` | none (cosmetic) |

The 152 unit tests pass after both fixes.

**Not fixed:** the pandas `PerformanceWarning` flood; see §5.

### In the runner (`scripts/experiments/10_clustering/run.sh`): fixed

| # | What happened | Cause | Fix |
|---|---|---|---|
| R1 | the splits chose different arms; the runner went on to run `ablation --best` with no arm, which failed | `BEST=${BEST:-$(pick best)} \|\| exit 1` does not pass the substitution's failure through | read the per-split choice and run each split on its own (§2) |
| R2 | the frozen group-split runs were skipped without an error, and `report` ran and said "missing: none" | the runner loops over the splits by reading lines from its input. The Windows Python, launched from WSL, inherited that input and used up the second line | Python now gets no input (`< /dev/null`) |

**Suggestion for `report.py`:** R2 went unnoticed because `report` lists whatever ablation files exist,
so a run that is missing doesn't show. It could check that each split has a first run, a frozen run
and a `--no-window` run.

## 5. Console output

What the person running it sees, and what to make of it. In short, it is hard to follow progress or
spot failures from the console alone.

- **Warnings bury the progress lines:**
  - **pandas `PerformanceWarning`:** every labelled step (`candidates` onwards) prints
    `PerformanceWarning: DataFrame is highly fragmented`, from `scripts/train_models.py:62-66`, while
    building the label rows. Each warning takes two lines and repeats for every session, so there
    are hundreds before the first progress line. They are harmless.
    - **Fix (not applied):** `df = df.copy()` before those lines, or add the columns in one
      `pd.concat`.
  - **numpy `RuntimeWarning: invalid value encountered in divide`:** printed throughout
    `signal_screen`. It is harmless and now silenced (§4).
- **Progress lines:**
  - `signal_screen`, `stability` and `candidates` print one line per session (`[i/n] <stem>`);
  - `feature_screen`, `importance` and `ablation` print one per fold (`activity fold test_01+02: …`);
  - none of them prints times. The runner's `=== <step> (<date>)` header goes to the console only,
    not to the log. So elapsed time can't be recovered from the logs, which is why §3 relies on file
    timestamps. A timestamp on each progress line would make the next feasibility estimate much
    easier.
- **A crash in a worker shows up late and understates the progress:** in `candidates`, the console
  stopped at `[60/260]` and then printed the traceback, but 195 sessions had been written. The worker
  pool finishes every session it has already started before the error reaches the main process, and
  those sessions are not printed. `[i/n]` therefore counts reported sessions, not finished ones. The
  rerun's first line, `260 sessions (195 already done)`, gives the real count.
- **The tracebacks are clear and point at the failing line.** Both crashes in §4 were located from the
  console output alone.
- **Silent skips:** runner error R2 printed nothing, and `report` ended with `missing: none` (§4).
- **What each step prints at the end:** the summary tables:
  - `stability`: the best settings;
  - `feature_screen`: the strongest candidates;
  - `ablation`: every arm's held-out scores and the verdicts.

  They are also in `outputs/experiments/10_clustering/logs/<step>.log`.

## 6. Outputs

All in `outputs/experiments/10_clustering/` (not yet tracked by git):
- **`clustering_report.json`:** every screen's key numbers, plus the verdict of each `ablation` run.
- **Tables (`.csv` and `.json`) and plots (`.png`):**
  - `signal_screen_*`, `event_check*`, `event_times`, `descriptor_subset`, `stability`;
  - `feature_screen*`, `importance*`, `importance_gate`.
- **`ablation/`:** one JSON per run. There should be 5: the first run, then a frozen run and a
  `--no-window` run for each split.
  - `b1c2d485d3db-b508e8b2.json`: first run, 7 arms, both splits;
  - `b1c2d485d3db-5a521af8.json`: activity split, `--best hdbscan`;
  - `b1c2d485d3db-997ff639.json`: activity split, `--best hdbscan --no-window`;
  - `b1c2d485d3db-0c9e7718.json`: group split, `--best agglomerative`;
  - `b1c2d485d3db-69213f02.json`: group split, `--best agglomerative --no-window`.

  `ablation.csv` and `ablation_verdict.csv` hold only the latest run.
- **`candidates/4c761244568f-536205a1/`:** 260 session files and `params.json`, about 1.2 GB.
  Regenerable with step 5 and the values in §2. Probably worth a `.gitignore` entry rather than a
  commit.

`logs/` holds each step's log, with its printed summary (for example `ablation`'s tables and
verdicts), and the runner's `done/` markers. The `.log` files are ignored by git (`*.log`), like
every experiment's logs, so they stay on the machine that ran the experiment.

## Appendix A. The runner's notes

The runner's notes from during the run, with the console output of each crash. The pandas
`PerformanceWarning` lines (§5) are removed, and so are 55 of the 60 progress lines of the first
crash. The fixes are described in §4.

### First run (step 5, `candidates`), 25 November, morning

"This is what I woke up to":

```
260 sessions (0 already done), 10 worker(s) -> outputs\experiments\10_clustering\candidates\4c761244568f-536205a1
  [1/260] group_01__experiment_01__context_cam_02.aug4-4f270e71: 2752 student rows
  [2/260] group_01__experiment_01__context_cam_02.aug2-4f270e71: 2870 student rows
  [3/260] group_01__experiment_01__context_cam_02.aug1-4f270e71: 2877 student rows
  ...
  [59/260] group_01__experiment_04__context_cam_03.aug3-4f270e71: 9600 student rows
  [60/260] group_01__experiment_04__context_cam_03.aug4-4f270e71: 9599 student rows
concurrent.futures.process._RemoteTraceback:
"""
Traceback (most recent call last):
  File "C:\Users\abi\miniconda3\envs\fyp-local\Lib\concurrent\futures\process.py", line 255, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
  File "E:\Coding\Projects\fyp\scripts\clustering\candidates.py", line 110, in write_session
    df = session_candidates(cfg, stem, channels, p, out)
  File "E:\Coding\Projects\fyp\scripts\clustering\candidates.py", line 104, in session_candidates
    recs += [{"stem": stem, "t_centre": centre, **r} for r in window_candidates(sd, w, tracks, X, chans, p)]
  File "E:\Coding\Projects\fyp\scripts\clustering\candidates.py", line 69, in window_candidates
    s, wd = sync_features(S, max_lag=max_lag, threshold=p["sync_threshold"])
  File "E:\Coding\Projects\fyp\src\cfmu\classrel.py", line 142, in sync_features
    lag, peak = lag_behind_class(S, max_lag, min_common)
  File "E:\Coding\Projects\fyp\src\cfmu\classrel.py", line 103, in lag_behind_class
    r = _corr(a, b, min_common)
  File "E:\Coding\Projects\fyp\src\cfmu\classrel.py", line 31, in _corr
    m = np.isfinite(a) & np.isfinite(b)
ValueError: operands could not be broadcast together with shapes (2,) (0,)
"""

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  ...
  File "E:\Coding\Projects\fyp\scripts\clustering\candidates.py", line 154, in main
    for i, (job, n) in enumerate(map_sessions(write_session, jobs, args.workers), 1):
  File "E:\Coding\Projects\fyp\scripts\clustering\common.py", line 66, in map_sessions
    yield futures[f], f.result()
  ...
ValueError: operands could not be broadcast together with shapes (2,) (0,)
FAILED candidates (see tmp/exp10/candidates.log); fix and rerun
```

Fixed at 13:50 (§4, error 1). The rerun continued where it left off. Apart from the performance
warnings, it began with:

```
260 sessions (195 already done), 10 worker(s) -> outputs\experiments\10_clustering\candidates\4c761244568f-536205a1
```

### Second run (step 6, `feature_screen`), 25 November, afternoon

```
Traceback (most recent call last):
  ...
  File "E:\Coding\Projects\fyp\scripts\clustering\feature_screen.py", line 135, in main
    recs += screen_split(cfg, rows, split, args, proto_set, out, rng)
  File "E:\Coding\Projects\fyp\scripts\clustering\feature_screen.py", line 81, in screen_split
    trm, tem = m & tr & orig, m & te
ValueError: operands could not be broadcast together with shapes (243883,) (227601,)
FAILED feature_screen (see tmp/exp10/feature_screen.log); fix and rerun
```

Fixed and checked on the real data at 15:21 (§4, error 2). Rerun at 16:24.

### Third run (the frozen `ablation` runs), 26 November, early morning

Stopped at 03:22 on runner error R1 (§4): the splits chose different arms, and `ablation --best`
was called with no arm. Fixed at 05:50 and rerun. Runner error R2 then skipped the group-split runs.
That was fixed, and the group-split runs started at 07:00.

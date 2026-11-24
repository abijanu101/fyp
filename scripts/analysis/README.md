# One-off analyses behind the docs

These scripts produced specific tables in the docs. They are kept so that every number in the
docs can be traced to the code that made it. They are **frozen**: they ran against the code and
caches of their day and are not maintained like the pipeline scripts in `scripts/`. Some may no
longer run as is.

Run them from the repository root. The results, plots and logs they produced are archived under
`outputs/experiments/04_six_session/` and `outputs/experiments/05_per_labeler/`. The six-session
caches they read (8 s windows, older settings) are in `outputs/cache/e2e6/` (not shared). New runs
write scratch output to `tmp/`.

| Script | Produced | Cited in |
|---|---|---|
| `label_interp_compare.py` | how each interpolation mode changes window labels (8 s windows, all nine DIPSER experiments, 483 student-sessions) | guide 13 O25 |
| `e2e6/run_per_labeler.py` | the six-session per-labeler training run (results `2d4ef2c0164c`, `feda61a37624`) | experiment 05, `docs/misc/targets.md` §7.4 |
| `e2e6/extract_aug.py` | one augmented copy of each of the six sessions | experiment 05 |
| `e2e6/plot_per_labeler.py`, `e2e6/infer_per_labeler.py`, `e2e6/viewer_snapshot.py`, `e2e6/tune_per_labeler.py` | the six-session per-labeler plots, inference smoke tests, viewer snapshot, expert tuning | experiment 05 |
| `e2e6/run.sh` | the six-session end-to-end run (extract, train, infer on an unseen session) | experiment 04 §1 |
| `e2e6/benchmark_metrics.py` | the six-session run scored with the benchmark papers' metrics | experiment 04 |
| `e2e6/analyse.py` | feature-level analysis of the six-session run | experiment 04 |
| `recording_runtime.py` | duration, frame count, frame rate and gaps of the 52 in-scope recordings (`outputs/dataset/recordings.json`), or with `--all` of all 135 recordings (`recordings_all.json`) | experiment 01 §1.1 |

Analyses that are part of the current pipeline live in `scripts/` itself, for example
`scripts/labels/label_agreement.py` (label agreement and the human ceiling, experiment 06 Part B) and
`scripts/labels/label_baselines.py` (`docs/misc/targets.md`, `docs/misc/benchmarks.md`).

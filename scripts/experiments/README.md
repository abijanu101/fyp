# Experiment scripts

The scripts that ran each experiment, kept so a run can be repeated or extended. Run them from the
repository root: `bash scripts/experiments/<NN_name>/<script>.sh`, or
`PYTHONPATH=. python scripts/experiments/<NN_name>/<script>.py`.

- **Run chains** (`.sh`) call `train_models` (or another `scripts.*` module) once per arm with the
  experiment's frozen config (`docs/experiments/NN_name.json`) plus `--set` overrides. They are
  resumable: an arm whose log already holds its `config <hash>` line is skipped.
- **Logs** go to `outputs/experiments/<NN_name>/logs/`. That folder already holds the original runs'
  logs, so rerunning a finished chain skips every arm.
- **Table builders** (`.py`) read the hashes and call `scripts/reporting/results_table.py`. They write
  their Markdown to `tmp/`, and the result goes into the experiment's document by hand.
- **Hard-coded paths:** the Python path in the run chains is the author's
  (`/c/Users/abi/miniconda3/envs/fyp-local/python.exe`). Change `P=` to your own environment's Python.
- **Configs:** three snapshots were renamed after their runs: `06_full_pipeline.json`,
  `07_features.json` and `08_features.json` are now `06_full_pipeline_and_labels.json`, `07_knn.json`
  and `08_pca.json`. Their contents are unchanged, so the hashes are the same. The scripts here use
  the current names.

| Folder | Contents |
|---|---|
| `06_full_pipeline/` | the experiment 06 retrains (`run_all`, `run_lane`, `run_models`, `run_one`, `run_rest`) and `regen.sh`, which rebuilds its Part 2 tables |
| `06_08_reruns/` | reruns across experiments 06–08: `rerun_06_07.sh` (training-entry baselines and spread metrics), `rerun_cindex.sh` (C-index and saved predictions), and the held-out timelines of the appendices (`timeline_variants.py` checks the variants and writes `timelines.sh`) |
| `07_knn/` | `knn.sh` (the k scan) and `pca.sh` (the first PCA arms) |
| `08_pca/` | `rf.sh`, `knn.sh` (k = 500), `knn5.sh`, `extra.sh`, and `build_head.py`, which builds one model's tables |
| `09_vfoa/` | `arms.sh` (every arm), `collect.py` (`all_arms.csv`), `tables.py` (the write-up's tables), `wallclock.py` (runtimes) |
| `10_clustering/` | `run.sh`, steps 4–7 of `scripts/clustering/README.md` with the choices between steps made by `pick.py`: the record of the **first** full run (25–26 November 2026; run report `docs/run_reports/10_clustering.md`). It uses `ablation` options removed since (`--best`, `--no-window`), so it no longer runs as is. The rerun, with the fixed design, is one cross-platform command: `python -m scripts.clustering.run_all` (resumable; `scripts/clustering/README.md`) |

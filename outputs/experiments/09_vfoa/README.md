# Experiment 09: runs (24 November 2026)

Write-up: `docs/experiments/09_vfoa.md`. Results files are in `outputs/results/<hash>.json`, predictions
in `outputs/predictions/<hash>.csv.gz`.

- `all_arms.csv`: every arm (176) with its hash, model, comparison, input and context, and the
  held-out (`split=test`) and training (`split=train`) metrics on both label sets (`track`).
- `logs/<arm>.log`: each run's training log; `arms.log`: the runner's output (arm -> hash, start times).
- The scripts are in `scripts/experiments/09_vfoa/`:
  - `arms.sh`: the runner (`all` = comparisons 1-3 with 08's four models, `knnreg` = k-NN regression
    at k = 5 and 50 and the classifier at k = 50). Arm names: `<model>_<comparison>_<input>[_ctx]`;
  - `collect.py` builds `all_arms.csv`, `tables.py` prints the write-up's tables, and `wallclock.py`
    the runtimes.

  The runs were made from `tmp/exp09/`; the scripts' paths now point here.
- `ceiling_lmh.log`: the three-level human ceiling (`outputs/labels/agreement_lmh.json`).
- `pilot/`: the pose-input pilot's overlays (Appendix B).

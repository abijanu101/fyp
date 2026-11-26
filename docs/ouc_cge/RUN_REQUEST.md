# Run request: OUC-CGE through the pipeline

For: Abi (GPU machine). From: Ahmed. Branch `ouc-cge-benchmark`. Background and results so far:
[`docs/experiments/ouc_cge_benchmark.md`](../experiments/ouc_cge_benchmark.md).

**Goal:** extract the pipeline's features for every OUC-CGE clip and score the classifiers on the
official split. Nothing in the existing DIPSER code path changes; the new code is additive.

## 0. Setup (once)

1. Check out the branch (from the patch or the pushed branch), in `fyp-local`:
   ```
   git checkout ouc-cge-benchmark
   python -m unittest discover -s tests -t .      # expect: all OK (5 new tests in test_ouc_cge.py)
   python -m scripts.setup --check                 # yolo11n, ReID, SCB, lib/yolov7 must be [ok]
   ```
2. **Data (about 28 GB of video).** Either copy Ahmed's folder (USB drive), or download from OSF with
   Ahmed's `_download.ps1` / `_extract.ps1`. Put it at `data/OUC-CGE/` so that it contains
   `high/`, `mid/`, `Low/` as extracted (nested folders are fine: the code searches them).
3. Split files: copy Ahmed's `OUC-CGE-main/` folder to `data/OUC-CGE-main/` (`train.csv`, `val.csv`,
   `test.csv`). Elsewhere: add `--set dataset.root=<path> dataset.split_dir=<path>` to every command.

## 1. Smoke test (about 30 clips; minutes)

```
python -m scripts.ouc_cge.extract --limit 30
python -m scripts.ouc_cge.benchmark
```
Check: no `FAILED` lines (else send `outputs/cache/features/<hash>/ouc_cge/errors.txt`); most clips
show several tracklets; note the seconds per clip in the log. Open one clip's
`ouc_cge__*.manifest.json` for the stage timings.

## 2. Full extraction (long: resumable)

```
python -m scripts.ouc_cge.extract --reuse --max-hours 10
```
Rerun the same command to continue where it stopped. If the GPU has room for two processes, run two
terminals with `--shard 1/2` and `--shard 2/2` (same flags otherwise). Estimate: 7,705 clips × ~27
frames each at 5 fps ≈ 210,000 frames; at the repo's ~185 ms per frame that is about 11 h for one
process (the 116 broken clips fail fast and are logged).

## 3. Benchmark (seconds)

```
python -m scripts.ouc_cge.benchmark
python -m scripts.ouc_cge.benchmark --set ouc_cge.min_duration_s=3
```

## Send back

1. `outputs/ouc_cge/results/*.json` (both benchmark runs) and the printed tables.
2. `outputs/cache/features/<feature hash>/ouc_cge/` zipped (the clip CSVs: small; lets Ahmed rerun the
   benchmark without a GPU). The tracks cache is not needed.
3. `errors.txt` if it exists, the GPU name, total time, and the commit hash you ran
   (`git rev-parse --short HEAD`).

## Not in this request

- SlowFast (the dataset authors' model): no weights are released, so it needs training from
  Kinetics weights. Only if the GPU has the memory; a separate request with a tiny trial first.

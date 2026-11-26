# Scripts

Run everything from the repository root as a module, e.g. `python -m scripts.train_models`. Each
script's docstring gives its full usage.

| Folder | Script | Purpose |
|---|---|---|
| (top) | `setup` | fetch every model the pipeline needs (first thing on a fresh clone; `--check` to report only) |
| | `run_pipeline` | extract features (frame pass and feature pass) for every session; inference on one camera |
| | `train_models` | leave-one-group-out training and evaluation, and the final model bundle |
| | `parallel_runs` | the long jobs (extraction, tuning) in parallel, resumable |
| `tuning/` | `tune_tracker` | detector confidence and tracker parameters against the manual boxes |
| | `tune_models` | gate and expert hyperparameters, nested leave-one-group-out |
| | `scan_detector` | which detector threshold keeps every student |
| | `benchmark` | per-stage timing per tracker variant |
| `reporting/` | `results_table` | the standard result tables (profile and metrics, held-out and train, per labeler) |
| | `compare_results` | two runs side by side, with confusion matrices |
| | `plot_results` | held-out engagement curves per session |
| `labels/` | `label_agreement` | labeler agreement (Krippendorff's α) and the human ceiling (experiment 06 Part 1) |
| | `label_baselines` | constant-prediction baselines on every published target (`docs/misc/benchmarks.md`) |
| `clustering/` | `signal_screen` | which per-frame channels and descriptors clear the noise floor, vary within a track, and are synchronous beyond chance (guide 15 §15.9, A1–A3) |
| | `event_check` | whether the class moves together at events (A4) |
| | `descriptor_subset` | the descriptors to cluster on, chosen without labels (A5) |
| | `stability` | how stable each clustering and threshold is (A6) |
| | `candidates` | every candidate class-relative feature on the label windows (B7) |
| | `feature_screen` | candidates against the ratings, training folds only (B8) |
| | `importance` | held-out permutation importance, per-labeler model and gate (B9) |
| | `ablation` | the clustering experiment's arms and the test of §15.6 (B10) |
| | `report` | every clustering result in one JSON file; the full-dataset run guide is `clustering/README.md` |
| `analysis/` | `feature_study` | the feature vector against every labeler (experiment 08 §1) |
| | `experiment_decoders`, `label_interp_compare`, `e2e6/` | frozen one-off analyses behind specific doc tables (`analysis/README.md`) |
| | `assoc_compare` | `association.method` per_frame against mean_box on a config's cached tracks |
| `experiments/` | `<NN_name>/` | each experiment's run chains and table builders, run with `bash` or `python` from the root (`experiments/README.md`) |
| `data_visualization/` | `output_viewer`, `augment_samples`, `inspect_labels`, viewers | looking at the data and the pipeline's output |
| `downloading/` | `dipser`, `scb`, `validate_dipser_remote` | getting the dataset, and the whole SCB dataset (`setup` fetches the one checkpoint used) |

# Experiment 07: runs (frozen 18 November 2026)

Results files are in `outputs/results/<hash>.json`, predictions in `outputs/predictions/<hash>.csv.gz`;
the training logs are in `logs/`. Each arm is
`python -m scripts.train_models --config docs/experiments/07_knn.json --set` plus the setting shown
(docs/experiments/07_knn.md).

| Arm | Setting | Results | Log |
|---|---|---|---|
| k = 1 | `expert.name=knn` `expert.params={"n_neighbors": 1, "n_jobs": 4}` | `0f2d8f2841a9` | `knn_k1.log` |
| k = 5 | `expert.name=knn` `expert.params={"n_neighbors": 5, "n_jobs": 4}` | `e79c4c117090` | `knn_k5.log` |
| k = 25 | `expert.name=knn` `expert.params={"n_neighbors": 25, "n_jobs": 4}` | `75a782eb8951` | `knn_k25.log` |
| k = 50 | `expert.name=knn` `expert.params={"n_neighbors": 50, "n_jobs": 4}` | `efa51da0cd61` | `knn_k50.log` |
| k = 100 | `expert.name=knn` `expert.params={"n_neighbors": 100, "n_jobs": 4}` | `2fd336ace408` | `knn_k100.log` |
| k = 250 | `expert.name=knn` `expert.params={"n_neighbors": 250, "n_jobs": 4}` | `dbc58c4983ba` | `knn_k250.log` |
| k = 500 | `expert.name=knn` `expert.params={"n_neighbors": 500, "n_jobs": 4}` | `2fcb719b3a54` | `knn_k500.log` (chosen) |
| k = 1000 | `expert.name=knn` `expert.params={"n_neighbors": 1000, "n_jobs": 4}` | `386791e78fd1` | `knn_k1000.log` |

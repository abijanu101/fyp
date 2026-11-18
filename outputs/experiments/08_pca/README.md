# Experiment 08: runs (frozen 18 November 2026)

Results files are in `outputs/results/<hash>.json`, predictions in `outputs/predictions/<hash>.csv.gz`;
the training logs are in `logs/`; the figures and CSVs next to this file are §1's vector study and
Appendices A–B. Each arm is
`python -m scripts.train_models --config docs/experiments/08_pca.json --set` plus the setting shown
(docs/experiments/08_pca.md §2). `engagement.late_context` is the key's old name; it is now
`engagement.use_class_context` (the runs were made, and hashed, under the old name). Arms without
PCA reuse experiment 07's (k-NN) or the base run (logistic regression), so a log may be shared.

| Arm | Setting | Results | Log |
|---|---|---|---|
| logistic regression: all 95 PCs | `expert.pca=95` | `a2096ec10bef` | `lr_pca_all.log` |
| logistic regression: all 95 | – (the config) | `b3fbc8430d44` | `base.log` |
| logistic regression: PCA 5 | `expert.pca=5` | `5580383edd8d` | `pca5.log` |
| logistic regression: PCA 10 | `expert.pca=10` | `9990d1832233` | `pca10.log` |
| logistic regression: PCA 20 | `expert.pca=20` | `69cc580dcd9a` | `pca20.log` |
| logistic regression: PCA 90% | `expert.pca=0.9` | `617bf1d8b13b` | `pca90.log` |
| logistic regression: PCA 95% | `expert.pca=0.95` | `9cd26eabef2a` | `pca95.log` |
| logistic regression: all 27 PCs | `engagement.late_context=false` `expert.pca=27` | `5d86ad5c10cf` | `lr_nctx_pca_all.log` |
| logistic regression: all 27 | `engagement.late_context=false` | `c724586518c0` | `nctx.log` |
| logistic regression: PCA 5 | `engagement.late_context=false` `expert.pca=5` | `049b908f551c` | `nctx_pca5.log` |
| logistic regression: PCA 10 | `engagement.late_context=false` `expert.pca=10` | `60fc84b50016` | `nctx_pca10.log` |
| logistic regression: PCA 20 | `engagement.late_context=false` `expert.pca=20` | `b4476ab7dc49` | `lr_nctx_pca20.log` |
| logistic regression: PCA 90% | `engagement.late_context=false` `expert.pca=0.9` | `84227cefa9b3` | `nctx_pca90.log` |
| logistic regression: PCA 95% | `engagement.late_context=false` `expert.pca=0.95` | `41511c48f91c` | `lr_nctx_pca95.log` |
| random forest: all 95 PCs | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `expert.pca=95` | `a1e44780a01f` | `rf_pca_all.log` |
| random forest: all 95 | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` | `e36bbdfd5379` | `rf_base.log` |
| random forest: PCA 5 | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `expert.pca=5` | `4b2ab09d6733` | `rf_pca5.log` |
| random forest: PCA 10 | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `expert.pca=10` | `783d388f9d4b` | `rf_pca10.log` |
| random forest: PCA 20 | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `expert.pca=20` | `be9ae5a21b9f` | `rf_pca20.log` |
| random forest: PCA 90% | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `expert.pca=0.9` | `998051404e42` | `rf_pca90.log` |
| random forest: PCA 95% | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `expert.pca=0.95` | `c1e4357404df` | `rf_pca95.log` |
| random forest: all 27 PCs | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `engagement.late_context=false` `expert.pca=27` | `c36f89504032` | `rf_nctx_pca_all.log` |
| random forest: all 27 | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `engagement.late_context=false` | `ef5d328b5033` | `rf_nctx.log` |
| random forest: PCA 5 | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `engagement.late_context=false` `expert.pca=5` | `a42cc72b352d` | `rf_nctx_pca5.log` |
| random forest: PCA 10 | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `engagement.late_context=false` `expert.pca=10` | `bca4287f6f19` | `rf_nctx_pca10.log` |
| random forest: PCA 20 | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `engagement.late_context=false` `expert.pca=20` | `9ee0dffa4325` | `rf_nctx_pca20.log` |
| random forest: PCA 90% | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `engagement.late_context=false` `expert.pca=0.9` | `0967841280a6` | `rf_nctx_pca90.log` |
| random forest: PCA 95% | `expert.name=random_forest` + `expert.params={"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20, "n_jobs": 4}` `engagement.late_context=false` `expert.pca=0.95` | `1a6494971aeb` | `rf_nctx_pca95.log` |
| k-NN, k = 500: all 95 PCs | `expert.name=knn` `n_neighbors` 500 `expert.pca=95` | `9ecaa0ec1591` | `knn_pca_all.log` |
| k-NN, k = 500: all 95 | `expert.name=knn` `n_neighbors` 500 | `2fcb719b3a54` | `knn_k500.log` |
| k-NN, k = 500: PCA 5 | `expert.name=knn` `n_neighbors` 500 `expert.pca=5` | `f4b118c8b4cd` | `knn_pca5.log` |
| k-NN, k = 500: PCA 10 | `expert.name=knn` `n_neighbors` 500 `expert.pca=10` | `742c43adc8f7` | `knn_pca10.log` |
| k-NN, k = 500: PCA 20 | `expert.name=knn` `n_neighbors` 500 `expert.pca=20` | `54378931db3d` | `knn_pca20.log` |
| k-NN, k = 500: PCA 90% | `expert.name=knn` `n_neighbors` 500 `expert.pca=0.9` | `9545f014ad42` | `knn_pca90.log` |
| k-NN, k = 500: PCA 95% | `expert.name=knn` `n_neighbors` 500 `expert.pca=0.95` | `7cfcbad29dcb` | `knn_pca95.log` |
| k-NN, k = 500: all 27 PCs | `expert.name=knn` `n_neighbors` 500 `engagement.late_context=false` `expert.pca=27` | `47550b2e37a7` | `knn_nctx_pca_all.log` |
| k-NN, k = 500: all 27 | `expert.name=knn` `n_neighbors` 500 `engagement.late_context=false` | `95a0f17474d2` | `knn_nctx.log` |
| k-NN, k = 500: PCA 5 | `expert.name=knn` `n_neighbors` 500 `engagement.late_context=false` `expert.pca=5` | `1c47d07d9025` | `knn_nctx_pca5.log` |
| k-NN, k = 500: PCA 10 | `expert.name=knn` `n_neighbors` 500 `engagement.late_context=false` `expert.pca=10` | `2fa68c93dbb6` | `knn_nctx_pca10.log` |
| k-NN, k = 500: PCA 20 | `expert.name=knn` `n_neighbors` 500 `engagement.late_context=false` `expert.pca=20` | `4551ccc115df` | `knn_nctx_pca20.log` |
| k-NN, k = 500: PCA 90% | `expert.name=knn` `n_neighbors` 500 `engagement.late_context=false` `expert.pca=0.9` | `659b59ec05eb` | `knn_nctx_pca90.log` |
| k-NN, k = 500: PCA 95% | `expert.name=knn` `n_neighbors` 500 `engagement.late_context=false` `expert.pca=0.95` | `02e4793a852a` | `knn_nctx_pca95.log` |
| k-NN, k = 5: all 95 PCs | `expert.name=knn` `n_neighbors` 5 `expert.pca=95` | `6015fa0595bc` | `knn5_pca_all.log` |
| k-NN, k = 5: all 95 | `expert.name=knn` `n_neighbors` 5 | `e79c4c117090` | `knn_k5.log` |
| k-NN, k = 5: PCA 5 | `expert.name=knn` `n_neighbors` 5 `expert.pca=5` | `4b839fb3be18` | `knn5_pca5.log` |
| k-NN, k = 5: PCA 10 | `expert.name=knn` `n_neighbors` 5 `expert.pca=10` | `32b20efcc0c1` | `knn5_pca10.log` |
| k-NN, k = 5: PCA 20 | `expert.name=knn` `n_neighbors` 5 `expert.pca=20` | `229aa6b96c09` | `knn5_pca20.log` |
| k-NN, k = 5: PCA 90% | `expert.name=knn` `n_neighbors` 5 `expert.pca=0.9` | `8720768ba51a` | `knn5_pca90.log` |
| k-NN, k = 5: PCA 95% | `expert.name=knn` `n_neighbors` 5 `expert.pca=0.95` | `88636418db8d` | `knn5_pca95.log` |
| k-NN, k = 5: all 27 PCs | `expert.name=knn` `n_neighbors` 5 `engagement.late_context=false` `expert.pca=27` | `f47fa833d860` | `knn5_nctx_pca_all.log` |
| k-NN, k = 5: all 27 | `expert.name=knn` `n_neighbors` 5 `engagement.late_context=false` | `bcbc82e02a33` | `knn5_nctx.log` |
| k-NN, k = 5: PCA 5 | `expert.name=knn` `n_neighbors` 5 `engagement.late_context=false` `expert.pca=5` | `f6b45f12fea1` | `knn5_nctx_pca5.log` |
| k-NN, k = 5: PCA 10 | `expert.name=knn` `n_neighbors` 5 `engagement.late_context=false` `expert.pca=10` | `370d203d04a5` | `knn5_nctx_pca10.log` |
| k-NN, k = 5: PCA 20 | `expert.name=knn` `n_neighbors` 5 `engagement.late_context=false` `expert.pca=20` | `8f5cfb454394` | `knn5_nctx_pca20.log` |
| k-NN, k = 5: PCA 90% | `expert.name=knn` `n_neighbors` 5 `engagement.late_context=false` `expert.pca=0.9` | `943933634b4a` | `knn5_nctx_pca90.log` |
| k-NN, k = 5: PCA 95% | `expert.name=knn` `n_neighbors` 5 `engagement.late_context=false` `expert.pca=0.95` | `2cc43b4528c2` | `knn5_nctx_pca95.log` |

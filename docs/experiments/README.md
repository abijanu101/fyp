# Experiments

Records of what we did to the data and the components, why, and what we found. Where
`docs/guide/` explains how the code works, these documents explain the decisions behind it and
the evidence for them. They are the raw material for the write-up.

| # | Document | Status |
|---|---|---|
| 1 | [DIPSER: dataset work and manual annotation](01_dipser.md) | current; label analysis to get its own document |
| 2 | [BoxMOT: detection and tracking](02_boxmot.md) | done: D3 sets `detector.conf` 0.1098 (augmented); T2 picks BoT-SORT + ReID + CMC (0.930 IDF1 on augmented data, against 0.914 without ReID and 0.877 for StrongSORT) |
| 3 | [SCB behaviour detection](03_scb.md) | current; feature-family ablation pending |
| 4 | [The first full-length run: six sessions, end to end](04_six_session_run.md) | done: the pipeline works at full length; engagement at chance, but the features carry consistent within-session signal the models do not use; directions for improvement |
| 5 | [Per-labeler training: first runs on the six sessions](05_per_labeler.md) | the new default path runs end to end; at chance (one session per activity); one expert made 59% of the entries; one augmented copy raised expert-track AUC in every fold (+0.02–0.03), still near chance |
| 6 | [The full pipeline experiment: mixture of experts against a single decoder](06_full_pipeline.md) | done: the mixture of experts and the single decoder tie (expert-entry AUC 0.526 / 0.527); both are barely above chance, so the bottleneck is the features. One augmented copy adds nothing (AUC 0.527 / 0.525 without); 4 copies running. Confusion matrices: 96% of entries predicted 3 |

Status as of 16 November 2026.

Conventions: every number states the configuration or command that produced it; anything not
yet verified on data says so.

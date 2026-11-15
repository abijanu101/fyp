# Experiments

Records of what we did to the data and the components, why, and what we found. Where
`docs/guide/` explains how the code works, these documents explain the decisions behind it and
the evidence for them. They are the raw material for the write-up.

| # | Document | Status |
|---|---|---|
| 1 | [DIPSER: dataset work and manual annotation](01_dipser.md) | current; label analysis to get its own document |
| 2 | [BoxMOT: detection and tracking](02_boxmot.md) | D1–D2 done (0.22 drops no student; optimum 0.166), D3 to be run; T1 done: BoT-SORT 0.905 vs StrongSORT 0.878 IDF1; T2 (augmented, three trackers) to be run |
| 3 | [SCB behaviour detection](03_scb.md) | current; feature-family ablation pending |
| 4 | [The first full-length run: six sessions, end to end](04_six_session_run.md) | done: the pipeline works at full length; engagement at chance, but the features carry consistent within-session signal the models do not use; directions for improvement |
| 5 | [Per-labeler training: first runs on the six sessions](05_per_labeler.md) | the new default path runs end to end; at chance (one session per activity); one expert made 59% of the entries; one augmented copy raised expert-track AUC in every fold (+0.02–0.03), still near chance |
| 6 | [The full pipeline experiment: mixture of experts against a single decoder](06_full_pipeline.md) | planned: all 52 session-cameras with augmented copies, two arms (`mixture.expert_scope`); compared with `scripts/compare_results.py` |

Status as of 16 November 2026.

Conventions: every number states the configuration or command that produced it; anything not
yet verified on data says so.

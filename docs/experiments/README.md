# Experiments

Records of what we did to the data and the components, why, and what we found. Where
`docs/guide/` explains how the code works, these documents explain the decisions behind it and
the evidence for them. They are the raw material for the write-up.

| # | Document | Status |
|---|---|---|
| 1 | [DIPSER: dataset work and manual annotation](01_dipser.md) | current; label analysis to get its own document |
| 2 | [BoxMOT: detection and tracking](02_boxmot.md) | detector threshold verified; full-dataset tracker experiments in progress (BoT-SORT 11/20, StrongSORT 6/20 pending restart, BoT-SORT + ReID queued) |
| 3 | [SCB behaviour detection](03_scb.md) | current; feature-family ablation pending |
| 4 | [Mixture of experts](04_mixture_of_experts.md) | not yet written |
| 5 | [The first full-length run: six sessions, end to end](05_six_session_run.md) | done: the pipeline works at full length; engagement at chance, but the features carry consistent within-session signal the models do not use; directions for improvement |

Status as of 15 November 2026.

Conventions: every number states the configuration or command that produced it; anything not
yet verified on data says so.

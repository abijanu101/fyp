# Per-labeler training: first runs on the six sessions

*16 November 2026. The first runs of the per-labeler path (`engagement.labelers = "per_labeler"`,
`docs/misc/targets.md` §7, guide 07 §7.9) on the six sessions of 04. Superseded at full scale by
06; kept for what it first showed.*

**Setup.** One logistic-regression model per labeler, trained on 8 s windows centred on that
labeler's rating entries, scored per entry, leaving one group out. 4,769 entries in total. Results
`2d4ef2c0164c` (no augmentation) and `1d448bd5a00c` (one augmented copy); outputs in
`outputs/experiments/05_per_labeler/`; scripts in `scripts/analysis/e2e6/`.

**What it showed:**

1. **Everything is at chance, by construction.** AUC was 0.49–0.52 and Spearman within ±0.06. With
   one session per activity, a held-out group's activities never appear in training (04 §2).
2. **One expert dominates the labels.** `labeler_01` made 59% of all entries and two thirds of the
   expert entries, 10–25 times the training rows of the other labelers. A pooled expert score is
   mostly `labeler_01`'s, so every result is now also reported per labeler (06).
3. **Constant 3 has the lowest MAE of any predictor** (0.605 on expert entries, against 0.66–0.70
   for every model). This was the first sign of the problem 06 Part 1 explains: on these labels, MAE
   rewards predicting the middle.
4. **Per-activity experts collapse to constant 3 for unseen activities.** With the true activity as
   context, an activity with no training rows gives a flat 3 (guide 13, O12).
5. **One augmented copy helped here and nowhere else.** Expert AUC rose 0.507 → 0.532, by
   +0.02–0.03 in every fold. At full scale the gain is gone (06 §2.3): with so little training data,
   any extra variation helps.

The tools on the new path (inference, held-out plots, the output viewer, expert tuning) were first
run here; 06 uses them at scale.

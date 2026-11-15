# Proposed metrics, for review

Metrics that are not in the code but may be worth adding. Each entry says what the metric measures,
how it would be computed, what it was used for before (if anything), and the questions to answer
before it goes back in. Nothing here is computed by any script.

## 1. `retention_r`: does the curve move with the labels?

**Status.** Removed from the code on 16 November 2026 (`scripts/train_models.py`,
`scripts/experiment_decoders.py`), because it was not understood well enough to be reported.
Results computed with it before then remain in `docs/experiments/05_six_session_run.md` (§3, §3.1,
§6.4); they are a historical record, not a current metric.

### What it measured

The question a retention graph exists to answer: *when the class's engagement rises and falls during
a session, does the predicted curve rise and fall with it?*

For each held-out session it took two curves over the session's windows:

- the **predicted curve**: the model's engagement score per window;
- the **labelled curve**: the labelled share of engaged students per window (engaged students ÷
  labelled students, Chapter 12 §12.1).

and computed their **Pearson correlation** r. `retention_r` was the mean of r over the held-out
sessions. 1 = the curves move perfectly together; 0 = no relation (chance); −1 = they move in
opposite directions.

### A worked example

Five windows of one session:

| Window | Labelled share | Model A | Model B | Model C |
|---|---|---|---|---|
| 1 | 0.4 | 0.50 | 0.40 | 0.55 |
| 2 | 0.6 | 0.70 | 0.60 | 0.55 |
| 3 | 0.8 | 0.90 | 0.80 | 0.55 |
| 4 | 0.6 | 0.70 | 0.60 | 0.55 |
| 5 | 0.4 | 0.50 | 0.40 | 0.55 |

- **Model A** is always 0.1 too high, but rises and falls exactly with the labels: r = **1**.
- **Model B** matches the labels exactly: r = **1** as well.
- **Model C** is flat: r is **undefined** (a constant has no correlation); such sessions were skipped.

So `retention_r` scored only the **shape** of the curve, never its height. Model A and model B are
equal by this metric; MAE would separate them (0.1 vs 0).

### Why it was computed within sessions

The labelled share varies for two reasons: some sessions are more engaged than others, and
engagement moves within a session. On the six-session run 69% of the variance was *between*
sessions (05 §6.1). A metric pooled over sessions mostly rewards getting each session's level right
(Chapter 12 §12.6: the same predictions gave AUC 0.85 pooled, 0.51 per fold). Computing r inside
each session removes the level and keeps only the within-session movement.

### Known issues, recorded before removal

- **A NaN anywhere voids the session.** The correlation used `np.corrcoef`, which returns NaN if any
  window's share is NaN (a window with no labelled student); one such window made the session's r,
  and the fold's `retention_r`, NaN (Chapter 13, O20).
- **It counted windows that had no window label.** Windows with fewer than `min_labeled_students`
  were excluded from AUC and F1 but included in `retention_r` (Chapter 13, O7).
- **Few windows per session make r noisy.** A 5-minute session at 8-second windows has 73 windows;
  the labelled share changed little within most sessions (±0.1), so r rests on small movements and
  labeler noise.
- **It ignores the curve's height.** A model can be badly calibrated and still score 1.

### Questions before it comes back

1. **Which curves?** The planned retention graph has one curve per labeler model and their median
   (`docs/misc/targets.md`, §7.1 item 8). Correlate each labeler model's curve with that labeler's
   own labels, or the median curve with the median of the labels, or both?
2. **Against what?** The labelled share was built from window labels thresholded at `tau_att`. With
   the planned normalised-rating target (targets.md §7.1 item 7), the natural comparison is the
   predicted rating curve against the labeler's interpolated rating curve.
3. **Per window or per label?** Evaluation is to be per label (targets.md §7.1 item 2). A
   correlation needs a series; per label, the series would be the labeler's entries in time order
   and the predictions at them.
4. **Pearson or Spearman?** Spearman (ranks) is less sensitive to a few extreme windows.
5. **Alongside an error measure?** Shape (correlation) and height (MAE on the 1–5 scale) answer
   different questions; reporting both avoids model A and model B looking the same.

# The full pipeline experiment: mixture of experts against a single decoder

*Planned. Runs after the tracker is chosen (`02_boxmot.md`, D3 and T2) and the full extraction.*

The first full-scale training of the per-labeler pipeline on all 52 session-cameras, with
augmented training copies, in two arms that differ in one setting: whether engagement is decoded
by a **mixture of activity experts** weighted by the activity gate, or by **a single decoder** for
every activity. It answers two questions at once: how well the pipeline does at full scale, and
whether the instructional-activity mixture of experts earns its cost.

## 1. Why one experiment

Both arms share everything else: the frame pass, the features, the label-centred windows, the
targets, the folds and the seed. Every rating entry is scored the same way in both, in its own
window centred on it (per-label evaluation, `docs/misc/targets.md` §7), so the two arms are compared
entry for entry. The windows themselves (4 s, `window.length_s`) are the unit a later temporal model
will consume (FYP-2); here each window is scored on its own.

## 2. The arms

| Arm | Setting | What decodes engagement |
|---|---|---|
| **MoE** | `mixture.expert_scope = "per_experiment"` (the config default) | per labeler, six activity experts, each trained only on its activity's windows, weighted by the gate's predicted activity (`mixture.mode = "soft"`) |
| **Single decoder** | `mixture.expert_scope = "shared"` | per labeler, one expert trained on every activity; the gate's prediction has no effect on engagement (it is still trained, for the graph's activity shading) |

Everything else is the default config: per-labeler models (`engagement.labelers = "per_labeler"`),
`train_with_interp` with step labels, 4-second windows, logistic-regression experts, `tau_att` 3.

Each run evaluates both variants internally (`configured` and `shared_expert` in its results), so
either run alone already contains the comparison; the two runs give each arm its own final model
bundle and retention graphs.

## 3. Commands

```
# 1. once the tracker is fixed: frame pass + features for every session, then the augmented copies
python -m scripts.run_pipeline extract --set augmentation.copies=1

# 2. the two arms (each: leave-one-group-out evaluation + final bundle)
python -m scripts.train_models --set augmentation.copies=1 labels.tau_att=3
python -m scripts.train_models --set augmentation.copies=1 labels.tau_att=3 mixture.expert_scope='"shared"'

# 3. compare (results files are named by config hash; train_models prints it)
python -m scripts.compare_results outputs/results/<MoE hash>.json outputs/results/<single hash>.json --names MoE single
python -m scripts.compare_results ... --context oracle        # the mixture with a perfect gate

# 4. held-out curves per session, per arm
python -m scripts.plot_results --set augmentation.copies=1 labels.tau_att=3 [mixture.expert_scope='"shared"']
```

**Cost.** The frame pass of the 52 originals is about 4–5 hours; each augmented copy is another
full frame pass (`copies=1` doubles it). Training both arms takes minutes.

## 4. How to read it

`compare_results` prints, for each track (expert entries, self entries): both arms' summary
metrics with the constant-3 and training-mean baselines, the AUC per held-out group, and AUC and
Spearman per activity.

- **Lead with AUC, Spearman and balanced accuracy**; MAE and accuracy reward predicting the middle
  of the scale (`05_per_labeler.md` §3).
- **The mixture helps** if it beats the single decoder on those metrics in both tracks, by more
  than the spread across the three folds, and in most activities. A gain under oracle context but
  not with the gate's prediction means the gate is the bottleneck; no gain even under oracle means
  activity-specific experts do not help these features.
- **Per activity** shows where: an activity whose cues differ (a lecture against group work, device
  use that means opposite things in different activities) is where a mixture should help most.
- **Expert track per labeler**: one expert made most of the entries in the six-session run
  (`05_per_labeler.md` §2); check the per-labeler breakdown before trusting the pooled numbers.

Each activity expert sees two groups' sessions per fold (up to three cameras each) against all six
activities for the single decoder: the mixture trades data per model for specialisation. That trade
is the question.

## 5. A preview on six sessions

On the six-session caches (8 s windows, no augmentation; `tmp/e2e6/run_per_labeler.py`, results
`2d4ef2c0164c` and `feda61a37624`), the arms are within ±0.03 AUC of each other in every fold,
and both are at chance. This says nothing about the mixture: with one session per activity, each
held-out activity has no expert (`04_six_session_run.md` §4). It only shows the comparison runs.

| Expert entries, classified context | AUC | Spearman | Balanced acc. |
|---|---|---|---|
| MoE | 0.507 | 0.006 | 0.491 |
| Single decoder | 0.513 | 0.025 | 0.488 |
| Constant 3 | 0.500 | – | 0.500 |

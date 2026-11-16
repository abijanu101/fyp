# The full pipeline experiment: mixture of experts against a single decoder

*Done 2026-11-16 (§6): the two arms tie, and both are barely above chance. Extensions: augmentation
dose (§7, copies 0 and 1 done, 4 running) and confusion matrices (§8).*

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
python -m scripts.train_models --set augmentation.copies=1 labels.tau_att=3 mixture.expert_scope=shared

# 3. compare: --latest picks the newest MoE (per_experiment) and single (shared) results
python -m scripts.compare_results --latest
python -m scripts.compare_results --latest --context oracle   # the mixture with a perfect gate

# 4. held-out curves per session, per arm
python -m scripts.plot_results --set augmentation.copies=1 labels.tau_att=3 [mixture.expert_scope=shared]
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

## 6. Result (full dataset)

**Setup.** All 52 session-cameras plus one augmented copy each (104 frame passes, features
`outputs/cache/features/2685b0745fde/`), with D3's detector threshold (0.1098) and T2's tracker
(BoT-SORT with ReID and CMC), both applied through `configs/derived.py`. 23,831 rating entries
(session-start entries excluded) give 45,166 training rows across the six labelers' models. In each
leave-one-group-out fold, 11,685 to 17,724 rows are held out. Results: MoE `38c985c0f859`, single
decoder `c5341cb6385e`.

**Expert entries, classified context** (the gate's predicted activity), mean over the three folds:

| | AUC | Spearman | Balanced acc. | Macro F1 | MAE |
|---|---|---|---|---|---|
| MoE | 0.526 | 0.055 | 0.500 | 0.135 | 0.677 |
| Single decoder | 0.527 | 0.072 | 0.499 | 0.134 | 0.674 |
| Constant 3 | 0.500 | – | 0.500 | 0.119 | 0.619 |
| Training mean | 0.501 | 0.012 | 0.500 | 0.119 | 0.656 |

AUC per held-out group, MoE minus single: +0.003, −0.005, −0.000. Per activity, the two arms are
within 0.01 AUC everywhere. The best activity is 05, at 0.56 for both arms.

**Self entries, classified context:** MoE 0.492 AUC / −0.026 Spearman, single 0.508 / 0.007, with
constant 3 at 0.500. Per fold the MoE is lower by up to 0.036, and per activity the two arms swing
both ways by up to 0.08 on small numbers of entries: noise around chance.

**Oracle context** (true activity): the MoE's expert-track AUC is 0.527, the same as with the gate.
The gate picks the right activity 22.8% of the time (chance is 16.7%), so the mixture is effectively
always mixing. But even a perfect gate does not help.

**Reading** (by §4's criteria):

1. **The mixture does not earn its cost.** It does not help with the predicted activity or the true
   one, so the gate is not the bottleneck. Activity-specific experts add nothing for these features.
   The single decoder is the simpler default.
2. **The pipeline as a whole is barely above chance.** It ranks expert entries slightly better than
   chance (AUC 0.53, Spearman 0.06–0.07), but loses to always predicting 3 on MAE. Macro F1 equals
   constant 3's to within 0.016, and balanced accuracy is 0.50. The rounded predictions are nearly
   all 3s, so the binary view at `tau_att` 3 calls almost every entry engaged. Self-reports are not
   predicted at all.
3. **The bottleneck is upstream of the decoder.** The window features (pose, gaze, flow, behaviour
   classes, scene) carry very little of what the labelers rated. Changing how they are decoded
   (mixture, output head) cannot recover signal they do not have. The next work is on the features
   and on label noise (`docs/misc/targets.md`), not on the decoder.

## 7. Extension: how much augmentation

Does training on more augmented copies help? Same pipeline, single decoder, `augmentation.copies`
0, 1 and 4. The originals are always the evaluation data; copies only add training rows (each
copy is a full frame pass of degraded frames, `docs/guide/08_scripts_caching_tuning.md` §8.4).

| Single decoder, expert entries, classified | Results | Training rows per fold | AUC | Spearman | Macro F1 | MAE |
|---|---|---|---|---|---|---|
| copies = 0 | `f6dcb84aaa6b` | 27–33k | 0.525 | 0.065 | 0.133 | 0.675 |
| copies = 1 | `c5341cb6385e` | 54–66k | 0.527 | 0.072 | 0.134 | 0.674 |
| copies = 4 | `53caa998218a` | ~135–165k | *running* | | | |
| Constant 3 | | | 0.500 | – | 0.119 | 0.619 |

AUC per held-out group, copies 1 minus copies 0: +0.007, +0.002, −0.001. Per activity the two are
within 0.01. Self entries: 0.508 against 0.510. **One copy does nothing measurable at full scale**:
the +0.02–0.03 gain in the six-session smoke test (`05_per_labeler.md` §4) does not hold up. That fits
§6's reading: the model is not overfitting (which augmentation fixes), the features are not
informative (which it cannot fix). Copies 4 tests that directly: if five times the training data
still lands at 0.53, data quantity is ruled out too.

```
python -m scripts.parallel_runs extract --set augmentation.copies=4 --workers 3      # copies 2-4, ~8.5 h
python -m scripts.train_models --set augmentation.copies=4 labels.tau_att=3 mixture.expert_scope=shared
python -m scripts.compare_results 53caa998218a c5341cb6385e --names copies4 copies1 --plot
python -m scripts.compare_results 53caa998218a f6dcb84aaa6b --names copies4 copies0 --plot
```

## 8. Confusion matrices

`compare_results` prints them for each run (summed over the three held-out folds, true rating
in rows against the prediction rounded to a whole rating in columns) and with `--plot` saves
heatmaps: `outputs/plots/compare_38c985c0f859_c5341cb6385e.png` (MoE against single) and
`outputs/plots/compare_c5341cb6385e_f6dcb84aaa6b.png` (copies 1 against 0).

**Single decoder, copies 1, expert entries** (count / share of the row):

| true \ predicted | 1 | 2 | 3 | 4 | 5 | entries | recall |
|---|---|---|---|---|---|---|---|
| **1** | 0 | 0 | 3 / 1.00 | 0 | 0 | 3 | 0.00 |
| **2** | 0 | 69 / 0.01 | 8614 / 0.97 | 240 / 0.03 | 0 | 8923 | 0.01 |
| **3** | 0 | 80 / 0.01 | 8810 / 0.96 | 253 / 0.03 | 0 | 9143 | 0.96 |
| **4** | 0 | 31 / 0.01 | 2237 / 0.95 | 96 / 0.04 | 0 | 2364 | 0.04 |
| **5** | 0 | 4 / 0.00 | 931 / 0.98 | 18 / 0.02 | 0 | 953 | 0.00 |
| precision | – | 0.38 | 0.43 | 0.16 | – | | |
| predicted | 0 | 184 | 20595 | 607 | 0 | 21,386 | |

At `tau_att` 3: of the 8,926 not-engaged entries (ratings 1–2), 69 are predicted not engaged; of
the 12,460 engaged, 12,345 are predicted engaged. The model calls 99% of entries engaged.

What the matrices show:

1. **The model is constant 3 with a few exceptions.** 96% of expert entries (20,595 of 21,386)
   are predicted 3; it never predicts 1 or 5. The other arms look the same: MoE 20,482, copies 0
   20,588. Self entries: 2,431 of 2,445 predicted 3 (99.4%). The summary metrics' small gains over
   constant 3 come from those 4% of exceptions, and even there the off-3 predictions are barely
   better than chance: of the 607 predicted 4s, 16% are 4s, about the 11% base rate of 4s.
2. **The expert ratings are mostly 2s and 3s, in equal measure.** 42% are 2 (8,923) and 43% are 3
   (9,143); 4s are 11%, 5s 4.5%, and 1s almost never occur (3 entries). So the task is first of
   all telling a 2 from a 3, and on that the model is at chance (row 2 and row 3 are predicted the
   same way, 97% and 96% as 3).
3. **With `tau_att` 3 the binary view is "2 against 3 or more"**, 42% against 58%; balanced
   accuracy 0.50 is a model that calls almost everyone engaged.
4. **Self entries are spread wider** (37% 2, 41% 3, 14% 4, 7% 5, 1% 1): students rate themselves
   higher than the experts do. The model predicts 3 for all of them.

The target for the next feature work (§6, `docs/progress/`) is concrete: move entries off 3, in
the right direction, more often than 4% of the time.

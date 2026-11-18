# Tests

The unit tests (`tests/`), documented as test cases in the report's template (Table 7.1): one
test case per test class, one step per test method.

- **Last run:** 2026-11-18, commit `e743616`: **121 of 121 tests passed**.
- **Run them:** `python -m unittest discover -s tests -t .` from the repository root.
- **The cases:** [test_cases.md](test_cases.md); the same in LaTeX for the report:
  [test_cases.tex](test_cases.tex) (`\input{}` it; needs the `array` package).
- **Regenerate** after changing tests: `python -m scripts.reporting.test_cases`. The descriptions
  come from `specs/<test file>.json`; status, dates, versions and the environment are measured.

| ID | Component | Test class | Steps | Status |
|---|---|---|---|---|
| TC-01 | Late aggregation settings | `tests/test_aggregation_modes.py` `TestLateSettings` | 3 | Passed |
| TC-02 | Late inference with no visible student | `tests/test_aggregation_modes.py` `TestNobodyVisible` | 1 | Passed |
| TC-03 | Late row construction and window scores | `tests/test_aggregation_modes.py` `TestRows` | 3 | Passed |
| TC-04 | Early aggregation equivalence | `tests/test_aggregation_modes.py` `TestEarlyEqualsDirectMixture` | 1 | Passed |
| TC-05 | Per-student labels for late aggregation | `tests/test_aggregation_modes.py` `TestStudentLabels` | 1 | Passed |
| TC-06 | Feature pass per-track rows | `tests/test_aggregation_modes.py` `TestFeaturePassRows` | 1 | Passed |
| TC-07 | Session-level data augmentation | `tests/test_augment.py` `TestAugmentation` | 8 | Passed |
| TC-08 | Colour augmentation (lookup-table implementation) | `tests/test_augment.py` `TestColourLookupTables` | 1 | Passed |
| TC-09 | Derived configuration values | `tests/test_derive.py` `TestDerive` | 3 | Passed |
| TC-10 | Head-direction geometry from COCO keypoints | `tests/test_head_pose.py` `TestHeadGeometry` | 4 | Passed |
| TC-11 | Pose-to-track matching | `tests/test_head_pose.py` `TestMatching` | 1 | Passed |
| TC-12 | Head-pose feature extractor (BFMU d) | `tests/test_head_pose.py` `TestUnit` | 3 | Passed |
| TC-13 | Inter-labeler agreement (Krippendorff's alpha) | `tests/test_label_agreement.py` `TestAlpha` | 2 | Passed |
| TC-14 | Human ceiling: predicting one labeler from the others | `tests/test_label_agreement.py` `TestCeiling` | 6 | Passed |
| TC-15 | Agreement on coarse units | `tests/test_label_agreement.py` `TestCoarse` | 1 | Passed |
| TC-16 | Rating curve interpolation | `tests/test_label_targets.py` `TestInterpolation` | 4 | Passed |
| TC-17 | Rating conversion and rating metrics | `tests/test_label_targets.py` `TestRatings` | 8 | Passed |
| TC-18 | Soft-label training of engagement experts | `tests/test_label_targets.py` `TestSoftLabels` | 3 | Passed |
| TC-19 | Per-labeler models and rating output | `tests/test_label_targets.py` `TestPerLabeler` | 4 | Passed |
| TC-20 | Joining rating entries to track features | `tests/test_label_targets.py` `TestJoinEntries` | 1 | Passed |
| TC-21 | Attention label parsing | `tests/test_labels.py` `TestAttentionTrack` | 6 | Passed |
| TC-22 | Window engagement labels | `tests/test_labels.py` `TestWindowLabeler` | 3 | Passed |
| TC-23 | Gated mixture of engagement experts | `tests/test_mixture.py` `TestMixture` | 6 | Passed |
| TC-24 | Leave-one-group-out split | `tests/test_mixture.py` `TestLogo` | 1 | Passed |
| TC-25 | Per-labeler ensemble on the window grid | `tests/test_per_labeler.py` `TestLabelerEnsemble` | 4 | Passed |
| TC-26 | SCB student-behaviour feature extractor (BFMU c) | `tests/test_scb.py` `TestScbExtractor` | 9 | Passed |
| TC-27 | SCB extractor wiring in the pipeline and frame-pass cache key | `tests/test_scb.py` `TestScbInPipeline` | 1 | Passed |
| TC-28 | Safe YOLOv7 checkpoint loader | `tests/test_scb.py` `TestSafeLoader` | 1 | Passed |
| TC-29 | Within/between-cluster spread decomposition | `tests/test_spread.py` `TestSpread` | 4 | Passed |
| TC-30 | Windowing (aggregation) | `tests/test_stages.py` `TestWindows` | 2 | Passed |
| TC-31 | Tracklet building and visibility filtering | `tests/test_stages.py` `TestTracklets` | 2 | Passed |
| TC-32 | SID-TID association | `tests/test_stages.py` `TestAssociation` | 2 | Passed |
| TC-33 | BFMU (a): Kalman/geometry motion features | `tests/test_stages.py` `TestKalmanGeometry` | 2 | Passed |
| TC-34 | Window aggregation (class-level vector Vec) | `tests/test_stages.py` `TestAggregation` | 1 | Passed |
| TC-35 | Configuration hashing, cache keys, pipeline wiring, track store and output graph | `tests/test_stages.py` `TestConfigAndWiring` | 6 | Passed |
| TC-36 | Inter-labeler agreement (Krippendorff's alpha, ordinal) | `tests/test_tuning.py` `TestAlpha` | 2 | Passed |
| TC-37 | Tracking quality metrics | `tests/test_tuning.py` `TestTrackingMetrics` | 3 | Passed |
| TC-38 | Hyperparameter search (random, beam bisection, nested leave-one-group-out) | `tests/test_tuning.py` `TestSearch` | 5 | Passed |
| TC-39 | Shared-frame multi-tracker runs and search prefetch | `tests/test_tuning.py` `TestSharedFrames` | 2 | Passed |

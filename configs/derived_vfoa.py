"""Constants for experiment 09 (docs/experiments/09_vfoa.md), applied to docs/experiments/09_vfoa.json on
every load (src/core/derive.py), like configs/derived.py for the base config.

The person detector is the head-pose model (models/yolo11s-pose.pt at 1280), so D2's threshold, found
for YOLO11n, does not apply. D3 is the same augmented scan (scripts/tuning/scan_detector.py --augment)
for the pose model, with the stricter coverage criterion of experiment 09: no student may lose more
than 1 point of recall against the confidence floor (--max-drop 0.01).
"""

from src.core.derive import Sourced, result

D3 = "outputs/tuning/detector_search_augmented_pose.json"
detector_conf = result(D3, "optimal_threshold")

# T2's tracker (BoT-SORT + ReID + CMC) and its options. Its det_thresh is stored but never read by
# BoxMOT's trackers (docs/experiments/02_detection.md §3); it is set to D3's value so the config says one
# threshold. Interim: T2's track_low_thresh (0.1745) drops every box below it, undoing D3's 0.0535;
# T3 (09 §1, tune_tracker --tag pose) re-tunes the options and replaces T2 here once it has run.
T2 = "outputs/tuning/tracker_botsort_full_reid_cmc_aug1-4f270e71_bisect.json"
tracker_name = result(T2, "tracker")
_options = dict(result(T2, "config_patch.tracker.options").value)
_options["det_thresh"] = detector_conf.value
tracker_options = Sourced(_options, f"{T2} (det_thresh from {D3})")

OVERRIDES = {
    "detector.conf": detector_conf,
    "tracker.name": tracker_name,
    "tracker.options": tracker_options,
}
assert all(isinstance(v, Sourced) for v in OVERRIDES.values())

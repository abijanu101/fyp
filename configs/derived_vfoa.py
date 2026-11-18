"""Constants for experiment 09 (docs/experiments/09_vfoa.md), applied to docs/experiments/09_vfoa.json on
every load (src/core/derive.py), like configs/derived.py for the base config.

The person detector is the head-pose model (models/yolo11s-pose.pt at 1280), so D3's threshold, found
for YOLO11n, does not apply. D4 is the same augmented scan (scripts/tuning/scan_detector.py --augment)
for the pose model, with the stricter coverage criterion of experiment 09: no student may lose more
than 1 point of recall against the confidence floor (--max-drop 0.01).
"""

from src.core.derive import Sourced, result

D4 = "outputs/tuning/detector_search_augmented_pose.json"
detector_conf = result(D4, "optimal_threshold")

# T2's tracker (BoT-SORT + ReID + CMC) and its options. Its det_thresh is stored but never read by
# BoxMOT's trackers (docs/experiments/02_detection.md §3); it is set to D4's value so the config says one
# threshold. Whether T2's options still track as well with the pose detector is checked in 09 §1.
T2 = "outputs/tuning/tracker_botsort_full_reid_cmc_aug1-4f270e71_bisect.json"
tracker_name = result(T2, "tracker")
_options = dict(result(T2, "config_patch.tracker.options").value)
_options["det_thresh"] = detector_conf.value
tracker_options = Sourced(_options, f"{T2} (det_thresh from {D4})")

OVERRIDES = {
    "detector.conf": detector_conf,
    "tracker.name": tracker_name,
    "tracker.options": tracker_options,
}
assert all(isinstance(v, Sourced) for v in OVERRIDES.values())

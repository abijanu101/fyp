"""Constants that come from experiments, applied to the config on every load (src/core/derive.py).

Each config names the results files its values come from, under `derived`:

  configs/deliverable3.json (YOLO11n):
    "detector_scan": D2, outputs/tuning/detector_search_augmented.json
                     (docs/experiments/02_detection.md: the highest threshold keeping every student)
    "tracker_scan":  T2, outputs/tuning/tracker_botsort_full_reid_cmc_aug1-4f270e71_bisect.json
                     (03_tracking.md §6-7: BoT-SORT + ReID + CMC, IDF1 0.930)
  docs/experiments/09_vfoa.json (the pose model):
    "detector_scan": D3, outputs/tuning/detector_search_augmented_pose.json
    "tracker_scan":  T3, outputs/tuning/tracker_botsort_full_reid_cmc_aug1-4f270e71_pose_bisect.json

To change a setting that an experiment decided, point the config at the new results file, not at a
typed-in number. Values set here override the JSON; `--set` still overrides them. Only compute values
here: every script that loads a config imports this module.
"""

from src.core.derive import Sourced, result


def overrides(params: dict) -> dict:
    """detector.conf from the detector scan; the tracker and its options from the tracker scan."""
    detector_scan, tracker_scan = params["detector_scan"], params["tracker_scan"]
    detector_conf = result(detector_scan, "optimal_threshold")
    # the tracker scan was run with the detector scan's threshold fixed (and passed as det_thresh): if
    # either file changes alone, the pair no longer belongs together
    if abs(result(tracker_scan, "config_patch.detector.conf").value - detector_conf.value) > 1e-9:
        raise ValueError(f"{tracker_scan} was tuned at a different detector threshold than {detector_scan}: "
                         f"rerun it with --det-conf-from {detector_scan}")
    out = {
        "detector.conf": detector_conf,
        "tracker.name": result(tracker_scan, "tracker"),
        "tracker.options": result(tracker_scan, "config_patch.tracker.options"),   # det_thresh = detector.conf
    }
    assert all(isinstance(v, Sourced) for v in out.values())
    return out

"""Constants that come from experiments, applied to the config on every load (src/core/derive.py).

Edit this file, not the JSON, when an experiment's outcome should change a setting. Values listed
here override configs/deliverable3.json; `--set` still overrides them. Only compute values here.
"""

from src.core.derive import Sourced, result

# D3: the augmented detector scan (docs/experiments/02_boxmot.md §3). The highest confidence
# threshold that keeps every student within 10 recall points of the floor on degraded footage.
D3 = "outputs/tuning/detector_search_augmented.json"
detector_conf = result(D3, "optimal_threshold")

# T2: the augmented tracker scans (02_boxmot.md §6-7). BoT-SORT with ReID and CMC won (IDF1 0.930);
# its best trial's tracker options, found with D3's threshold fixed (and passed as det_thresh).
T2 = "outputs/tuning/tracker_botsort_full_reid_cmc_aug1-4f270e71_bisect.json"
tracker_name = result(T2, "tracker")
tracker_options = result(T2, "config_patch.tracker.options")

# T2 was run with D3's threshold: if either file changes alone, the pair no longer belongs together.
if abs(result(T2, "config_patch.detector.conf").value - detector_conf.value) > 1e-9:
    raise ValueError(f"{T2} was tuned at a different detector threshold than {D3}: rerun T2 with "
                     f"--det-conf-from {D3}")

OVERRIDES = {
    "detector.conf": detector_conf,
    "tracker.name": tracker_name,
    "tracker.options": tracker_options,     # includes det_thresh = detector.conf
}
assert all(isinstance(v, Sourced) for v in OVERRIDES.values())

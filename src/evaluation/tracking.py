"""Tracking quality against the static manual boxes (design §11): IDF1-style score, ID switches,
fragmentation, coverage.

Ground truth assumption: every annotated subject is present in every frame of the evaluated span
(abi_labels has one box per subject per camera, not per frame). A subject who briefly leaves
lowers coverage equally for every tracker configuration, so comparisons stay fair even though
absolute numbers are a lower bound. Track boxes on people outside the annotation (e.g. the
teacher) count as false positives in IDF1, again equally across configurations.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core.types import Tracklet
from ..tracking.association import frame_matches


def tracking_metrics(tracklets: list[Tracklet], subjects: dict[str, np.ndarray], n_frames: int,
                     min_iou: float) -> dict[str, float]:
    """tracklets: raw (unfiltered) tracker output for the evaluated frames."""
    n_subj = len(subjects)
    if n_subj == 0 or n_frames == 0:
        return {}
    seq: dict[str, list[int]] = defaultdict(list)       # per subject: matched TID per frame, in time order
    counts: Counter = Counter()                         # (tid, sid) -> frames matched
    n_track_obs = 0
    for _, n_boxes, pairs in frame_matches(tracklets, subjects, min_iou):
        n_track_obs += n_boxes
        for tid, sid in pairs:
            seq[sid].append(tid)
            counts[tid, sid] += 1
    # IDF1 uses the one-to-one TID<->SID mapping that maximises matched frames, so a fragmented
    # subject only gets credit for one of its tracks.
    tids = sorted({t for t, _ in counts})
    sids = sorted(subjects)
    m = np.array([[counts[t, s] for s in sids] for t in tids]) if tids else np.zeros((0, len(sids)))
    rows, cols = linear_sum_assignment(-m) if tids else ([], [])
    idtp = int(m[rows, cols].sum()) if tids else 0
    idfp = n_track_obs - idtp
    idfn = n_subj * n_frames - idtp
    switches = sum(int(np.count_nonzero(np.diff(s))) for s in seq.values())
    return {
        "idf1": 2 * idtp / max(2 * idtp + idfp + idfn, 1),
        "coverage": sum(len(s) for s in seq.values()) / (n_subj * n_frames),
        "id_switches_per_subject": switches / n_subj,
        "tids_per_subject": float(np.mean([len(set(seq[s])) if seq[s] else 0 for s in subjects])),
        "subjects_found": sum(1 for s in subjects if seq[s]) / n_subj,
        "n_tracks": float(len({tr.tid for tr in tracklets})),
    }

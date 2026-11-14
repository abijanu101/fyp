"""SID-TID association (design §4). Training/evaluation only, never at inference.

DIPSER's manual annotations (abi_labels) are one static box per subject per camera, so the
per-frame Hungarian assignment matches every frame's track boxes against the same subject
boxes; a majority vote over frames then gives each track its SID.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core.types import Tracklet


def _area(r: np.ndarray) -> np.ndarray:
    return (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    inter = _intersection(a, b)
    union = _area(a)[:, None] + _area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def containment_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise share of each box in `a` that lies inside each box in `b` (intersection / area(a))."""
    return _intersection(a, b) / np.maximum(_area(a), 1e-9)[:, None]


def frame_matches(tracklets: list[Tracklet], subjects: dict[str, np.ndarray],
                  min_iou: float = 0.3) -> Iterator[tuple[float, int, list[tuple[int, str]]]]:
    """Per frame (time order): yield (t, n_track_boxes, [(tid, sid), ...]) from Hungarian IoU
    assignment of that frame's track boxes to the subject boxes, keeping pairs with IoU >= min_iou.

    Tie-breaking (design §12, named rather than claimed away): linear_sum_assignment is
    deterministic for a given matrix but equal-IoU ties resolve by index order."""
    sids = sorted(subjects)
    sub_boxes = np.stack([subjects[s] for s in sids]) if sids else np.zeros((0, 4))
    frames: dict[float, list[tuple[int, np.ndarray]]] = {}
    for tr in tracklets:
        for t, box in zip(tr.t, tr.xyxy):
            frames.setdefault(float(t), []).append((tr.tid, box))
    for t in sorted(frames):
        tids, boxes = zip(*frames[t])
        pairs = []
        if sids:
            iou = iou_matrix(np.stack(boxes), sub_boxes)
            rows, cols = linear_sum_assignment(-iou)
            pairs = [(tids[r], sids[c]) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
        yield t, len(tids), pairs


def associate_sids(tracklets: list[Tracklet], subjects: dict[str, np.ndarray],
                   min_iou: float = 0.3, min_match_share: float = 0.5) -> dict[int, str]:
    """Return {tid: sid} by per-frame assignment plus majority vote; vote ties go to the
    lexicographically first SID.

    Two guards keep unannotated people (TAs, the teacher walking around) unassociated:
    a frame only votes if its IoU >= min_iou, and a track only gets the winning SID if that SID
    matched at least `min_match_share` of ALL the track's frames (a TA passing a seated student
    overlaps that student's box for a few frames, not most of the track)."""
    n_frames = {tr.tid: 0 for tr in tracklets}
    for tr in tracklets:
        n_frames[tr.tid] += len(tr.t)
    votes: dict[int, Counter] = {}
    for _, _, pairs in frame_matches(tracklets, subjects, min_iou):
        for tid, sid in pairs:
            votes.setdefault(tid, Counter())[sid] += 1
    out = {}
    for tid, c in votes.items():
        sid, n = max(sorted(c.items()), key=lambda kv: kv[1])
        if n >= min_match_share * n_frames[tid]:
            out[tid] = sid
    return out

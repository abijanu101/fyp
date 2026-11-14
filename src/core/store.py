"""Persist a SessionTrack (the expensive frame pass) as one compressed .npz per session.

Layout: per-observation arrays concatenated over tracklets (tid, t, xyxy, conf, velocity and one
array per signal key, NaN where a tracklet lacks that signal) plus per-frame arrays (frame_t,
scene matrix) and a JSON header for everything else. Loading rebuilds identical Tracklets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import SessionKey, SessionTrack, Tracklet


def save_track(st: SessionTrack, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trs = st.tracklets
    sig_keys = sorted({k for tr in trs for k in tr.signals})
    scene_keys = sorted({k for r in st.scene_rows for k in r})
    cat = lambda parts, shape: np.concatenate(parts) if parts else np.zeros(shape)
    arrays = {
        "tid": cat([np.full(len(tr.t), tr.tid) for tr in trs], (0,)),
        "t": cat([tr.t for tr in trs], (0,)),
        "xyxy": cat([tr.xyxy for tr in trs], (0, 4)),
        "conf": cat([tr.conf for tr in trs], (0,)),
        "velocity": cat([tr.velocity for tr in trs], (0, 2)),
        "frame_t": np.asarray(st.frame_t, float),
        "scene": np.array([[r.get(k, np.nan) for k in scene_keys] for r in st.scene_rows], float)
                 .reshape(len(st.scene_rows), len(scene_keys)),
    }
    for k in sig_keys:
        arrays[f"sig:{k}"] = cat([tr.signals.get(k, np.full(len(tr.t), np.nan)) for tr in trs], (0,))
    header = {"key": [st.key.group, st.key.experiment, st.key.camera] if st.key else None,
              "frame_wh": list(st.frame_wh), "timing": st.timing, "stage_seconds": st.stage_seconds,
              "signal_keys": sig_keys, "scene_keys": scene_keys}
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, header=np.array(json.dumps(header)), **arrays)
    tmp.replace(path)                     # atomic: a crash never leaves a half-written cache entry


def load_track(path: str | Path) -> SessionTrack:
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}      # read each array once (every z[k] access re-decompresses it)
    h = json.loads(str(a["header"]))
    tid = a["tid"]
    tracklets = []
    for u in np.unique(tid):              # observations of one tid are contiguous and time-ordered
        m = tid == u
        tracklets.append(Tracklet(int(u), a["t"][m], a["xyxy"][m], a["conf"][m], a["velocity"][m],
                                  {k: a[f"sig:{k}"][m] for k in h["signal_keys"]}))
    scene_rows = [dict(zip(h["scene_keys"], map(float, row))) for row in a["scene"]]
    return SessionTrack(SessionKey(*h["key"]) if h["key"] else None, a["frame_t"], tuple(h["frame_wh"]),
                        tracklets, scene_rows, h["timing"], h["stage_seconds"])

"""Video-file frame source (registry name "video"), for datasets shipped as mp4 (OUC-CGE).

Timestamps are seconds from the start of the file (frame index / file fps): a video has no wall
clock, and every downstream stage only uses differences of `Frame.t`.

`target_fps` subsamples to about that rate (step = round(file fps / target_fps)), so a 30-fps clip
runs at the ~5 fps the DIPSER-tuned tracker and BFMU settings assume (DIPSER context cameras are
~5 fps; `tracker.max_lost_frames` and the flow statistics are in frames). Clips at 27–28 fps get a
step of 5–6 as well. `stride` is applied on top, `max_frames` caps the count (smoke tests). Skipped
frames are `grab()`bed, not decoded, which is most of the saving on 1080p footage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..core.types import Frame
from .source import FRAME_SOURCES, FrameSource


@FRAME_SOURCES.register("video")
class VideoFrameSource(FrameSource):
    def __init__(self, path: str | Path, stride: int = 1, max_frames: int | None = None,
                 target_fps: float | None = None):
        self.path = Path(path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise OSError(f"cannot open video {self.path}")
        self.file_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        step = max(1, int(round(self.file_fps / target_fps))) if target_fps else 1
        self.step = step * max(1, int(stride))
        self._idx = np.arange(0, max(n, 0), self.step)[:max_frames]
        self.max_frames = max_frames

    @property
    def timestamps(self) -> np.ndarray:
        """Expected timestamps from the header's frame count (the decoded frames are authoritative)."""
        return self._idx / self.file_fps

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        try:
            i, out = 0, 0
            while self.max_frames is None or out < self.max_frames:
                if i % self.step == 0:
                    ok, image = cap.read()
                    if not ok:
                        break
                    yield Frame(out, i / self.file_fps, image)
                    out += 1
                elif not cap.grab():
                    break
                i += 1
        finally:
            cap.release()

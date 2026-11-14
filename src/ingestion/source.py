"""Frame sources (design §3 stage 1). Frames are streamed from the ZIP, never extracted to disk."""

from __future__ import annotations

import zipfile
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import Iterator

import cv2
import numpy as np

from ..core.clock import parse_clock
from ..core.registry import Registry
from ..core.types import Frame

FRAME_SOURCES = Registry("frame source")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


class FrameSource(ABC):
    @property
    @abstractmethod
    def timestamps(self) -> np.ndarray:
        """Sorted frame timestamps (seconds since midnight)."""

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]: ...


@FRAME_SOURCES.register("zip")
class ZipFrameSource(FrameSource):
    """Reads timestamp-named images from `images/` inside a DIPSER camera ZIP, in time order.

    `stride` keeps every n-th frame and `max_frames` caps the count (for smoke tests).
    """

    def __init__(self, path: str | Path, stride: int = 1, max_frames: int | None = None):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as zf:
            # Zero-byte members (DIPSER has hundreds in some sessions) are not frames: dropping them
            # here keeps timestamps, measured fps and the gap audit honest.
            members = [i.filename for i in zf.infolist() if i.file_size > 0 and _is_frame(i.filename)]
        members.sort(key=lambda m: parse_clock(PurePosixPath(m).stem))
        members = members[::stride][:max_frames]
        self._members = members
        self._t = np.array([parse_clock(PurePosixPath(m).stem) for m in members])

    @property
    def timestamps(self) -> np.ndarray:
        return self._t

    def __iter__(self) -> Iterator[Frame]:
        with zipfile.ZipFile(self.path) as zf:
            for i in range(len(self._members)):
                frame = self._decode(zf, i)
                if frame is not None:     # unreadable frame; shows up as a gap in the manifest audit
                    yield frame

    def read(self, i: int) -> Frame | None:
        """Random access to frame i (None if unreadable)."""
        with zipfile.ZipFile(self.path) as zf:
            return self._decode(zf, i)

    def _decode(self, zf: zipfile.ZipFile, i: int) -> Frame | None:
        """None for an unreadable frame. DIPSER contains empty (0-byte) PNGs, on which cv2.imdecode
        raises instead of returning None."""
        data = zf.read(self._members[i])
        if not data:
            return None
        try:
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            return None
        return None if image is None else Frame(i, float(self._t[i]), image)


def _is_frame(member: str) -> bool:
    p = PurePosixPath(member)
    return (len(p.parts) == 2 and p.parts[0] == "images"
            and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."))

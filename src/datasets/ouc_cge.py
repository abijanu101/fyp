"""OUC-CGE as a second dataset (Lu et al., *A Video Dataset for Classroom Group Engagement
Recognition*, Scientific Data 2025): ~10-s mp4 clips of group work, one group-level label each
(low 0 / mid 1 / high 2), and an official per-clip train/val/test split.

Split files:  <split_dir>/{train,val,test}.csv, rows `videos/<level>/viewN.mp4 <label>`
              (the authors' GitHub repo, OUC-CGE-main). test.csv shipped with one placeholder row,
              `videos/anonymity/a.mp4`, which matches no clip; rows that are not clips are skipped.
Videos:       <root>/<level folder>/... nested as the archives unpack, e.g.
              `high/high-2636(1-1000)/view12.mp4`, `Low/low-3028(1-1000)/view7.mp4`; found by a
              recursive scan (level folders matched case-insensitively; `* sample` folders ignored).

Clip id = `<level>_<stem>` (`high_view12`): the same viewN number exists in every level folder.
No session or camera id is published, so the official split cannot be grouped by recording
(clips of one session can sit on both sides: likely optimistic scores; see the benchmark doc).

As SessionKey: group "ouc_cge", experiment = level, camera = stem. Nothing here imports torch, so the
data scripts (audit, floor) run in any environment with numpy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from ..core.types import SessionKey

LEVELS = {"low": 0, "mid": 1, "high": 2}
LEVEL_NAMES = ["low", "mid", "high"]
SPLITS = ("train", "val", "test")
GROUP = "ouc_cge"
_ROW = re.compile(r"^videos/(\w+)/(view\d+)\.mp4\s+(\d+)\s*$", re.IGNORECASE)
_VIDEO = re.compile(r"^view\d+\.mp4$", re.IGNORECASE)


@dataclass(frozen=True)
class Clip:
    clip_id: str          # "<level>_<stem>"
    level: str            # low | mid | high
    stem: str             # viewN
    label: int            # 0 | 1 | 2
    split: str            # train | val | test

    @property
    def key(self) -> SessionKey:
        return SessionKey(GROUP, self.level, self.stem)


def read_splits(split_dir: str | Path, verbose: bool = True) -> list[Clip]:
    clips, skipped = [], []
    for split in SPLITS:
        text = Path(split_dir, f"{split}.csv").read_text(encoding="utf-8")
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            m = _ROW.match(line.strip())
            if not m or m.group(1).lower() not in LEVELS:
                skipped.append(f"{split}.csv:{n}: {line.strip()}")
                continue
            level, stem, label = m.group(1).lower(), m.group(2).lower(), int(m.group(3))
            if LEVELS[level] != label:
                raise ValueError(f"{split}.csv:{n}: folder '{level}' but label {label}")
            clips.append(Clip(f"{level}_{stem}", level, stem, label, split))
    ids = [c.clip_id for c in clips]
    if len(ids) != len(set(ids)):
        raise ValueError("a clip appears more than once in the split files")
    if verbose:
        for s in skipped:
            print(f"skipped split row (not a clip): {s}")
    return clips


def find_videos(root: str | Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for top in sorted(Path(root).iterdir()):
        level = top.name.lower()
        if not top.is_dir() or level not in LEVELS:
            continue
        for p in sorted(top.rglob("*.mp4")):
            if any("sample" in part.lower() for part in p.relative_to(top).parts[:-1]):
                continue
            if not _VIDEO.match(p.name):
                continue                       # e.g. a stray `7.mp4` in one low archive
            cid = f"{level}_{p.stem.lower()}"
            if cid in found:
                raise ValueError(f"two files for {cid}: {found[cid]} and {p}")
            found[cid] = p
    return found


class OucCgeIndex:
    """Clips of the official split whose video exists under `root`."""

    def __init__(self, root: str | Path, split_dir: str | Path, splits: list[str] | None = None):
        self.root, self.split_dir = Path(root), Path(split_dir)
        self.splits = list(splits or SPLITS)

    @classmethod
    def from_config(cls, cfg: dict) -> "OucCgeIndex":
        d = cfg["dataset"]
        return cls(d["root"], d["split_dir"], d.get("splits"))

    @cached_property
    def videos(self) -> dict[str, Path]:
        return find_videos(self.root)

    @cached_property
    def all_clips(self) -> list[Clip]:
        return read_splits(self.split_dir)

    def clips(self) -> list[Clip]:
        return [c for c in self.all_clips if c.split in self.splits and c.clip_id in self.videos]

    def missing(self) -> list[Clip]:
        return [c for c in self.all_clips if c.split in self.splits and c.clip_id not in self.videos]

    def video(self, clip: Clip) -> Path:
        return self.videos[clip.clip_id]

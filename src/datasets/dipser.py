"""Scoped view of the DIPSER dataset (design §2).

Raw data:     <root>/DIPSEER/group_XX/experiment_YY/context_cam_ZZ.zip
Labels:       <root>/DIPSEER/group_XX/experiment_YY/subjects/subject_NN.zip  (labels/*.json inside)
Annotations:  <root>/abi_labels/group_XX/experiment_YY.json  ({camera: {subject: {x, y, w, h}}})
Metadata:     <root>/abi_labels/experiments.tsv
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from ..core.types import SessionKey


@dataclass(frozen=True)
class ExperimentInfo:
    exp_id: str
    layout: str
    device_use: str
    activity: str


class DipserIndex:
    def __init__(self, root: str | Path, groups: list[str], experiments: list[str], cameras: list[str]):
        self.root = Path(root)
        self.raw = self.root / "DIPSEER"
        self.annotations_dir = self.root / "abi_labels"
        self.groups, self.experiments, self.cameras = groups, experiments, cameras

    @classmethod
    def from_config(cls, cfg: dict) -> "DipserIndex":
        d = cfg["dataset"]
        return cls(d["root"], d["groups"], d["experiments"], d["cameras"])

    def sessions(self) -> list[SessionKey]:
        """In-scope session-cameras whose ZIP exists on disk (at most 54 for Deliverable III)."""
        keys = [SessionKey(g, e, c)
                for g in self.groups for e in self.experiments for c in self.cameras]
        return [k for k in keys if self.frames_zip(k).is_file()]

    def frames_zip(self, key: SessionKey) -> Path:
        return self.raw / key.group / key.experiment / f"{key.camera}.zip"

    def subject_zips(self, key: SessionKey) -> dict[str, Path]:
        d = self.raw / key.group / key.experiment / "subjects"
        return {p.stem: p for p in sorted(d.glob("subject_*.zip"))}

    def annotations(self, key: SessionKey) -> dict[str, np.ndarray]:
        """Static manual boxes for this camera as {subject_id: xyxy}. Training/eval only."""
        path = self.annotations_dir / key.group / f"{key.experiment}.json"
        if not path.is_file():
            return {}
        cam = json.loads(path.read_text(encoding="utf-8")).get(key.camera, {})
        return {sid: np.array([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], float)
                for sid, b in cam.items()}

    @cached_property
    def experiment_info(self) -> dict[str, ExperimentInfo]:
        with open(self.annotations_dir / "experiments.tsv", encoding="utf-8") as f:
            rows = csv.DictReader(f, delimiter="\t")
            return {r["exp"]: ExperimentInfo(r["exp"], r["layout"], r["device_use"], r["activity"])
                    for r in rows}

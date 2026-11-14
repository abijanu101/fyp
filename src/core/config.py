"""Config loading and hashing (design §9: SHA-256 of canonical JSON, excluding the output dir)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

# None of these changes what a run produces: output_dir is where it goes; tuning and benchmark only
# drive their own scripts.
HASH_EXCLUDED_KEYS = ("output_dir", "tuning", "benchmark")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict:
    """Load a JSON config and apply dotted-key overrides, e.g. {"window.length_s": 10}."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for dotted, value in (overrides or {}).items():
        set_dotted(cfg, dotted, value)
    return cfg


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def config_hash(cfg: dict) -> str:
    """SHA-256 of canonical JSON, excluding output_dir and tuning. Cache keys for the two pipeline
    passes are hashes of config subsets (core.pipeline.tracking_hash / feature_hash)."""
    clean = copy.deepcopy(cfg)
    for k in HASH_EXCLUDED_KEYS:
        clean.pop(k, None)
    canonical = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

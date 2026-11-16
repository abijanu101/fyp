"""Constants derived from experiment results (config key `derived`, e.g. configs/derived.py).

The core config holds settings; some values are not settings but results of experiments (the
detector threshold from the augmented detector scan, the tracker parameters from the tracker scans).
A derived module computes them in plain Python, reading the results files, and lists them in a
dict `OVERRIDES = {dotted config key: value}`. `load_config` imports the module once per process,
writes the resolved values into the config before any `--set` override, and records where each
came from under `derived_from` (left out of the config hash: the hash sees the values, not their
source). A missing results file raises: the pipeline never falls back to a stale constant.

A derived module must only compute values: no writing files, no launching work, since every script
that loads the config imports it.
"""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple


class Sourced(NamedTuple):
    """A derived value and where it came from."""
    value: Any
    source: str


def result(path: str | Path, dotted: str) -> Sourced:
    """The value at `dotted` (e.g. "config_patch.tracker.options") in a JSON results file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"derived config: results file {path} does not exist (run the experiment that "
                                f"writes it, or override the key with --set)")
    node: Any = json.loads(path.read_text(encoding="utf-8"))
    parts, i = dotted.split("."), 0
    while i < len(parts):                      # longest key first: results use keys like "tracker.options"
        j = next((j for j in range(len(parts), i, -1)
                  if isinstance(node, dict) and ".".join(parts[i:j]) in node), None)
        if j is None:
            raise KeyError(f"derived config: {path} has no '{dotted}' ('{parts[i]}' missing)")
        node, i = node[".".join(parts[i:j])], j
    return Sourced(node, f"{path.as_posix()}#{dotted}")


@lru_cache(maxsize=None)
def _load(path: str) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(f"_derived_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"derived config module {path} not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not isinstance(getattr(module, "OVERRIDES", None), dict):
        raise TypeError(f"{path} must define a dict OVERRIDES")
    return module.OVERRIDES


def derived_values(path: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    """({dotted key: value}, {dotted key: source}) from a derived module."""
    values, sources = {}, {}
    for key, v in _load(str(Path(path))).items():
        if isinstance(v, Sourced):
            values[key], sources[key] = v.value, v.source
        else:
            values[key], sources[key] = v, str(Path(path).as_posix())
    return values, sources

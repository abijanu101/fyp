"""Per-stage wall-clock timer (design §11 compute target: <= 3 min per 45-min session)."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self.totals: dict[str, float] = defaultdict(float)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - start

    def report(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.totals.items()}

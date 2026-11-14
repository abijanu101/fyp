"""Leave-one-group-out split (design §7): all cameras of a held-out group stay in the test fold."""

from __future__ import annotations

from typing import Iterator

import numpy as np


def logo_splits(groups: np.ndarray) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (held_out_group, train_idx, test_idx) over the row-level group array."""
    groups = np.asarray(groups)
    for g in sorted(set(groups.tolist())):
        yield g, np.flatnonzero(groups != g), np.flatnonzero(groups == g)

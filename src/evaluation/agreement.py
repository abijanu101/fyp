"""Krippendorff's alpha, ordinal metric (design §11: inter-labeler agreement as the accuracy ceiling)."""

from __future__ import annotations

import numpy as np


def krippendorff_alpha_ordinal(units: list[list[float]]) -> float:
    """units: one list of labeler values per rated item (NaN / missing allowed). Items with fewer
    than two values are not pairable and are ignored. Returns NaN when undefined."""
    units = [[v for v in u if np.isfinite(v)] for u in units]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return np.nan
    values = np.unique(np.concatenate(units))
    pos = {v: i for i, v in enumerate(values)}
    k = len(values)
    o = np.zeros((k, k))                                 # coincidence matrix
    for u in units:
        idx = [pos[v] for v in u]
        m = len(idx)
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[idx[a], idx[b]] += 1 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    cum = np.cumsum(n_c)
    # ordinal squared distance: (sum_{g=c..k} n_g - (n_c + n_k)/2)^2
    lo, hi = np.minimum.outer(np.arange(k), np.arange(k)), np.maximum.outer(np.arange(k), np.arange(k))
    between = cum[hi] - cum[lo] + n_c[lo]
    delta = (between - (n_c[:, None] + n_c[None, :]) / 2) ** 2
    d_o = (o * delta).sum()
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n - 1)
    return float(1 - d_o / d_e) if d_e > 0 else np.nan

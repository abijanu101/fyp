"""Within/between-cluster spread decomposition (design §10.4).

    var_within[d]  = sum_c (n_c/N) * mean_{i in c} (x_i[d] - mu_c[d])^2
    var_between[d] = sum_c (n_c/N) * (mu_c[d] - mu_g[d])^2
    var_total      = var_within + var_between   (population variance of all N tracks)
"""

from __future__ import annotations

import numpy as np


def spread_decomposition(x: np.ndarray, labels: np.ndarray,
                         blocks: dict[str, list[int]] | None = None) -> dict[str, np.ndarray | dict]:
    """x: (N, D) per-track descriptors (already scaled); labels: (N,) cluster ids.

    Returns per-feature SD vectors `within_sd`, `between_sd`, `total_sd` and, if `blocks`
    maps block name -> feature columns, `between_share` per block (0 when total spread is 0).
    """
    x = np.asarray(x, float)
    labels = np.asarray(labels)
    n = len(x)
    mu_g = x.mean(axis=0)
    var_within = np.zeros(x.shape[1])
    var_between = np.zeros(x.shape[1])
    for c in np.unique(labels):
        members = x[labels == c]
        w = len(members) / n
        mu_c = members.mean(axis=0)
        var_within += w * ((members - mu_c) ** 2).mean(axis=0)
        var_between += w * (mu_c - mu_g) ** 2
    var_total = var_within + var_between
    out: dict = {"within_sd": np.sqrt(var_within), "between_sd": np.sqrt(var_between),
                 "total_sd": np.sqrt(var_total)}
    if blocks:
        out["between_share"] = {
            name: float(var_between[cols].sum() / var_total[cols].sum()) if var_total[cols].sum() > 0 else 0.0
            for name, cols in blocks.items()}
    return out

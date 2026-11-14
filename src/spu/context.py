"""Context sources (design §7 oracle-vs-classified ablation)."""

from __future__ import annotations

import numpy as np

from .gate import ContextGate


def context_probs(mode: str, classes: list[str], true_exp: np.ndarray | None = None,
                  gate: ContextGate | None = None, X_gate: np.ndarray | None = None) -> np.ndarray:
    """'oracle': one-hot on the folder-structure context (upper bound on expert quality alone).
    'classified': the gate's predicted probabilities (what a deployed system actually gets)."""
    if mode == "oracle":
        g = np.zeros((len(true_exp), len(classes)))
        g[np.arange(len(true_exp)), [classes.index(e) for e in true_exp]] = 1.0
        return g
    if mode == "classified":
        return gate.predict_proba(X_gate)
    raise ValueError(f"context_mode must be 'oracle' or 'classified', got {mode!r}")

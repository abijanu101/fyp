"""Gated mixture of engagement experts (design §8).

    p_engaged = sum_k g_k * p_k(Vec)      (soft, headline)
    p_engaged = p_argmax(g)(Vec)          (hard top-1)
Oracle is either mode with g one-hot on the true context (see spu.context).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .expert import EngagementExpert


class ExpertMixture:
    def __init__(self, classes: list[str], make_expert: Callable[[str | None], EngagementExpert],
                 scope: str = "per_experiment", mode: str = "soft"):
        """make_expert(exp_id) builds an untrained expert; exp_id is None for the shared scope."""
        if scope not in ("per_experiment", "shared") or mode not in ("soft", "hard"):
            raise ValueError(f"bad mixture config: scope={scope!r} mode={mode!r}")
        self.classes, self.scope, self.mode = list(classes), scope, mode
        if scope == "shared":
            shared = make_expert(None)
            self.experts = {c: shared for c in self.classes}
        else:
            self.experts = {c: make_expert(c) for c in self.classes}

    def fit(self, X: np.ndarray, y: np.ndarray, exp: np.ndarray) -> "ExpertMixture":
        """Each expert trains only on windows from its own experiment (§8.3)."""
        if self.scope == "shared":
            self.experts[self.classes[0]].fit(X, y)
        else:
            for c, expert in self.experts.items():
                m = exp == c
                expert.fit(X[m], y[m])
        return self

    def expert_probs(self, X: np.ndarray) -> np.ndarray:
        """(n, K) matrix of every expert's P(engaged)."""
        return np.column_stack([self.experts[c].predict_proba(X) for c in self.classes])

    def predict_proba(self, X: np.ndarray, g: np.ndarray) -> np.ndarray:
        P = self.expert_probs(X)
        if self.mode == "soft":
            return (g * P).sum(axis=1)
        return P[np.arange(len(X)), g.argmax(axis=1)]

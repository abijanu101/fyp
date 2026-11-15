"""SPU context gate (design §7): predicts experiment context, outputs class probabilities that
weight the engagement experts."""

from __future__ import annotations

from functools import partial

import numpy as np

from ..core.classifiers import CLASSIFIERS, build_classifier
from ..core.registry import Registry

GATES = Registry("context gate")


class ContextGate:
    def __init__(self, model: str, classes: list[str], seed: int = 0, **params):
        self.classes = list(classes)
        self.model = build_classifier(model, seed, **params)
        self._only: str | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ContextGate":
        """A single training class cannot be fitted by most classifiers: predict it with certainty."""
        labels = np.unique(np.asarray(y))
        self._only = str(labels[0]) if len(labels) == 1 else None
        if self._only is None:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, len(classes)) probabilities in `self.classes` order; classes absent from training get 0."""
        if self._only is not None:
            out = np.zeros((len(X), len(self.classes)))
            out[:, self.classes.index(self._only)] = 1.0
            return out
        p = self.model.predict_proba(X)
        out = np.zeros((len(X), len(self.classes)))
        for col, c in enumerate(self.model.classes_):
            out[:, self.classes.index(c)] = p[:, col]
        return out


for _name in CLASSIFIERS:
    GATES.register(_name)(partial(ContextGate, _name))

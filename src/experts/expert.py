"""Per-experiment engagement experts (design §8): Vec -> P(window engaged)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial

import numpy as np

from ..core.classifiers import build_classifier
from ..core.registry import Registry

EXPERTS = Registry("engagement expert")


class EngagementExpert(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EngagementExpert": ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n,) probability that each window is engaged."""


class SklearnExpert(EngagementExpert):
    """Small, strongly regularised, class-balanced model (§8.3: at most 6 session-cameras per expert).

    Classes are balanced by `class_weight="balanced"` where the model supports it, and otherwise
    (k-NN) by training on a random class-balanced subsample. `max_train_rows` caps the training
    rows by a class-preserving random subsample, for models whose cost grows quickly with rows
    (SVM with probabilities is roughly quadratic)."""

    WEIGHTED = ("logreg", "random_forest", "svm")

    def __init__(self, model: str, seed: int = 0, feature_names=None, exp_id=None,
                 max_train_rows: int | None = None, **params):
        self.name, self.seed, self.max_train_rows = model, seed, max_train_rows
        extra = {"class_weight": "balanced"} if model in self.WEIGHTED else {}
        self.model = build_classifier(model, seed, **extra, **params)
        self._constant: float | None = None

    def fit(self, X, y):
        y = np.asarray(y, int)
        if len(np.unique(y)) < 2:          # one-class training fold: predict its prior
            self._constant = float(y.mean()) if len(y) else 0.5
            return self
        self._constant = None
        keep = np.arange(len(y))
        rng = np.random.default_rng(self.seed)
        if self.name not in self.WEIGHTED:                      # balance by undersampling the majority class
            n = min(np.bincount(y))
            keep = np.concatenate([rng.choice(np.flatnonzero(y == c), n, replace=False) for c in (0, 1)])
        if self.max_train_rows and len(keep) > self.max_train_rows:
            keep = rng.choice(keep, self.max_train_rows, replace=False)
        if self.name == "knn":                                  # k cannot exceed the training rows
            k = self.model.get_params()["clf__n_neighbors"]
            self.model.set_params(clf__n_neighbors=min(k, len(keep)))
        self.model.fit(np.asarray(X)[keep], y[keep])
        return self

    def predict_proba(self, X):
        if self._constant is not None:
            return np.full(len(X), self._constant)
        return self.model.predict_proba(X)[:, list(self.model.classes_).index(1)]


@EXPERTS.register("hand_weighted")
class HandWeightedExpert(EngagementExpert):
    """Baseline every learned expert must beat (§8.2): sigmoid of a fixed-sign weighted sum of
    standardised Vec features. fit() only learns standardisation statistics, never uses y.

    weights: {vec_key: weight}; phone_key/phone_sign apply the per-experiment phone-use sign."""

    def __init__(self, feature_names: list[str], weights: dict[str, float], exp_id: str | None = None,
                 phone_key: str | None = None, phone_sign: dict[str, float] | None = None, seed: int = 0):
        w = dict(weights)
        if phone_key and phone_sign and exp_id in phone_sign:
            w[phone_key] = w.get(phone_key, 1.0) * phone_sign[exp_id]
        self.idx = [feature_names.index(k) for k in w if k in feature_names]
        self.w = np.array([w[feature_names[i]] for i in self.idx])

    def fit(self, X, y=None):
        cols = np.asarray(X, float)[:, self.idx]
        self.mu = np.nanmean(cols, 0) if len(cols) else np.zeros(len(self.idx))
        self.sd = np.nanstd(cols, 0) + 1e-9 if len(cols) else np.ones(len(self.idx))
        return self

    def predict_proba(self, X):
        z = (np.asarray(X, float)[:, self.idx] - self.mu) / self.sd
        return 1 / (1 + np.exp(-np.nan_to_num(z) @ self.w))


EXPERTS.register("logreg")(partial(SklearnExpert, "logreg"))
EXPERTS.register("random_forest")(partial(SklearnExpert, "random_forest"))
EXPERTS.register("svm")(partial(SklearnExpert, "svm"))
EXPERTS.register("knn")(partial(SklearnExpert, "knn"))

"""Evaluation metrics (design §11): engagement and context. Inter-labeler agreement is in
agreement.py, tracking quality in tracking.py."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score


def binary_metrics(y: np.ndarray, p: np.ndarray, tau: float = 0.5) -> dict:
    """Window-level engaged / not engaged: F1, AUC, balanced accuracy, plus support."""
    y = np.asarray(y, int)
    if len(y) == 0:
        return {"n": 0, "pos_rate": np.nan, "f1": np.nan, "auc": np.nan, "balanced_acc": np.nan}
    pred = (np.asarray(p) >= tau).astype(int)
    both = len(np.unique(y)) == 2       # AUC and balanced accuracy are undefined on one-class folds
    return {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if both else np.nan,
        "balanced_acc": float(balanced_accuracy_score(y, pred)) if both else np.nan,
    }


def context_metrics(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict:
    """6-way context: per-class precision/recall/F1, macro-F1, accuracy."""
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    return {
        "accuracy": float(np.mean(np.asarray(y) == np.asarray(pred))),
        "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(a), "recall": float(b), "f1": float(f), "support": int(s)}
                      for c, a, b, f, s in zip(classes, pr, rc, f1, sup)},
    }

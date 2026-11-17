"""Evaluation metrics (design §11): engagement and context. Inter-labeler agreement is in
agreement.py, tracking quality in tracking.py."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)


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


def spread_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """How the prediction's spread compares with the labels' (population SDs, ddof 0).

    The squared error splits exactly into three parts (r = Pearson correlation of p and y):
        MSE = (mean_p - mean_y)^2  +  (sd_p - r * sd_y)^2  +  (1 - r^2) * sd_y^2
              mse_bias2              mse_spread              mse_floor
    For a given r, the error is smallest at sd_p = r * sd_y: with little signal the right prediction
    stays near the middle, and spreading it further only adds error. `calib_ratio` = sd_p / (r sd_y)
    is 1 for that optimum, < 1 for an over-shrunk model, > 1 for an overconfident one. `share_pred_3`
    is the share of predictions that round to 3 (halves toward 3), the median-hugging check."""
    from ..labels.interp import round_rating

    sd_p, sd_y = float(p.std()), float(y.std())
    r = float(np.corrcoef(p, y)[0, 1]) if sd_p > 0 and sd_y > 0 else np.nan
    cls = round_rating(p).astype(int)
    out = {"pred_mean": float(p.mean()), "label_mean": float(y.mean()),
           "pred_median": float(np.median(p)), "label_median": float(np.median(y)),
           "pred_sd": sd_p, "label_sd": sd_y, "sd_ratio": sd_p / sd_y if sd_y > 0 else np.nan,
           "pearson": r, "mse_bias2": float((p.mean() - y.mean()) ** 2),
           "share_pred_3": float(np.mean(cls == 3)),
           "pred_shares": [float(np.mean(cls == k)) for k in range(1, 6)]}
    rr = 0.0 if np.isnan(r) else r                       # constant prediction: no correlation term
    out["mse_spread"] = float((sd_p - rr * sd_y) ** 2)
    out["mse_floor"] = float((1 - rr ** 2) * sd_y ** 2)
    out["calib_ratio"] = sd_p / (r * sd_y) if np.isfinite(r) and r > 0 else np.nan
    return out


def rating_metrics(rating: np.ndarray, pred: np.ndarray, tau_att: float) -> dict:
    """Per-label scores (docs/misc/targets.md §7.1 items 2 and 7): `rating` = the entries as made
    (1-5), `pred` = predicted ratings (1-5, continuous or already discrete). Regression view on the
    raw prediction; classification views on the prediction rounded to a whole rating (halves toward
    3); binary view at tau_att (engaged = rating >= tau_att), with AUC ranking the raw prediction."""
    from scipy.stats import spearmanr

    from ..labels.interp import round_rating

    y = np.clip(np.asarray(rating, float), 1, 5)
    p = np.clip(np.asarray(pred, float), 1, 5)
    if len(y) == 0:
        return {"n": 0}
    cls = round_rating(p).astype(int)
    yi = np.rint(y).astype(int)
    yb, pb = (y >= tau_att).astype(int), (cls >= tau_att).astype(int)
    both = len(np.unique(yb)) == 2
    varied = np.ptp(p) > 0 and np.ptp(y) > 0
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(p - y))),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        **spread_metrics(y, p),
        "spearman": float(spearmanr(p, y).statistic) if varied else np.nan,
        "accuracy": float(np.mean(cls == yi)),
        "within_1": float(np.mean(np.abs(cls - yi) <= 1)),
        "macro_f1": float(f1_score(yi, cls, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)),
        "auc": float(roc_auc_score(yb, p)) if both else np.nan,
        "balanced_acc": float(balanced_accuracy_score(yb, pb)) if both else np.nan,
        "engaged_rate": float(yb.mean()),
        # rows = true rating 1-5 / not engaged, engaged; columns = predicted (rounded) the same way
        "confusion": confusion_matrix(yi, cls, labels=[1, 2, 3, 4, 5]).tolist(),
        "confusion_binary": confusion_matrix(yb, pb, labels=[0, 1]).tolist(),
    }

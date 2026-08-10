"""Metric and cost helpers kept independent from model inference for testing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve


@dataclass(frozen=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    average_precision: float
    pr_auc: float
    false_negative_rate: float
    tp: int
    fp: int
    tn: int
    fn: int
    support: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def threshold_predictions(y_score: Iterable[float], threshold: float) -> np.ndarray:
    """Apply the historical repo rule: a score must be strictly above threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1; received {threshold}")
    scores = np.asarray(y_score, dtype=float)
    if np.any(~np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("y_score must contain finite sigmoid probabilities in [0, 1]")
    return (scores > threshold).astype(np.int8)


def calculate_binary_metrics(
    y_true: Iterable[int], y_score: Iterable[float], threshold: float
) -> BinaryMetrics:
    """Calculate threshold-dependent and probability-ranking metrics for one class."""
    truth = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(y_score, dtype=float)
    if truth.ndim != 1 or scores.ndim != 1 or truth.size != scores.size:
        raise ValueError("y_true and y_score must be one-dimensional arrays of equal length")
    if truth.size == 0 or np.any((truth != 0) & (truth != 1)):
        raise ValueError("y_true must be a non-empty binary array")

    prediction = threshold_predictions(scores, threshold)
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fnr = fn / (tp + fn) if tp + fn else 0.0

    # All supplied Severstal classes contain positives.  Returning NaN makes an
    # undefined score explicit for a future split that does not.
    if truth.sum() == 0:
        average_precision = float("nan")
        pr_auc = float("nan")
    else:
        precision_curve, recall_curve, _ = precision_recall_curve(truth, scores)
        average_precision = float(average_precision_score(truth, scores))
        pr_auc = float(auc(recall_curve[::-1], precision_curve[::-1]))

    return BinaryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        average_precision=average_precision,
        pr_auc=pr_auc,
        false_negative_rate=fnr,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        support=int(truth.sum()),
    )


def calculate_cost(fn: int, fp: int, fn_cost: float, fp_cost: float) -> float:
    """Cost model specified for this project: FN_cost * FN + FP_cost * FP."""
    if fn < 0 or fp < 0:
        raise ValueError("fn and fp must be non-negative")
    if fn_cost < 0 or fp_cost < 0:
        raise ValueError("costs must be non-negative")
    return float(fn_cost * fn + fp_cost * fp)

"""Binary classification metrics used during training.

Phase 2 only needs them to monitor training and pick the best epoch. Phase 3
adds threshold tuning, confidence intervals and per-group reports.

Why these:
- PR-AUC (average precision): threshold-free and focused on the positive
  class. With 17% positives it is far more informative than accuracy, and it
  is what we select the best epoch on.
- ROC-AUC: threshold-free ranking quality; easy to compare across papers.
- recall / precision / specificity at 0.5: a first look only. 0.5 is NOT the
  operating threshold; Phase 3 tunes it on validation.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(y_true, prob, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def ratio(a: int, b: int) -> float:
        return a / b if b else math.nan

    both_classes = 0 < y.sum() < len(y)
    recall = ratio(tp, tp + fn)
    precision = ratio(tp, tp + fp)
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else math.nan,
        "pr_auc": float(average_precision_score(y, p)) if both_classes else math.nan,
        "recall": recall,
        "precision": precision,
        "specificity": ratio(tn, tn + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "accuracy": ratio(tp + tn, len(y)),
        "positive_rate": float(y.mean()) if len(y) else math.nan,
    }

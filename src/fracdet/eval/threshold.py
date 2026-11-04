"""Choosing the decision threshold, on validation only.

A classifier outputs a probability; the threshold turns it into a yes/no.
0.5 is arbitrary for an imbalanced problem: in Phase 2 it missed 44% of
validation fractures. A screening tool should miss few fractures and accept
more false alarms, so the rule here is:

    **the highest threshold that still catches at least ``target_recall`` of
    the validation fractures.**

Highest, because among all thresholds meeting the recall target it raises the
fewest false alarms (best precision and specificity). The rule is fixed in
``configs/eval.yaml`` and the resulting number is committed to git *before* the
test set is scored, so the test score cannot have influenced it.

Where exactly to put the threshold: the k-th highest fracture score ``s_k``
already gives the target recall, and so does any value down to the next lower
score ``s_next``. We take the midpoint of that gap. Validation results are
identical anywhere in the gap, but the midpoint leaves a margin on both sides,
so tiny numeric differences between machines (AMP on another GPU, say) don't
flip images that sit exactly on the line.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fracdet.eval.metrics import eval_metrics

RULE_MAX_THRESHOLD_AT_RECALL = "max_threshold_at_recall"


def operating_points(y_true, prob) -> pd.DataFrame:
    """One row per distinct score: the counts and rates if that score were the threshold.

    Sorted from the highest threshold (few alarms) to the lowest (everything
    flagged). Built with cumulative sums, so it is O(n log n), not one
    confusion matrix per candidate.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    p_sorted, y_sorted = p[order], y[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    # Predicting p >= t flags every sample with the same score too, so read the
    # cumulative counts at the LAST position of each distinct score.
    last_of_value = np.r_[p_sorted[1:] != p_sorted[:-1], True]
    tp, fp = tp_cum[last_of_value], fp_cum[last_of_value]
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    fn, tn = n_pos - tp, n_neg - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        df = pd.DataFrame(
            {
                "threshold": p_sorted[last_of_value],
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "recall": tp / n_pos if n_pos else np.nan,
                "precision": tp / (tp + fp),
                "specificity": tn / n_neg if n_neg else np.nan,
                "f1": 2 * tp / (2 * tp + fp + fn),
            }
        )
    df["youden_j"] = df["recall"] + df["specificity"] - 1
    return df.reset_index(drop=True)


def threshold_for_recall(y_true, prob, target_recall: float) -> dict:
    """The highest threshold whose recall on (y_true, prob) is >= target_recall.

    Returns the threshold, the gap it sits in, and the counts it gives.
    """
    if not 0 < target_recall <= 1:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    pos_scores = np.sort(p[y == 1])[::-1]
    if len(pos_scores) == 0:
        raise ValueError("No positives: recall is undefined")

    # k positives must score >= threshold. The tiny epsilon stops float error in
    # target * n (e.g. 0.9 * 100 = 90.00000000000001) from asking for one more.
    k = math.ceil(target_recall * len(pos_scores) - 1e-9)
    s_k = float(pos_scores[k - 1])
    below = p[p < s_k]
    s_next = float(below.max()) if len(below) else 0.0
    threshold = (s_k + s_next) / 2
    return {
        "threshold": threshold,
        "gap": [s_next, s_k],
        "positives_required": k,
        "metrics": eval_metrics(y, p, threshold),
    }


def reference_points(y_true, prob) -> dict[str, dict]:
    """Other common threshold choices, reported on validation for context only.

    - ``default_0.5``: what Phase 2 reported
    - ``max_f1``: best balance of precision and recall
    - ``max_youden``: maximises recall + specificity - 1 (the ROC point
      furthest above the diagonal)

    None of these is the operating threshold; they show what the recall-first
    rule costs and buys compared with the usual alternatives.
    """
    ops = operating_points(y_true, prob)
    out = {"default_0.5": 0.5}
    out["max_f1"] = float(ops.loc[ops["f1"].idxmax(), "threshold"])
    out["max_youden"] = float(ops.loc[ops["youden_j"].idxmax(), "threshold"])
    return {
        name: {"threshold": t, "metrics": eval_metrics(y_true, prob, t)} for name, t in out.items()
    }


def recall_table(y_true, prob, targets=(0.80, 0.85, 0.90, 0.95, 0.98)) -> pd.DataFrame:
    """What each candidate recall target would cost on validation (for choosing the target)."""
    rows = []
    for target in targets:
        r = threshold_for_recall(y_true, prob, target)
        m = r["metrics"]
        rows.append(
            {
                "target_recall": target,
                "threshold": r["threshold"],
                "recall": m["recall"],
                "precision": m["precision"],
                "specificity": m["specificity"],
                "tp": m["tp"],
                "fn": m["fn"],
                "fp": m["fp"],
                "alarms_per_100": 100 * (m["tp"] + m["fp"]) / len(np.asarray(y_true)),
            }
        )
    return pd.DataFrame(rows)

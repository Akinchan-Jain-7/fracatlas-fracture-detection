"""The full metric set at one decision threshold.

``fracdet.train.metrics.binary_metrics`` is what training logs every epoch.
Evaluation needs a few more numbers, so this module extends it rather than
changing what training writes:

- confusion counts (tp/fp/tn/fn): every rate below is a ratio of these, and
  reporting the raw counts lets a reader check them ("57 of 102", not "0.559");
- NPV (negative predictive value): of the images the model clears, how many are
  really normal. For a screening tool that "clears" X-rays this matters as much
  as precision;
- Brier score: mean squared error of the probabilities. Lower is better; it
  measures calibration as well as ranking (0.25 = always answering 0.5).
"""

from __future__ import annotations

import math

import numpy as np

from fracdet.train.metrics import binary_metrics


def confusion_counts(y_true, prob, threshold: float) -> dict[str, int]:
    """tp / fp / tn / fn when predicting "fractured" for ``prob >= threshold``."""
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(prob, dtype=float) >= threshold
    return {
        "tp": int((pred & (y == 1)).sum()),
        "fp": int((pred & (y == 0)).sum()),
        "tn": int((~pred & (y == 0)).sum()),
        "fn": int((~pred & (y == 1)).sum()),
    }


COUNT_KEYS = ("tp", "fp", "tn", "fn")


def eval_metrics(y_true, prob, threshold: float) -> dict[str, float | int]:
    """``binary_metrics`` plus NPV, Brier score and the confusion counts (as ints)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    c = confusion_counts(y, p, threshold)
    npv_den = c["tn"] + c["fn"]
    return {
        **binary_metrics(y, p, threshold),
        "npv": c["tn"] / npv_den if npv_den else math.nan,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else math.nan,
        **c,
    }

"""Confidence intervals by cluster bootstrap.

**Why intervals at all.** The test set has 102 fractures. Catching 90 of them
reads as "recall 0.88", but a different 102 fractures from the same hospitals
could easily give 0.82 or 0.93. A bare number hides that; an interval shows how
much of a difference between two models is noise.

**Why bootstrap.** Resample the test set with replacement many times,
recompute each metric on every resample, and take the 2.5th and 97.5th
percentiles. It works the same way for recall, PR-AUC or anything else, with
no formula per metric.

**Why *cluster* bootstrap.** Test images are not independent. Phase 1 grouped
similar images and adjacent IDs (likely the same patient) into ``split_group``,
and the test set's 102 fractures come from only about 47 groups. Resampling
single images would pretend we have 102 independent fractures and give
intervals that are too narrow. Resampling whole groups keeps correlated images
together, so the interval reflects how many *independent* cases we really have.

The threshold is held fixed (it came from validation), so these intervals
cover sampling noise in the test set, not uncertainty in the threshold itself.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from fracdet.eval.metrics import COUNT_KEYS, eval_metrics

MetricFn = Callable[[np.ndarray, np.ndarray], dict[str, float]]


def cluster_bootstrap(
    y_true,
    prob,
    groups,
    metric_fn: MetricFn,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Percentile intervals for every metric ``metric_fn`` returns.

    Args:
        groups: one cluster id per sample; clusters are resampled whole.
            Pass ``np.arange(n)`` for the ordinary per-image bootstrap.

    Returns ``{metric: {"value", "low", "high", "n_valid"}}``. ``value`` is the
    metric on the real data, not the bootstrap mean. ``n_valid`` counts
    resamples where the metric was defined (ROC-AUC needs both classes); a
    resample where it was undefined is skipped rather than counted as 0.
    """
    y = np.asarray(y_true)
    p = np.asarray(prob, dtype=float)
    g = np.asarray(groups)
    if not len(y) == len(p) == len(g):
        raise ValueError("y_true, prob and groups must have the same length")

    point = metric_fn(y, p)
    # Map each cluster to the row positions it owns, once.
    _, group_idx = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(group_idx == k) for k in range(group_idx.max() + 1)]
    n_groups = len(members)

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        picked = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([members[k] for k in picked])
        for k, v in metric_fn(y[rows], p[rows]).items():
            samples[k].append(v)

    out = {}
    for k, v in point.items():
        arr = np.asarray(samples[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        low, high = (
            np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            if len(arr)
            else (np.nan,) * 2
        )
        out[k] = {"value": v, "low": float(low), "high": float(high), "n_valid": int(len(arr))}
    return out


def metric_fn_at(threshold: float) -> MetricFn:
    """The evaluation metrics at a fixed threshold, minus raw counts.

    Counts are dropped: an interval on "tp" from resamples of varying size has
    no useful meaning, while rates do.
    """

    def fn(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
        m = eval_metrics(y, p, threshold)
        return {k: float(v) for k, v in m.items() if k not in COUNT_KEYS}

    return fn

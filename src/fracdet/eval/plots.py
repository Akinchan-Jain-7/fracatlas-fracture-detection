"""Evaluation figures, in the same style as the Phase 2 learning curves.

- threshold sweep (validation): how recall, precision and specificity trade off
  as the threshold moves, with the chosen threshold marked;
- PR and ROC curves: validation and test overlaid, operating point marked;
- confusion matrix at the operating threshold.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1f9e74"]  # categorical slots 1-3
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DASH = (0, (4, 3))


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
            "axes.titleweight": "bold", "axes.titlelocation": "left",
            "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
            "grid.color": GRID, "xtick.color": INK_2, "ytick.color": INK_2,
            "font.size": 10, "savefig.dpi": 160, "savefig.bbox": "tight",
        }
    )  # fmt: skip


def plot_threshold_sweep(ops: pd.DataFrame, threshold: float, out: Path, title: str) -> None:
    """Recall / precision / specificity against the threshold (log x: scores bunch near 0)."""
    _style()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    t = ops["threshold"].clip(lower=1e-4)
    for (col, label), color in zip(
        [("recall", "recall"), ("precision", "precision"), ("specificity", "specificity")],
        SERIES,
        strict=True,
    ):
        ax.plot(t, ops[col], color=color, linewidth=2, label=label, drawstyle="steps-post")
    ax.axvline(threshold, color=INK_2, linewidth=1, linestyle=DASH)
    ax.text(threshold * 1.08, 0.04, f"chosen {threshold:.3f}", color=INK_2, fontsize=9)
    ax.axvline(0.5, color=GRID, linewidth=1)
    ax.text(0.5 * 1.08, 0.12, "0.5 (Phase 2)", color=INK_2, fontsize=9)  # own row: no overlap
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Threshold (predict fracture if probability ≥ threshold)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="center left")
    fig.savefig(out)
    plt.close(fig)


def plot_pr_roc(
    sets: dict[str, tuple[np.ndarray, np.ndarray]], threshold: float, out: Path, title: str
) -> None:
    """PR (left) and ROC (right) for each named (labels, probs) set, operating point marked."""
    _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for (name, (y, p)), color in zip(sets.items(), SERIES, strict=False):
        y, p = np.asarray(y), np.asarray(p)
        prec, rec, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        a.plot(rec, prec, color=color, linewidth=2, label=name, drawstyle="steps-post")
        b.plot(fpr, tpr, color=color, linewidth=2, label=name)
        pred = p >= threshold
        tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
        r_op = tp / max(y.sum(), 1)
        a.plot(r_op, tp / max(tp + fp, 1), "o", color=color, markersize=7,
               markeredgecolor=SURFACE)  # fmt: skip
        b.plot(fp / max((y == 0).sum(), 1), r_op, "o", color=color, markersize=7,
               markeredgecolor=SURFACE)  # fmt: skip
        a.axhline(y.mean(), color=color, linewidth=1, linestyle=DASH, alpha=0.6)
    b.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    a.set(xlabel="Recall (fractures found)", ylabel="Precision (alarms that are real)",
          xlim=(0, 1.01), ylim=(0, 1.02), title="Precision–recall")  # fmt: skip
    b.set(xlabel="False-positive rate (1 − specificity)", ylabel="Recall",
          xlim=(0, 1), ylim=(0, 1.02), title="ROC")  # fmt: skip
    a.text(0.02, 0.03, "dashed: random-ranking baseline (fracture rate)\ndots: operating threshold",
           color=INK_2, fontsize=8)  # fmt: skip
    for ax in (a, b):
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.suptitle(title, x=0.01, y=1.03, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(out)
    plt.close(fig)


def plot_confusion(counts: dict[str, int], out: Path, title: str) -> None:
    """2x2 confusion matrix; rows = truth, columns = prediction, counts and row %."""
    _style()
    grid = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    row_share = grid / grid.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1.4)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i, j]}\n{row_share[i, j]:.0%} of row", ha="center",
                    va="center", color=INK, fontsize=10)  # fmt: skip
    ax.set_xticks([0, 1], ["normal", "fracture"])
    ax.set_yticks([0, 1], ["normal", "fracture"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out)
    plt.close(fig)

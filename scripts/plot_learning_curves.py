"""Plot a run's learning curves: losses, and validation ranking metrics.

Two separate panels (never a dual y-axis): loss has no fixed scale, while
PR-AUC / ROC-AUC live on [0, 1].

Usage:
    python scripts/plot_learning_curves.py outputs/runs/<run>/metrics.csv out.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"  # validated categorical slots 1-2
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot(metrics: pd.DataFrame, out: Path, best_epoch: int | None = None) -> None:
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
    pr_col = "val_pr_auc"
    roc_col = "val_roc_auc"
    if best_epoch is None:
        best_epoch = int(metrics.loc[metrics[pr_col].idxmax(), "epoch"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, cols, title in [
        (a, [("train_loss", "train loss"), ("val_loss", "val loss")], "Loss"),
        (b, [(pr_col, "val PR-AUC"), (roc_col, "val ROC-AUC")], "Validation ranking"),
    ]:
        for (col, label), color in zip(cols, [SERIES_1, SERIES_2], strict=True):
            ax.plot(metrics.epoch, metrics[col], color=color, linewidth=2, label=label)
            ax.text(metrics.epoch.iloc[-1] + 0.3, metrics[col].iloc[-1], label,
                    color=INK, fontsize=9, va="center")  # fmt: skip
        ax.axvline(best_epoch, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="best")
        ax.set_xlim(metrics.epoch.min() - 0.5, metrics.epoch.max() + 4)
    b.set_ylim(0, 1)
    b.text(best_epoch + 0.3, 0.05, f"best epoch {best_epoch}", color=INK_2, fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    df = pd.read_csv(args.metrics_csv)
    # accept both the run's metrics.csv and the rounded copy in results/
    df = df.rename(columns={"val_recall@0.5": "val_recall", "val_precision@0.5": "val_precision"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(df, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

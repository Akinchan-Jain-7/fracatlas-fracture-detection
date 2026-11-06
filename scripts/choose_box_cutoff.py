"""Phase 8: choose the score above which the app shows *extra* detector boxes.

The app always shows the detector's top box on an X-ray the classifier flags.
Further boxes are shown only if their score is at least this cut-off. It is
chosen on VALIDATION (seed 42's boxes), never on test, with a false-box budget:
the lowest score at which normal validation X-rays get on average at most
``--budget`` boxes (every box on a normal X-ray is a false box).

Usage:
    python scripts/choose_box_cutoff.py            # prints the value for configs/serve.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

DEFAULT_CSV = "results/phase7_detect/val_s42/val_detections.csv"


def box_cutoff(detections: pd.DataFrame, budget: float) -> float:
    """Lowest observed score s with mean(#boxes scoring >= s per normal image) <= budget."""
    n_normal = detections.loc[detections.label == 0, "image_id"].nunique()
    if n_normal == 0:
        raise ValueError("no normal images in the detections")
    scores = np.sort(
        detections.loc[(detections.label == 0) & (detections["rank"] >= 0), "score"].to_numpy()
    )[::-1]
    allowed = int(np.floor(budget * n_normal))  # this many false boxes in total
    if allowed >= len(scores):
        return float(scores[-1]) if len(scores) else 0.0
    # Boxes strictly above the (allowed+1)-th highest score stay within the budget.
    return float(np.nextafter(scores[allowed], 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--budget", type=float, default=0.1, help="false boxes per normal X-ray")
    args = parser.parse_args()
    d = pd.read_csv(args.csv)
    c = box_cutoff(d, args.budget)
    n_normal = d.loc[d.label == 0, "image_id"].nunique()
    shown = ((d.label == 0) & (d["rank"] >= 0) & (d.score >= c)).sum()
    print(f"cut-off {c:.4f}: {shown} boxes on {n_normal} normal validation X-rays "
          f"({shown / n_normal:.3f} per image, budget {args.budget})")  # fmt: skip


if __name__ == "__main__":
    main()

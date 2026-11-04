"""Compare a sweep's runs on validation and apply its pre-registered selection rule.

Reads the sweep manifest (outputs/sweeps/<name>/manifest.json) for the run
folders, so only this sweep's runs are compared. Writes to --out:
    runs.csv        one row per run (val PR-AUC, specificity at the recall target, ...)
    compare.json    per-variant mean / sd / paired delta, and the decision
    compare.md      the table, ready to paste
    compare.png     every seed as a dot, per variant

Usage:
    python scripts/compare_runs.py configs/sweep_phase4.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from fracdet.eval.compare import aggregate, decide, markdown, run_record  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1f9e74", "#8a5cc2", "#c9a227"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, 6)
    return x


def plot(runs: pd.DataFrame, order: list[str], out: Path, recall: float) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (col, title) in zip(
        axes,
        [
            ("pr_auc", "Validation PR-AUC"),
            ("spec_at_recall", f"Specificity at recall ≥ {recall:.0%}"),
        ],
        strict=True,
    ):
        for i, exp in enumerate(order):
            v = runs.loc[runs.experiment == exp, col]
            ax.scatter([i] * len(v), v, color=SERIES[i % len(SERIES)], s=36, zorder=3)
            ax.hlines(v.mean(), i - 0.25, i + 0.25, color=INK, linewidth=2, zorder=4)
        ax.set_xticks(range(len(order)), [o.replace("imb_", "") for o in order], rotation=15)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].text(0, 1.02, "dots: seeds · bar: mean", transform=axes[0].transAxes,
                 color=INK_2, fontsize=8)  # fmt: skip
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    sweep = yaml.safe_load(args.sweep.read_text())
    sel = sweep["selection"]
    out = args.out or Path("outputs/sweeps") / sweep["name"]
    manifest = json.loads((args.manifest or out / "manifest.json").read_text())
    done = [j for j in manifest["jobs"] if j["status"] == "done"]
    runs = pd.DataFrame([run_record(j["run_dir"], sel["recall_target"]) for j in done])
    order = [
        e
        for e in dict.fromkeys(j["experiment"] for j in manifest["jobs"])
        if e in set(runs.experiment)
    ]
    runs["experiment"] = pd.Categorical(runs.experiment, order, ordered=True)
    runs = runs.sort_values(["experiment", "seed"]).reset_index(drop=True)
    runs["experiment"] = runs.experiment.astype(str)

    agg = aggregate(runs, sweep["baseline"])
    decision = decide(agg, runs, sweep["baseline"], sel["primary"], sel["guard"])
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    table = markdown(agg, sweep["baseline"])
    (out / "compare.md").write_text(table + "\n")
    result = {
        "sweep": sweep["name"],
        "recall_target": sel["recall_target"],
        "n_runs": len(runs),
        "n_failed": len(manifest["jobs"]) - len(done),
        "decision": decision,
        "variants": agg.to_dict(orient="records"),
    }
    (out / "compare.json").write_text(json.dumps(_clean(result), indent=2, allow_nan=False) + "\n")
    plot(runs, order, out / "compare.png", sel["recall_target"])
    print(table)
    print("\nDECISION " + json.dumps(_clean(decision)))


if __name__ == "__main__":
    main()

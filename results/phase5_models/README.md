# Phase 5: resolution and backbone sweep (validation only)

| File | What |
|---|---|
| `runs.csv` | One row per run (15): best epoch, minutes, val PR-AUC / ROC-AUC, threshold, specificity / precision / alarms per 100 at recall ≥ 0.90, PR-AUC without hardware images |
| `compare.json` | Per-variant mean / sd / paired delta vs baseline, and the pre-registered decision |
| `compare.md` | The comparison table |

`compare.json` and `compare.md` are byte-identical to the Kaggle run's files (SHA-256).
`runs.csv` was rebuilt from the output viewer and reproduces `compare.json` exactly.
The figure is `docs/figures/phase-5/compare.png`.

**Kaggle:** notebook `notebook34061d5c16`, Version 2 ("phase5 models sweep (5 variants x 3 seeds)"), Save & Run All, T4 x2,
98 min, code at commit `1f7d7c8`. Each run's `best.pt`, predictions and logs are in that version's output
under `outputs/runs/`.

**Decision:** **ConvNeXt-Tiny wins**, with gain in specificity at 90% recall +0.124 against seed noise 0.091,
and PR-AUC +0.105. Its representative (median-seed) checkpoint is
`outputs/runs/20261104-110146_p5_convnext_tiny_s44`. The test set has not been used yet.

# Phase 5 winner: ConvNeXt-Tiny (seed 44), one-shot test evaluation

Same protocol as the baseline (`results/phase3_baseline_resnet50/`, `docs/phase-3-evaluation.md`).

| File | What | Origin |
|---|---|---|
| `threshold.json` | Operating threshold 0.0968 (val recall ≥ 0.90) and how it was chosen | Pre-registered in commit `c010ec8`, before the test run |
| `test_metrics.json` | Test + validation metrics per subgroup, 95% cluster-bootstrap CIs, metrics at 0.5, protocol record | Written by `scripts/evaluate_test.py` |
| `report.md` | The same tables in markdown | Written by `scripts/evaluate_test.py` |

All three are byte-identical to the files the Kaggle run wrote (checked by
SHA-256 against the notebook output viewer), apart from the trailing newline
the viewer strips.

**Kaggle:** notebook `notebook2ac59126bb`, runner `notebooks/kaggle_test_eval.ipynb`.
- **Version 4** ("convnext s44 step1 threshold (val only)"): `outputs/eval/p5_convnext_tiny_s44/tune/`.
- **Version 5** ("convnext s44 step2 test evaluation (one-shot)"): `outputs/eval/p5_convnext_tiny_s44/test/`.
  That folder holds `test_predictions.csv` (one row per test image), the figures and the logs.

**Model:** `best.pt` of run `p5_convnext_tiny_s44` (epoch 4, trained at commit `1f7d7c8`),
the median seed of the Phase 5 winner, read from the Phase 5 sweep's output.
**Evaluation code:** commit `c010ec8`. **GPU:** Tesla T4.
**Protocol checks:** threshold committed before scoring; same model (epoch, commit, split MD5);
validation re-scored with PR-AUC identical and 0 flipped decisions; test scored once.

Headline (test, 589 images, 102 fractures), with the ResNet-50 baseline for comparison:

| Metric | ConvNeXt-Tiny [95% CI] | ResNet-50 baseline |
|---|---|---|
| Recall | **0.922** [0.859, 0.976] (94 / 102) | 0.892 (91 / 102) |
| Specificity | **0.708** [0.643, 0.764] | 0.548 |
| Precision | 0.398 [0.298, 0.498] | 0.293 |
| PR-AUC | **0.853** [0.776, 0.914] | 0.739 |
| ROC-AUC | 0.937 [0.902, 0.967] | 0.870 |
| Flagged per 100 X-rays | 40.1 | 52.8 |
| PR-AUC without hardware | 0.825 [0.731, 0.897] | 0.695 |

- Hardware images: 11 of 11 fractures found.
- By region: hand recall 0.963 but specificity only 0.298 (baseline 0.347); leg recall
  0.800 (24 / 30, baseline 27 / 30), specificity 0.867; mixed 17 / 17. The overall gain is
  not uniform: ConvNeXt misses 3 more leg fractures than the baseline.
- At the default threshold 0.5 (not the operating point): recall 0.784, specificity 0.955.

# Phase 3: baseline ResNet-50, one-shot test evaluation

| File | What | Origin |
|---|---|---|
| `threshold.json` | Operating threshold 0.00477 (val recall ≥ 0.90) and how it was chosen | Pre-registered in commit `c805cfa`, before the test run |
| `test_metrics.json` | Test + validation metrics per subgroup, 95% cluster-bootstrap CIs, metrics at 0.5, protocol record | Written by `scripts/evaluate_test.py` |
| `report.md` | The same tables in markdown | Written by `scripts/evaluate_test.py` |

All three are byte-identical to the files the Kaggle run wrote (checked by
SHA-256 against the notebook output viewer), apart from the trailing newline
the viewer strips.

**Kaggle:** notebook `notebook2ac59126bb`.
- **Version 2** ("phase3 step1 threshold tuning (val only)"): `outputs/eval/baseline_resnet50/tune/`.
- **Version 3** ("phase3 step2 test evaluation (one-shot)"): `outputs/eval/baseline_resnet50/test/`.
  That folder holds `test_predictions.csv` (one row per test image), the figures and the logs.
  A compressed copy of all of them is in `p3.tar.xz`.

**Model:** Phase 2 `best.pt` (epoch 16, trained at commit `a579c9f`), read from notebook Version 1's output.
**Evaluation code:** commit `c805cfa`. **GPU:** Tesla T4.

Headline (test, 589 images, 102 fractures):
- **Recall 0.892** [0.825, 0.955]
- Precision 0.293 [0.208, 0.379]
- Specificity 0.548 [0.496, 0.600]
- PR-AUC 0.739 [0.625, 0.832]
- Without hardware images: recall 0.879, PR-AUC 0.695

See `docs/phase-3-evaluation.md`.

# Bone Fracture Detection from X-rays (FracAtlas)

> 🚧 Work in progress, built phase by phase. Results tables, Grad-CAM examples and
> the live demo link will be added only from real, logged runs.

Binary fracture classification (and later localization) on musculoskeletal
radiographs from the [FracAtlas](https://doi.org/10.6084/m9.figshare.22363012)
dataset: 4,083 X-rays, about 18% fractured.

**Not a medical device. For education and research only.**

## Results so far

Current model: ConvNeXt-Tiny (ImageNet-22k pretrained), fine-tuned on grayscale 384 px X-rays.
It replaced the ResNet-50 baseline after winning a seeded comparison on validation (Phase 5).
**Test set, scored once**, with a threshold fixed on validation beforehand:

| Metric | Value [95% CI] | ResNet-50 baseline | Meaning |
|---|---|---|---|
| Recall | **0.922** [0.859, 0.976] | 0.892 | 94 of 102 fractures found |
| Specificity | **0.708** [0.643, 0.764] | 0.548 | 7 in 10 normal X-rays are cleared |
| Flagged per 100 X-rays | 40 | 53 | about a quarter fewer false alarms to review |
| PR-AUC | **0.853** [0.776, 0.914] | 0.739 | 5× the random baseline (0.17) |
| PR-AUC without surgical hardware | 0.825 | 0.695 | the gain is not just the metal shortcut |

Known weak spots: normal hand X-rays are still mostly flagged (specificity 0.30), and leg
recall is 0.80 (24 / 30). Details are in [Phase 5](docs/phase-5-models.md).

Where it looks ([Phase 6](docs/phase-6-explainability.md)): Grad-CAM heatmap peaks land on the
radiologist's fracture box in 22.5% of test fractures, against 7.8% for an "image centre" guess
and 1% by chance. None of the 18 missed fractures had the heatmap on the fracture: the misses are
fractures the model never located, mostly in feet and ankles.

How these numbers were kept honest:
- **Leakage-safe split.** Near-duplicate and likely same-patient images never cross splits (Phase 1).
- **Threshold pre-registered.** It was chosen on validation and committed to git before the test set was opened. The evaluation script refuses to run otherwise (Phase 3).
- **Patient-level uncertainty.** The confidence intervals resample whole patient groups: 102 fractures come from only 47 groups.
- **Seeded comparisons.** Model changes are compared over 3 seeds on validation, with the selection rule fixed before the runs. Weighted loss, oversampling and focal loss all failed that test (Phase 4). The new backbone passed it and was then confirmed by its own single test evaluation (Phase 5).

## Repository layout

```
configs/      YAML experiment configs (every run is defined by a config file)
src/fracdet/  installable Python package: config, data, models, training, evaluation
scripts/      command-line entry points (download data, verify data, train, ...)
notebooks/    thin Kaggle/Colab runners and EDA; no core logic lives in notebooks
splits/       the committed train/val/test split (identical on every machine)
tests/        pytest unit tests (run in CI on every push)
docs/         per-phase design notes and interview Q&A
app/          FastAPI service + web frontend (Phase 8)
```

## Quick start (local)

```bash
git clone https://github.com/Akinchan-Jain-7/fracatlas-fracture-detection.git
cd fracatlas-fracture-detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python scripts/download_data.py          # ~340 MB from Figshare, checksum-verified
python scripts/verify_data.py            # checks layout and counts
python scripts/make_split.py --verify    # rebuilds the split, checks it matches splits/
pytest                                   # unit tests
```

## Progress

| Phase | Status | Notes |
|---|---|---|
| 0 Setup | ✅ | [docs](docs/phase-0-setup.md): config, seeding, checksum-verified data, CI |
| 1 Data | ✅ | [docs](docs/phase-1-data.md): EDA, shortcuts, duplicates, leakage-safe split |
| 2 Baseline | ✅ | [docs](docs/phase-2-baseline.md): ResNet-50 transfer learning; **val PR-AUC 0.714, ROC-AUC 0.869** (test not yet used) |
| 3 Evaluation | ✅ | [docs](docs/phase-3-evaluation.md): threshold pre-registered on val, test scored once. **Test recall 0.892 [0.825, 0.955] at specificity 0.548; PR-AUC 0.739 [0.625, 0.832]** (0.695 without hardware images) |
| 4 Class imbalance | ✅ | [docs](docs/phase-4-imbalance.md): weighted BCE, balanced sampler, focal loss vs baseline, 3 seeds each, rule fixed before the runs. **No remedy helped**: all lowered val specificity at 90% recall (Δ −0.045 to −0.074); baseline kept |
| 5 Model comparison | ✅ | [docs](docs/phase-5-models.md): 4 challengers × 3 seeds; **ConvNeXt-Tiny won** (val PR-AUC 0.709 → 0.814) and was **confirmed on test, scored once: recall 0.922 [0.859, 0.976] at specificity 0.708 [0.643, 0.764], PR-AUC 0.853** (baseline 0.892 / 0.548 / 0.739). It is now the current model. 448 px resolution gave nothing |
| 6 Explainability | ✅ | [docs](docs/phase-6-explainability.md): Grad-CAM scored against the radiologist fracture boxes, plan committed first. **Test pointing-game hit rate 0.225 [0.135, 0.318]** vs centre guess 0.078 and chance 0.010; **0 of 18 missed fractures** had the heatmap on the fracture; random-weight maps fall to 0.03 (sanity check passed); heat follows surgical hardware |

## Quick start (Kaggle)

See [`docs/phase-0-setup.md`](docs/phase-0-setup.md) for the full guide and
[`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb) for the runner.

## Dataset citation

Abedeen, I., Rahman, M. A., Prottyasha, F. Z., et al. *FracAtlas: A Dataset for
Fracture Classification, Localization and Segmentation of Musculoskeletal
Radiographs.* Scientific Data 10, 521 (2023). Data licensed CC BY 4.0.

## License

Code: MIT. Dataset: CC BY 4.0 (not redistributed here).

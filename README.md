# Bone Fracture Detection from X-rays (FracAtlas)

> 🚧 Work in progress, built phase by phase. Results tables, Grad-CAM examples and
> the live demo link will be added only from real, logged runs.

Binary fracture classification (and later localization) on musculoskeletal
radiographs from the [FracAtlas](https://doi.org/10.6084/m9.figshare.22363012)
dataset: 4,083 X-rays, about 18% fractured.

**Not a medical device. For education and research only.**

## Results so far

Current model: ImageNet-pretrained ResNet-50, fine-tuned on grayscale 384 px X-rays.
**Test set, scored once**, with a threshold fixed on validation beforehand:

| Metric | Value [95% CI] | Meaning |
|---|---|---|
| Recall | **0.892** [0.825, 0.955] | 91 of 102 fractures found |
| Specificity | 0.548 [0.496, 0.600] | about half the normal X-rays are cleared |
| PR-AUC | 0.739 [0.625, 0.832] | 4× the random baseline (0.17) |
| PR-AUC without surgical hardware | 0.695 | metal is part of what the model uses |

How these numbers were kept honest:
- **Leakage-safe split.** Near-duplicate and likely same-patient images never cross splits (Phase 1).
- **Threshold pre-registered.** It was chosen on validation and committed to git before the test set was opened. The evaluation script refuses to run otherwise (Phase 3).
- **Patient-level uncertainty.** The confidence intervals resample whole patient groups: 102 fractures come from only 47 groups.
- **Seeded comparisons.** Model changes are compared over 3 seeds on validation, with the selection rule fixed before the runs. Weighted loss, oversampling and focal loss all failed that test (Phase 4).

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
| 5 Model comparison | ✅ val / 🔨 test pending | [docs](docs/phase-5-models.md): 4 challengers × 3 seeds. **ConvNeXt-Tiny wins: val PR-AUC 0.709 → 0.814, false alarms at 90% recall 52 → 42 per 100**, gain holds without hardware images. Higher resolution (448 px) gave nothing |

## Quick start (Kaggle)

See [`docs/phase-0-setup.md`](docs/phase-0-setup.md) for the full guide and
[`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb) for the runner.

## Dataset citation

Abedeen, I., Rahman, M. A., Prottyasha, F. Z., et al. *FracAtlas: A Dataset for
Fracture Classification, Localization and Segmentation of Musculoskeletal
Radiographs.* Scientific Data 10, 521 (2023). Data licensed CC BY 4.0.

## License

Code: MIT. Dataset: CC BY 4.0 (not redistributed here).

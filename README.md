# Bone Fracture Detection from X-rays (FracAtlas)

> 🚧 Work in progress, built phase by phase. Results tables, Grad-CAM examples and
> the live demo link will be added only from real, logged runs.

Binary fracture classification (and later localization) on musculoskeletal
radiographs from the [FracAtlas](https://doi.org/10.6084/m9.figshare.22363012)
dataset: 4,083 X-rays, about 18% fractured.

**Not a medical device. For education and research only.**

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

## Quick start (Kaggle)

See [`docs/phase-0-setup.md`](docs/phase-0-setup.md) for the full guide and
[`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb) for the runner.

## Dataset citation

Abedeen, I., Rahman, M. A., Prottyasha, F. Z., et al. *FracAtlas: A Dataset for
Fracture Classification, Localization and Segmentation of Musculoskeletal
Radiographs.* Scientific Data 10, 521 (2023). Data licensed CC BY 4.0.

## License

Code: MIT. Dataset: CC BY 4.0 (not redistributed here).

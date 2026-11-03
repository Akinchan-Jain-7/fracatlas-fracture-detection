# Phase 0: Project setup

**Goal:** a repo where every later phase plugs into the same structure. Experiments
are defined by config files, runs are reproducible, the dataset is acquired and
verified by script (never by hand), and every push is linted and tested
automatically.

---

## 1. Kaggle workflow (one-time setup)

1. **Verify your phone number** on Kaggle (Settings → Phone verification).
   Without it, notebooks get no internet (needed for `git clone`) and no GPU.
2. **Create the dataset once** (so you never re-download it):
   - Download `FracAtlas.zip` (~338 MB) in your browser from
     <https://doi.org/10.6084/m9.figshare.22363012>.
   - Kaggle → *Datasets* → *New Dataset* → upload the zip. Title it `FracAtlas`,
     keep it **Private**. Kaggle extracts zip uploads automatically.
   - If the *New Dataset* dialog offers a link/URL option, you can paste
     `https://ndownloader.figshare.com/files/65518038` instead of uploading.
3. **Create the runner notebook:** *Code* → *New Notebook* → *File* → *Import
   Notebook* → upload `notebooks/kaggle_runner.ipynb` from this repo (or paste
   its GitHub URL).
4. In the notebook's right panel: *Settings* → **Internet On**, *Accelerator* →
   **GPU T4 x2** (or P100). *Add Input* → *Your Datasets* → `FracAtlas`.
5. *Run All*.

The repo must stay **public** for the notebook's `git clone` to work without a
token. (If you ever make it private, store a GitHub token in Kaggle *Add-ons →
Secrets* and clone with it; never paste tokens into notebook cells.)

**Daily loop from here on:** code changes land in GitHub → in Kaggle, re-run the
clone cell (it does `git pull`) → run the script cell for that phase → send back
the printed output.

### Why this split (repo = logic, notebook = runner)?
Notebooks are poor at version control (JSON diffs, hidden state, out-of-order
execution). Keeping logic in `.py` files means it is diffable, testable, importable
by the FastAPI app later, and runs the same on Kaggle, a laptop or a Docker
container. The notebook is a remote control.

---

## 2. What each file does

| Path | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, the **only** dependency list, ruff + pytest config. `pip install -e .` makes `fracdet` importable from anywhere. |
| `configs/base.yaml` | Seed, paths, dataset URL + MD5 + expected counts. Every future experiment config inherits from it. |
| `src/fracdet/config.py` | Loads YAML, resolves `_base_` inheritance, applies `key=value` overrides, saves the resolved config next to a run. |
| `src/fracdet/seed.py` | Seeds `random`, NumPy and torch; optional deterministic cuDNN; DataLoader worker seeding. |
| `src/fracdet/paths.py` | Finds the data on any machine: `FRACDET_DATA_DIR` env var beats the config; locates the dataset root by searching for `dataset.csv`. |
| `src/fracdet/logging_utils.py` | One log format for all scripts; console + optional log file; no duplicate handlers in notebooks. |
| `src/fracdet/data/download.py` | Streamed download, MD5 check, atomic rename, zip-slip-safe extraction, idempotent `prepare_dataset`. |
| `src/fracdet/data/verify.py` | Checks CSV ↔ image matching, duplicates, counts vs. the paper; reports the real layout. |
| `scripts/*.py` | Thin CLIs over the package: `download_data`, `verify_data`, `env_check`. |
| `tests/` | 30+ unit tests, including a miniature fake FracAtlas so tests never need the real data. |
| `.github/workflows/ci.yml` | On every push: install CPU torch, install the package, `ruff check`, `ruff format --check`, `pytest`. |
| `notebooks/kaggle_runner.ipynb` | Clone/pull → install → set data path → run scripts. |

---

## 3. Key decisions and trade-offs

- **`src/` layout + installable package.** Tests import the *installed* package,
  not whatever happens to be in the current folder, which catches packaging
  mistakes early. The FastAPI app (Phase 8) imports the same code, so training
  and inference preprocessing can't drift apart.
- **One dependency list with lower bounds only.** Kaggle and Colab ship their own
  CUDA builds of torch. Pinning `torch==2.x.y` would make pip download a different
  multi-GB build and possibly break CUDA. Trade-off: less exact reproducibility
  of library versions. That's why `env_check.py` output goes into every run
  report. Exact pins come in Phase 8 for the Docker image, where we control the
  whole environment.
- **Own config loader instead of Hydra/OmegaConf.** About 100 lines that I can
  explain fully, with no extra dependency. Switch if we ever need sweeps or
  multi-run launching.
- **MD5 for integrity, not security.** MD5 is broken against *deliberate*
  collisions, but here it only detects accidental corruption, and it's the
  checksum Figshare publishes. Using SHA-256 would mean computing our own
  reference hash, which proves nothing about the original.
- **Verification is discovery-based.** I (Claude) couldn't download the archive
  while writing this, so `verify_data.py` doesn't hard-code folder names. It
  searches, matches and *reports*. Phase 1 code will be written against what it
  prints on Kaggle.
- **ruff** replaces black + isort + flake8 with a single fast tool.

### A bug the tests caught
PyYAML implements YAML 1.1, where `3e-4` is **a string**, not a float (YAML 1.1
needs a dot: `3.0e-4`). An override like `--set train.lr=3e-4` would have
silently passed a string to the optimizer. `parse_value` now converts that
pattern. **In YAML files, always write `3.0e-4`.**

---

## 4. Likely interview questions

**Q: Why not just do everything in a Jupyter notebook?**
Notebooks hide state (cells run out of order), diff badly in git, can't be unit
tested or imported by a web service. I keep logic in a Python package with tests
and use the notebook only to trigger scripts on a GPU.

**Q: How do you make an experiment reproducible?**
Fixed seeds for all three RNGs (Python, NumPy, torch, incl. DataLoader workers);
config files instead of hard-coded values, with the resolved config saved per
run; dataset verified by checksum and counts; data splits saved to files (Phase 1);
environment printed per run. Caveat: GPU kernels are non-deterministic by
default. Bit-exact reruns need deterministic cuDNN and cost speed, so I use it for
debugging, not every run.

**Q: Why is the download written to a `.part` file first?**
If the process dies mid-download, a half file named `FracAtlas.zip` would look
complete to the "skip if exists" check. Downloading to `.part`, verifying, then
`os.replace` (atomic on one filesystem) means the final name only ever points to
a verified file.

**Q: What is zip slip?**
A zip entry named like `../../.bashrc` makes naive extraction write outside the
target folder. I resolve every entry's target path and refuse anything not inside
the destination.

**Q: What does CI give a solo project?**
Every push proves the code installs cleanly on a fresh machine, passes lint and
passes tests. Without it, "works on my Kaggle session" is the only evidence.

**Q: Why install CPU torch first in CI?**
`pip install -e .` would otherwise pull the default CUDA wheel (~2.5 GB) onto a
runner with no GPU. Pre-installing the CPU wheel satisfies `torch>=2.1`, so pip
skips it.

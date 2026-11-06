"""Fracture Check on a free Hugging Face Space (Gradio SDK).

Hugging Face runs this file with `python app.py` and shows whatever listens on
port 7860. Instead of a Gradio interface, it starts the project's own FastAPI app,
so the page is exactly the one you run locally or in Docker.

At every start (and restart) it:
  1. downloads the latest code from the GitHub repo (main branch),
  2. downloads the two model files from the Hugging Face model repo,
  3. starts the FastAPI app, which verifies both models before answering.

To deploy a new version: push to GitHub, then "Restart" the Space (Settings).
New Python dependencies go in requirements.txt here (that needs a rebuild).
"""

from __future__ import annotations

# Free Gradio Spaces run on ZeroGPU hardware, which stops any app that does not
# report at least one @spaces.GPU function at start-up. Importing `spaces` first
# (before torch), declaring one, and reporting it (report_zerogpu_ready below)
# satisfies that check. The app itself runs its models on the CPU, exactly as
# tested locally, and never calls this function.
try:
    import spaces
except ImportError:  # running somewhere other than a ZeroGPU Space
    spaces = None

if spaces is not None:

    @spaces.GPU
    def zerogpu_declaration() -> None:
        """Declared for ZeroGPU's start-up check; never called."""


import importlib.util
import io
import logging
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

CODE_REPO = os.environ.get("CODE_REPO", "Akinchan-Jain-7/fracatlas-fracture-detection")
CODE_REF = os.environ.get("CODE_REF", "main")
MODEL_REPO = os.environ.get("MODEL_REPO", "akinchanJain7/fracture-detection-model")
MODEL_FILES = ("classifier_convnext_tiny_s44.pt", "detector_s42_best.pt")
WORK = Path(os.environ.get("WORK_DIR", "/tmp/fracture-check"))
PORT = int(os.environ.get("PORT", "7860"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("space")


def report_zerogpu_ready() -> None:
    """Send ZeroGPU its start-up report.

    `spaces` normally sends it when a Gradio app calls launch(). This Space serves
    FastAPI instead, so nothing calls launch(); run the same start-up step here.
    """
    if spaces is None:
        return
    zero = importlib.import_module("spaces.zero")
    startup = getattr(zero, "startup", None)  # defined only on ZeroGPU hardware
    if startup is not None:
        startup()
        log.info("ZeroGPU start-up report sent")


def fetch_code() -> Path:
    """The GitHub branch as a folder (a tarball, so no git is needed)."""
    url = f"https://github.com/{CODE_REPO}/archive/refs/heads/{CODE_REF}.tar.gz"
    log.info("downloading code: %s", url)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    target = WORK / "code"
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        root = tar.getmembers()[0].name.split("/")[0]
        tar.extractall(target, filter="data")
    return target / root


def fetch_models() -> Path:
    from huggingface_hub import hf_hub_download

    target = WORK / "models"
    for name in MODEL_FILES:
        log.info("downloading model: %s/%s", MODEL_REPO, name)
        hf_hub_download(MODEL_REPO, name, local_dir=target)
    return target


def main() -> None:
    report_zerogpu_ready()
    code = fetch_code()
    models = fetch_models()
    # Use the downloaded code's own package (src/fracdet), so code and config match.
    sys.path.insert(0, str(code / "src"))
    os.environ["FRACDET_MODELS_DIR"] = str(models)
    os.environ["FRACDET_SERVE_CONFIG"] = str(code / "configs" / "serve.yaml")

    # Load app/main.py by path: this file is itself named "app", so "import app.main"
    # would find this file instead of the project's app package.
    spec = importlib.util.spec_from_file_location("fracture_check_app", code / "app" / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import uvicorn

    log.info("starting Fracture Check on port %d", PORT)
    uvicorn.run(module.app, host="0.0.0.0", port=PORT, workers=1)


if __name__ == "__main__":
    main()

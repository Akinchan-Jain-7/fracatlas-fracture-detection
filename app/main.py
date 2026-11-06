"""Phase 8 web server: the FracturePipeline behind three HTTP endpoints + the web page.

    GET  /               the web page (app/static/index.html)
    GET  /api/health     is the server up and are the models loaded?
    GET  /api/info       threshold, box cut-off, test-set model card (for the page)
    POST /api/predict    multipart upload "file" -> probability, heatmap, boxes

Run locally (from the repo root):
    uvicorn app.main:app --port 8000
    then open http://localhost:8000

Settings (environment variables):
    FRACDET_MODELS_DIR    folder with the two .pt files (default: ./models)
    FRACDET_SERVE_CONFIG  serve config (default: configs/serve.yaml)
    FRACDET_TORCH_THREADS CPU threads for PyTorch (default: all cores)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import fracdet
from fracdet.serve.pipeline import (
    BadImageError,
    FracturePipeline,
    ModelFileError,
    load_serve_config,
)

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("fracdet.app")


def _load_pipeline() -> FracturePipeline:
    cfg = load_serve_config(os.environ.get("FRACDET_SERVE_CONFIG"))
    models_dir = Path(os.environ.get("FRACDET_MODELS_DIR", "models"))
    threads = os.environ.get("FRACDET_TORCH_THREADS")
    if threads:
        torch.set_num_threads(int(threads))
    log.info("loading models from %s", models_dir.resolve())
    return FracturePipeline.from_dir(models_dir, cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models ONCE when the server starts (several seconds), not per request.
    # If a file is missing or is not the evaluated model, the server still starts so
    # /api/health can say what is wrong, but /api/predict refuses to answer.
    try:
        app.state.pipeline = _load_pipeline()
        app.state.load_error = None
    except ModelFileError as e:
        log.error("models not loaded: %s", e)
        app.state.pipeline, app.state.load_error = None, str(e)
    yield


app = FastAPI(title="Fracture Check", version=fracdet.__version__, lifespan=lifespan)


def _pipeline(request: Request) -> FracturePipeline:
    p = request.app.state.pipeline
    if p is None:
        raise HTTPException(503, f"Models are not loaded: {request.app.state.load_error}")
    return p


@app.get("/api/health")
def health(request: Request):
    ok = request.app.state.pipeline is not None
    body = {"status": "ok" if ok else "models_missing", "error": request.app.state.load_error}
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/api/info")
def info(request: Request):
    return _pipeline(request).info()


@app.post("/api/predict")
def predict(request: Request, file: Annotated[UploadFile, File()]):
    # A plain "def" (not "async def"): FastAPI runs it in a worker thread, so a
    # 3-4 s model call does not freeze the server for other requests.
    pipeline = _pipeline(request)
    limits = pipeline.cfg["limits"]
    if file.content_type not in limits["accepted_types"]:
        raise HTTPException(415, "Upload a JPEG or PNG image of an X-ray.")
    max_bytes = int(limits["max_upload_mb"] * 1024 * 1024)
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"The image is larger than {limits['max_upload_mb']} MB.")
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    try:
        result = pipeline.predict(data)
    except BadImageError as e:
        raise HTTPException(400, str(e)) from e
    log.info("predict %s: p=%.3f flagged=%s boxes=%d %s", file.filename, result["probability"],
             result["flagged"], len(result["boxes"]), result["timing_ms"])  # fmt: skip
    return result


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

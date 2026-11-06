# Fracture Check web app (Phase 8), CPU only.
#
# Build:   docker build -t fracture-check .
# Run:     docker run --rm -p 8000:7860 -v "$PWD/models:/app/models:ro" fracture-check
#          (Windows PowerShell: -v "${PWD}\models:/app/models:ro")
# Open:    http://localhost:8000
#
# The two model files (~280 MB) are NOT baked into the image: they are mounted
# from ./models at run time, so the image stays small and a new model never
# needs a rebuild. The app checks each file's checksum / provenance at start-up.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 1. PyTorch first, from the CPU-only index: the default wheel bundles CUDA (~2.5 GB)
#    that a CPU server never uses. Its own layer, so code changes don't re-download it.
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Run as a normal user, not root (also what Hugging Face Spaces expects: uid 1000).
RUN useradd --create-home --uid 1000 app
WORKDIR /app

# 3. The package and its dependencies. Copy only what the install needs, so editing
#    the web page later does not invalidate this (slow) layer.
COPY --chown=app pyproject.toml README.md ./
COPY --chown=app src ./src
RUN pip install -e ".[serve]"

# 4. The app itself and its settings. Results, data, notebooks, tests: not needed.
COPY --chown=app configs/serve.yaml ./configs/serve.yaml
COPY --chown=app app ./app
RUN mkdir -p /app/models && chown app /app/models

USER app
ENV FRACDET_MODELS_DIR=/app/models \
    FRACDET_SERVE_CONFIG=/app/configs/serve.yaml \
    PORT=7860
EXPOSE 7860

# Docker (and most hosts) mark the container unhealthy if this fails, e.g. models missing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"

# One worker process: each one would load its own copy of both models (~1.5 GB RAM).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]

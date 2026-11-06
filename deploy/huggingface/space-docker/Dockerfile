# Fracture Check on Hugging Face Spaces (Docker SDK, CPU basic).
#
# This file and README.md are the whole Space repository. Everything else is
# fetched while the Space builds:
#   - the code: the latest commit of the GitHub repo (CODE_REPO @ CODE_REF)
#   - the models: the Hugging Face model repo (MODEL_REPO)
# So the GitHub repo stays the single source of truth. To deploy a new version:
# push to GitHub, then Space settings -> "Factory rebuild" (or Restart).

FROM python:3.11-slim

ARG CODE_REPO=Akinchan-Jain-7/fracatlas-fracture-detection
ARG CODE_REF=main
ARG MODEL_REPO=akinchanJain7/fracture-detection-model

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 1. CPU-only PyTorch (the default wheel would add ~2.5 GB of unused CUDA).
RUN pip install --timeout 120 --retries 10 torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 2. The model files. ADD of the repo's file list makes Docker notice when the
#    models change (and only then re-download the ~280 MB).
RUN pip install --timeout 120 --retries 10 "huggingface_hub>=0.24"
ADD https://huggingface.co/api/models/${MODEL_REPO}/tree/main /tmp/model_files.json
RUN python -c "from huggingface_hub import hf_hub_download as get; \
[get('${MODEL_REPO}', f, local_dir='/app/models') for f in ('classifier_convnext_tiny_s44.pt', 'detector_s42_best.pt')]" \
    && rm -rf /app/models/.cache

# 3. The code: a snapshot of the GitHub branch. ADD re-downloads it on every build,
#    so a new commit invalidates the cache from here on (PyTorch and models stay cached).
WORKDIR /app
ADD https://github.com/${CODE_REPO}/archive/refs/heads/${CODE_REF}.tar.gz /tmp/code.tar.gz
RUN tar -xzf /tmp/code.tar.gz --strip-components=1 -C /app && rm /tmp/code.tar.gz
RUN pip install --timeout 120 --retries 10 -e ".[serve]"

# 4. Check the models now: a wrong or corrupted file fails the BUILD (you see it in
#    the build log) instead of producing a live app that can't answer.
RUN python -c "from fracdet.serve.pipeline import FracturePipeline; FracturePipeline.from_dir('/app/models'); print('models verified')"

# 5. Run as a normal user (Spaces run containers as uid 1000). The app only reads
#    the code and models, so they can stay owned by root.
RUN useradd --create-home --uid 1000 app
USER app
ENV FRACDET_MODELS_DIR=/app/models \
    FRACDET_SERVE_CONFIG=/app/configs/serve.yaml \
    PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]

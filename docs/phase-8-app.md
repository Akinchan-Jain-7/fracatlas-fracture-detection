# Phase 8: The web app (Fracture Check) and Docker

**Goal:** turn the evaluated models into something a person can use: upload an
X-ray, get the fracture score, see where the model looked and where the break
most likely is, and read what to do about a fracture. Then package it with
Docker so it runs the same way on any computer or host.

Nothing here changes a model, a threshold or a result: the app loads exactly
the Phase 5 classifier and the Phase 7 detector, and checks that it has.

## 1. How the pieces fit

```
 Browser                              Server (one Python process)
 ┌──────────────────────────┐        ┌───────────────────────────────────────────┐
 │ index.html + app.js      │  POST  │ app/main.py (FastAPI)                     │
 │  1. user drops an X-ray  │ ─────► │  /api/predict: checks type and size       │
 │                          │ image  │        │                                  │
 │                          │        │        ▼                                  │
 │                          │        │ src/fracdet/serve/pipeline.py             │
 │                          │        │  FracturePipeline.predict(bytes)          │
 │                          │        │   1. classifier  -> probability, flag     │
 │                          │        │   2. Grad-CAM    -> heatmap PNG           │
 │                          │  JSON  │   3. detector    -> boxes (if flagged)    │
 │  2. draws score, heatmap │ ◄───── │                                           │
 │     and boxes            │        │ models loaded ONCE at start-up            │
 └──────────────────────────┘        └───────────────────────────────────────────┘
```

| File | What it does |
|---|---|
| `configs/serve.yaml` | Which model files, the threshold, the box cut-off, upload limits, the test-set numbers shown on the page. Every value is checked against its source in `results/` by `tests/test_serve.py`. |
| `src/fracdet/serve/pipeline.py` | `FracturePipeline`: loads and verifies both models, `predict(bytes)` returns everything the page needs. No web code. |
| `app/main.py` | FastAPI server: `GET /`, `GET /api/health`, `GET /api/info`, `POST /api/predict`. |
| `app/static/` | The page: `index.html`, `styles.css`, `app.js` (upload and drawing), `care.js` (care guide content), `fonts/` (served locally). |
| `scripts/choose_box_cutoff.py` | Chooses the extra-box cut-off on validation. |
| `Dockerfile`, `.dockerignore` | The container image. |

**Why the pipeline is separate from the server.** The pipeline is plain Python:
it can be tested without starting a server, called from a notebook, and moved to
another web framework without touching the model code. The server only deals
with HTTP: reading the upload, refusing bad input, turning errors into status codes.

## 2. What the app decides, and why

- **Flag:** probability ≥ 0.0968, the Phase 5 threshold set for 90% recall on
  validation. On test it caught 94 of 102 fractures; about 4 in 10 flagged
  X-rays had a fracture. The page says both.
- **Heatmap:** Grad-CAM at the classifier's last stage, exactly as in Phase 6,
  sent as a transparent PNG and laid over the X-ray. Labelled "where the model
  looked", not "where the fracture is": Phase 6 measured that its peak is on
  the fracture in about 1 in 5 X-rays.
- **Boxes only on flagged X-rays.** On its own the detector put a box scoring
  ≥ 0.5 on about 1 normal X-ray in 5 (Phase 7). The classifier decides
  *whether*, the detector *where*.
- **Which boxes:** the top box always (solid if it scores ≥ the cut-off, dashed
  if not), plus up to 3 more that score ≥ **0.731**. That cut-off was chosen on
  *validation*, never test: it is the lowest score at which normal validation
  X-rays get at most 0.1 boxes each (`scripts/choose_box_cutoff.py`).
- **Same preprocessing as training.** Each model gets its own decode of the
  upload through the same `DraftGrayscale` + transforms used in training and
  evaluation; a test checks the app's probability equals the evaluation code's.
- **Refuses the wrong model.** At start-up the app checks the detector's
  SHA-256 and both models' training round, git commit and data-split checksum
  against `serve.yaml`. A wrong or corrupted file means the server starts but
  `/api/health` reports the problem and `/api/predict` answers 503.

## 3. The care guide

A section of the page explains what to do for a suspected or confirmed
fracture, with a body-part selector (any bone, hand and wrist, arm and
shoulder, leg and ankle, hip): emergency signs, first aid before seeing a
doctor, how fractures are treated, healing times, do's and don'ts in a cast,
and warning signs during recovery. It is general information summarised from
NHS and Mayo Clinic pages (linked on the page and in `app/static/care.js`),
shown the same way whatever the model says, and never as advice for a specific
X-ray. Emergency numbers are listed for India, the EU, the UK and the US.

## 4. Running it

**Model files.** Put both in `models/` at the repo root (git ignores `*.pt`):

| File | Where from |
|---|---|
| `models/classifier_convnext_tiny_s44.pt` | Kaggle `notebook34061d5c16` v2 output: `outputs/runs/20261104-110146_p5_convnext_tiny_s44/best.pt` (111.36 MB, SHA-256 `f54efa24…ea2784`, in `configs/serve.yaml`) |
| `models/detector_s42_best.pt` | Kaggle `notebook2ac59126bb` v8 output (SHA-256 in `results/phase7_detect/README.md`) |

**Without Docker** (good for development):

```bash
pip install -e ".[serve]"
uvicorn app.main:app --port 8000          # add --reload to restart on code changes
# open http://localhost:8000
```

**With Docker:**

```bash
docker build -t fracture-check .
docker run --rm -p 8000:7860 -v "$PWD/models:/app/models:ro" fracture-check
# Windows PowerShell: -v "${PWD}\models:/app/models:ro"
# open http://localhost:8000
```

`-p 8000:7860` connects port 8000 on your computer to port 7860 inside the
container. `-v ...:/app/models:ro` lets the container read your `models` folder
without copying the files into the image.

**API from the command line:**

```bash
curl localhost:8000/api/health
curl -F "file=@some_xray.jpg;type=image/jpeg" localhost:8000/api/predict
```

FastAPI also generates interactive API docs at `http://localhost:8000/docs`.

## 5. The Docker image

| Step in the Dockerfile | Why |
|---|---|
| `FROM python:3.11-slim` | A small Linux with Python; nothing else. |
| CPU-only PyTorch from its own index, in its own layer | The default PyTorch wheel bundles ~2.5 GB of CUDA a CPU server never uses. A separate layer means code edits don't re-download it. |
| Non-root user `app` (uid 1000) | A process that is compromised can do less; Hugging Face Spaces also expects uid 1000. |
| Copy `pyproject.toml` + `src` and install, *then* copy `app/` | Docker caches layers in order: editing the page doesn't redo the slow install. |
| Models mounted, not copied | Smaller image, no rebuild for a new model, and model files stay out of the image registry. |
| `HEALTHCHECK` on `/api/health` | Docker and hosts can see when the app is up but has no models. |
| One worker | Each worker process would load its own copy of both models (~1.5 GB RAM). |

CI builds the image on every push and starts it without models: the page must
load, `/api/health` must report the models missing, and the server must run as
the non-root user.

## 6. Performance (CPU, measured in development)

About 0.2–0.4 s for the classifier and heatmap and 3–4 s for the detector per
X-ray on a 4-core CPU; the first request after start-up is slower. The page
shows a scanning animation while it waits. Uploads are capped at 15 MB, JPEG or PNG.

## 7. Deployment (Hugging Face)

```
GitHub repo (code)  ──tarball of main──►  Space akinchanJain7/fracture-check  ──► public app
HF model repo akinchanJain7/fracture-detection-model  ──two .pt files──┘
```

- **Model repo** `akinchanJain7/fracture-detection-model` (public): the two `.pt`
  files plus a model card (`deploy/huggingface/models/README.md`).
- **Space** `akinchanJain7/fracture-check` (free, CPU basic). Docker Spaces need a
  paid plan, so the Space uses the free Gradio SDK, which simply runs `app.py` and
  shows whatever listens on port 7860. Our `app.py` (`deploy/huggingface/space/`)
  downloads the latest GitHub code and the two models at start-up, then serves the
  same FastAPI app as locally. The app verifies both models before answering.
  `requirements.txt` installs CPU PyTorch and the project's dependencies.
- **Updating:** push to GitHub, then *Restart* the Space: it fetches the new code at
  start-up. A change to dependencies needs *Factory rebuild* (reinstalls requirements).
- `deploy/huggingface/space-docker/` keeps the Docker-Space version (Dockerfile that
  verifies the models at build time), for a paid Space or any Docker host.
- Why not push the whole repo to the Space: Hugging Face repos store binary files
  (figures, fonts) through LFS/Xet, and the Space would then hold a second copy
  of the code. Fetching from GitHub keeps one source of truth.

## 8. Interview questions

**Q: Why FastAPI?**
It is a small Python web framework, so the models stay in the same language
and process as the server. It validates inputs, turns exceptions into HTTP
errors, and documents the API automatically at `/docs`.

**Q: Why load the models at start-up and not per request?**
Loading takes seconds and ~1 GB of memory. Done once, each request only runs
inference.

**Q: Why is `/api/predict` a plain `def` and not `async def`?**
Model inference is CPU-bound and blocking. FastAPI runs plain `def` endpoints in
a worker thread, so a 4-second prediction doesn't freeze the server; an `async`
function would block the event loop for that time.

**Q: How do you know the deployed model is the evaluated one?**
The detector file's SHA-256 is recorded in the repo, and both checkpoints carry
their training round, git commit and data-split checksum. The app compares all
of these with `serve.yaml` at start-up, and a test compares `serve.yaml` with
the committed results.

**Q: Why not show the detector's boxes on every X-ray?**
Phase 7 measured that it puts a confident box on about 1 normal X-ray in 5. The
classifier was selected and tested as the screening decision; the detector only
says where, once the classifier has flagged the X-ray.

**Q: What is the difference between a Docker image and a container?**
The image is the packaged, read-only recipe result: OS, Python, libraries and
code. A container is one running instance of it, with its own process and
filesystem changes. You build an image once and can run many containers from it.

**Q: Isn't a care guide risky in a medical demo?**
It is general information from NHS and Mayo Clinic pages, the same for every
user, with sources linked, and it says it doesn't replace the treating doctor.
The page also says clearly, in several places, that the app is not a diagnosis
and not a medical device.

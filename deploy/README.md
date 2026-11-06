# Deployment files

The app itself lives in `app/` and `src/fracdet/serve/`; this folder holds what each
hosting target needs on top of it. How it all fits together, and why, is in
[docs/phase-8-app.md](../docs/phase-8-app.md#7-deployment-hugging-face).

| Folder | What it is | Where it is used |
|---|---|---|
| [`huggingface/models/`](huggingface/models/) | Model card for the two `.pt` files | [akinchanJain7/fracture-detection-model](https://huggingface.co/akinchanJain7/fracture-detection-model) |
| [`huggingface/space/`](huggingface/space/) | The live Space: `app.py` downloads this repo and the models at start-up and serves the FastAPI app; `requirements.txt`; Space card | [akinchanJain7/fracture-check](https://huggingface.co/spaces/akinchanJain7/fracture-check) (free, ZeroGPU hardware, inference on CPU) |
| [`huggingface/space-docker/`](huggingface/space-docker/) | The same app as a Docker Space, verifying the models at build time | A paid Space or any Docker host |

The root [`Dockerfile`](../Dockerfile) is the general container image, built and
smoke-tested in CI on every push.

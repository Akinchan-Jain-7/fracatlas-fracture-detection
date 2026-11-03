"""Seeding for reproducible runs.

Three independent random number generators matter in a PyTorch project:
Python's ``random``, NumPy, and torch (CPU and every CUDA device). Seeding only
one of them still leaves splits, augmentations or weight init non-repeatable.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) torch.

    Args:
        seed: the seed value, normally ``cfg["seed"]``.
        deterministic: also force deterministic cuDNN kernels. Gives bit-exact
            reruns on the same GPU at a speed cost (often 10-30%). Use it to debug
            a result; leave it off for normal training.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seeds legacy np.random calls in libraries
    # Only affects subprocesses started after this point (e.g. DataLoader workers
    # on spawn-based platforms); the current interpreter's hash seed is fixed at startup.
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:  # lets data-only scripts and CI run without torch
        return

    torch.manual_seed(seed)  # also seeds all CUDA devices
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, reproducible NumPy/random seed.

    torch seeds each worker's torch RNG automatically, but not NumPy or
    ``random``. Without this, augmentation libraries that use NumPy can produce
    identical "random" augmentations in every worker.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)

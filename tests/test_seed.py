import random

import numpy as np
import pytest

from fracdet.seed import seed_everything


def _draw() -> tuple[float, float]:
    return random.random(), float(np.random.rand())  # noqa: NPY002


def test_same_seed_same_numbers():
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    assert _draw() == first


def test_different_seed_different_numbers():
    seed_everything(1)
    a = _draw()
    seed_everything(2)
    assert _draw() != a


def test_torch_is_seeded_too():
    torch = pytest.importorskip("torch")
    seed_everything(7)
    a = torch.rand(3)
    seed_everything(7)
    assert torch.equal(a, torch.rand(3))

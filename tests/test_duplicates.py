import numpy as np
import pytest
from PIL import Image

from fracdet.data.duplicates import (
    compute_hashes,
    dhash,
    exact_duplicate_pairs,
    group_pairs,
    hamming_matrix,
    near_duplicate_pairs,
    nearest_neighbor_distance,
)
from fracdet.data.integrity import find_corrupt
from tests.conftest import write_xray


@pytest.mark.parametrize("hash_size", [8, 16])
def test_resized_recompressed_copy_is_near_identical(tmp_path, hash_size):
    write_xray(tmp_path / "a.jpg", seed=1, size=(400, 500))
    with Image.open(tmp_path / "a.jpg") as im:
        im.resize((200, 250)).save(tmp_path / "a_small.jpg", quality=60)
    write_xray(tmp_path / "b.jpg", seed=2, size=(400, 500))
    h = np.stack([dhash(tmp_path / n, hash_size) for n in ["a.jpg", "a_small.jpg", "b.jpg"]])
    assert h.shape == (3, hash_size * hash_size // 8)
    d = hamming_matrix(h)
    n_bits = hash_size * hash_size
    assert d[0, 1] <= 0.08 * n_bits  # same picture, re-encoded and resized
    assert d[0, 2] >= 0.25 * n_bits  # unrelated picture


def test_hamming_matches_python_popcount():
    rng = np.random.default_rng(0)
    h = rng.integers(0, 256, size=(15, 32), dtype=np.uint8)
    d = hamming_matrix(h)
    as_int = [int.from_bytes(row.tobytes(), "big") for row in h]
    for i in range(15):
        for j in range(15):
            assert d[i, j] == bin(as_int[i] ^ as_int[j]).count("1")


def test_pairs_and_nearest_neighbor():
    h = np.array([[0b0], [0b1], [0b11111111], [0b11111110]], dtype=np.uint8)
    assert near_duplicate_pairs(h, threshold=1, chunk=3) == [(0, 1, 1), (2, 3, 1)]
    assert nearest_neighbor_distance(h, chunk=3).tolist() == [1, 1, 1, 1]


def test_groups_are_transitive():
    groups = group_pairs(6, [(0, 1, 2), (1, 2, 3), (4, 5, 0)])
    assert groups.tolist() == [0, 0, 0, 3, 4, 4]


def test_exact_duplicate_pairs():
    assert exact_duplicate_pairs(["a", "b", "a", "c", "a", "b"]) == [(0, 2), (0, 4), (1, 5)]


def test_compute_hashes_parallel_matches_serial(tmp_path):
    paths = []
    for i in range(6):
        write_xray(tmp_path / f"{i}.jpg", seed=i)
        paths.append(tmp_path / f"{i}.jpg")
    parallel = compute_hashes(paths, workers=3)
    assert np.array_equal(parallel, np.stack([dhash(p) for p in paths]))


def test_truncated_jpeg_is_detected_but_still_hashable(tmp_path):
    write_xray(tmp_path / "ok.jpg", seed=3, size=(300, 300))
    data = (tmp_path / "ok.jpg").read_bytes()
    (tmp_path / "cut.jpg").write_bytes(data[: len(data) - 200])  # drop the tail
    corrupt = find_corrupt([tmp_path / "ok.jpg", tmp_path / "cut.jpg"])
    assert list(corrupt) == [str(tmp_path / "cut.jpg")]
    assert compute_hashes([tmp_path / "cut.jpg"]).shape == (1, 32)

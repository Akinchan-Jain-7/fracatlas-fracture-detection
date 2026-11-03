"""Exact and near-duplicate image detection.

Why it matters: if the same X-ray (or a re-saved, resized copy) lands in both
train and test, the test score measures memorisation, not generalisation.

- Exact duplicates: identical bytes, found with MD5.
- Near-duplicates: same picture after re-encoding or resizing, found with a
  perceptual "difference hash" (dHash). Shrink the image to (s+1) x s grayscale
  and record, for each pixel, whether it is brighter than its right neighbour:
  s*s bits describing the coarse structure, robust to JPEG noise and rescaling.
  Two images are near-duplicates when their hashes differ in few bits (small
  Hamming distance).

Hash size matters for X-rays: at the textbook 8x8 (64 bits) most radiographs
(dark background, bright bone) collide. On FracAtlas, 221 images had a 64-bit
"exact" match while only 27 are true byte duplicates. We use 16x16 = 256 bits.

dHash cannot recognise the same patient in a *different view* (frontal vs
lateral look different). That is handled separately, see ``split.py``.

Hashes are stored as uint8 arrays of shape (n, s*s/8), so any hash size works
with the same distance code.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from fracdet.data.download import md5_of_file
from fracdet.data.integrity import allow_truncated_images

DEFAULT_HASH_SIZE = 16


def dhash(path: str | Path, hash_size: int = DEFAULT_HASH_SIZE) -> np.ndarray:
    """Difference hash of an image file, as ``hash_size**2 / 8`` packed bytes."""
    with Image.open(path) as im:
        # JPEG "draft" mode decodes at 1/2, 1/4 or 1/8 scale in the DCT domain:
        # a 2880px X-ray decodes ~8x faster, and we only need a few pixels.
        im.draft("L", (hash_size * 8, hash_size * 8))
        small = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        px = np.asarray(small, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return np.packbits(bits)


def compute_hashes(
    paths: Sequence[str | Path], hash_size: int = DEFAULT_HASH_SIZE, workers: int = 8
) -> np.ndarray:
    """Hashes for many files in parallel -> uint8 array of shape (n, bytes).

    PIL releases the GIL while decoding, so threads give a real speed-up.
    Truncated files are hashed too (they may duplicate an intact image); they
    are excluded from training separately, via ``integrity.find_corrupt``.
    """
    with allow_truncated_images(), ThreadPoolExecutor(max_workers=workers) as pool:
        return np.stack(list(pool.map(lambda p: dhash(p, hash_size), paths)))


def compute_md5s(paths: Sequence[str | Path], workers: int = 8) -> list[str]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(md5_of_file, paths))


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x)
    return np.unpackbits(x[..., None], axis=-1).sum(-1).astype(np.uint8)


def hamming_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Pairwise Hamming distances between two hash arrays of shape (n, bytes)."""
    b = a if b is None else b
    return _popcount(a[:, None, :] ^ b[None, :, :]).sum(axis=-1, dtype=np.int32)


def nearest_neighbor_distance(hashes: np.ndarray, chunk: int = 256) -> np.ndarray:
    """For each hash, the distance to its closest *other* hash. Chunked so the
    full n x n x bytes tensor never sits in memory."""
    n = len(hashes)
    out = np.empty(n, dtype=np.int32)
    for start in range(0, n, chunk):
        d = hamming_matrix(hashes[start : start + chunk], hashes)
        rows = np.arange(d.shape[0])
        d[rows, start + rows] = np.iinfo(np.int32).max  # ignore self-distance
        out[start : start + chunk] = d.min(axis=1)
    return out


def near_duplicate_pairs(
    hashes: np.ndarray, threshold: int, chunk: int = 256
) -> list[tuple[int, int, int]]:
    """All index pairs (i, j, distance) with i < j and distance <= threshold."""
    pairs: list[tuple[int, int, int]] = []
    for start in range(0, len(hashes), chunk):
        d = hamming_matrix(hashes[start : start + chunk], hashes)
        ii, jj = np.nonzero(d <= threshold)
        for i, j in zip(ii + start, jj, strict=True):
            if i < j:
                pairs.append((int(i), int(j), int(d[i - start, j])))
    return pairs


class UnionFind:
    """Merge items into groups: if A~B and B~C then A, B, C share one group.

    Pairs are edges; groups are connected components. Path halving keeps
    ``find`` close to O(1).
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> np.ndarray:
        """Group id per item: the smallest index in its component."""
        return np.array([self.find(i) for i in range(len(self.parent))])


def group_pairs(n: int, pairs: Sequence[Sequence[int]]) -> np.ndarray:
    """Connected components over index pairs; extra tuple fields are ignored."""
    uf = UnionFind(n)
    for i, j, *_ in pairs:
        uf.union(i, j)
    return uf.groups()


def exact_duplicate_pairs(md5s: Sequence[str]) -> list[tuple[int, int]]:
    """Pairs (first, other) for every file whose bytes repeat an earlier one."""
    first: dict[str, int] = {}
    pairs: list[tuple[int, int]] = []
    for i, digest in enumerate(md5s):
        if digest in first:
            pairs.append((first[digest], i))
        else:
            first[digest] = i
    return pairs

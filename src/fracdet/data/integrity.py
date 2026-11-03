"""Detect image files that do not fully decode.

FracAtlas v7 contains 59 truncated JPEGs (IDs 4028-4347, all non-fractured).
Pillow can still open them with ``LOAD_TRUNCATED_IMAGES``, but the missing data
is filled with flat grey rows at the bottom. A grey bar that appears only on
negatives is a shortcut a CNN can learn instead of fractures, so these files
are excluded from all splits (see ``split.py`` and docs/phase-1-data.md).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile


@contextlib.contextmanager
def allow_truncated_images(enabled: bool = True) -> Iterator[None]:
    """Temporarily set Pillow's process-wide truncated-image flag, then restore it."""
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = enabled
    try:
        yield
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def decode_error(path: str | Path) -> str | None:
    """Return the decoder error message, or None if the image decodes fully."""
    try:
        with Image.open(path) as im:
            im.load()
    except OSError as exc:
        return str(exc)
    return None


def find_corrupt(paths: Sequence[str | Path], workers: int = 8) -> dict[str, str]:
    """Map path -> error for every file that fails a strict full decode."""
    with allow_truncated_images(False), ThreadPoolExecutor(max_workers=workers) as pool:
        errors = list(pool.map(decode_error, paths))
    return {str(p): e for p, e in zip(paths, errors, strict=True) if e is not None}

"""Download and extract FracAtlas safely and idempotently.

Design goals:
- Idempotent: re-running does nothing if the data is already there, so it is safe
  at the top of every notebook.
- Integrity: the archive's MD5 must match the checksum Figshare publishes. A
  truncated or corrupted download fails loudly instead of producing a dataset
  with silently missing images.
- Atomic: download to ``*.part`` and rename only after the checksum passes, so
  an interrupted run never leaves a file that looks complete.
- Safe extraction: reject zip entries that would write outside the target
  folder ("zip slip"), since archives are untrusted input.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

from fracdet.config import Config
from fracdet.logging_utils import get_logger
from fracdet.paths import find_dataset_root

logger = get_logger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MiB


class ChecksumError(RuntimeError):
    pass


def md5_of_file(path: str | Path) -> str:
    """MD5 of a file, read in chunks so a 340 MB archive never sits in RAM at once."""
    digest = hashlib.md5()  # integrity check against Figshare's published hash, not security
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: str | Path, expected_md5: str, timeout: float = 60.0) -> Path:
    """Stream ``url`` to ``dest`` and verify its MD5 before making it visible."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")

    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        with (
            tmp.open("wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                bar.update(len(chunk))

    actual = md5_of_file(tmp)
    if actual != expected_md5:
        tmp.unlink(missing_ok=True)
        raise ChecksumError(f"MD5 mismatch for {dest.name}: expected {expected_md5}, got {actual}")
    os.replace(tmp, dest)  # atomic rename on the same filesystem
    logger.info("Checksum OK (%s)", actual)
    return dest


def safe_extract(zip_path: str | Path, dest_dir: str | Path) -> None:
    """Extract a zip, refusing any entry that resolves outside ``dest_dir``."""
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest_dir / member).resolve()
            if not target.is_relative_to(dest_dir):
                raise ValueError(f"Unsafe path in archive, refusing to extract: {member}")
        zf.extractall(dest_dir)


def prepare_dataset(cfg: Config, data_dir: str | Path) -> Path:
    """Ensure FracAtlas is extracted under ``data_dir``; return the dataset root."""
    data_dir = Path(data_dir)
    existing = find_dataset_root(data_dir)
    if existing is not None:
        logger.info("Dataset already present at %s, skipping download", existing)
        return existing

    data_cfg = cfg["data"]
    archive = data_dir / data_cfg["archive_name"]
    if archive.is_file() and md5_of_file(archive) == data_cfg["md5"]:
        logger.info("Verified archive already downloaded: %s", archive)
    else:
        download_file(data_cfg["source_url"], archive, data_cfg["md5"])

    logger.info("Extracting %s", archive)
    safe_extract(archive, data_dir)
    root = find_dataset_root(data_dir)
    if root is None:
        raise RuntimeError(f"Extraction finished but no dataset.csv found under {data_dir}")
    logger.info("Dataset ready at %s", root)
    return root

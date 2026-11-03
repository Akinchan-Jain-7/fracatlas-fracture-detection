"""Tests for download helpers and dataset verification, using tiny fake data."""

import hashlib
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from fracdet.data.download import md5_of_file, prepare_dataset, safe_extract
from fracdet.data.verify import verify_dataset


def _make_fake_fracatlas(root: Path, n_frac: int = 2, n_normal: int = 3) -> Path:
    """Write a miniature FracAtlas-shaped folder: images/<label>/*.jpg + dataset.csv."""
    rows = []
    for label, folder, n in [(1, "Fractured", n_frac), (0, "Non_fractured", n_normal)]:
        (root / "images" / folder).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            name = f"IMG{label}{i:04d}.jpg"
            (root / "images" / folder / name).write_bytes(b"\xff\xd8fake")
            rows.append({"image_id": name, "hand": 1, "leg": 0, "fractured": label})
    pd.DataFrame(rows).to_csv(root / "dataset.csv", index=False)
    return root


def test_md5_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    payload = b"fracture" * 10_000
    f.write_bytes(payload)
    assert md5_of_file(f) == hashlib.md5(payload).hexdigest()


def test_safe_extract_rejects_zip_slip(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escaped.txt", "pwned")
    with pytest.raises(ValueError, match="Unsafe path"):
        safe_extract(evil, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_safe_extract_normal_archive(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("FracAtlas/dataset.csv", "image_id\n")
    safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out" / "FracAtlas" / "dataset.csv").is_file()


def test_prepare_dataset_skips_when_present(tmp_path, monkeypatch):
    _make_fake_fracatlas(tmp_path / "FracAtlas")

    def _fail(*args, **kwargs):
        raise AssertionError("download must not be called when data exists")

    monkeypatch.setattr("fracdet.data.download.download_file", _fail)
    cfg = {"data": {"archive_name": "FracAtlas.zip", "md5": "x", "source_url": "http://x"}}
    assert prepare_dataset(cfg, tmp_path) == tmp_path / "FracAtlas"


def test_prepare_dataset_uses_verified_local_archive(tmp_path, monkeypatch):
    src = _make_fake_fracatlas(tmp_path / "src" / "FracAtlas")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    archive = data_dir / "FracAtlas.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src.parent))
    monkeypatch.setattr(
        "fracdet.data.download.download_file",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not download")),
    )
    cfg = {"data": {"archive_name": "FracAtlas.zip", "md5": md5_of_file(archive), "source_url": ""}}
    root = prepare_dataset(cfg, data_dir)
    assert (root / "dataset.csv").is_file()


def test_verify_passes_on_consistent_data(tmp_path):
    _make_fake_fracatlas(tmp_path / "FracAtlas", n_frac=2, n_normal=3)
    report = verify_dataset(tmp_path, expected_num_images=5, expected_num_fractured=2)
    assert report.ok, report.errors
    assert report.stats["image_files"] == 5
    assert report.stats["fractured"] == 2


def test_verify_flags_missing_image_and_wrong_counts(tmp_path):
    root = _make_fake_fracatlas(tmp_path / "FracAtlas")
    next((root / "images" / "Fractured").iterdir()).unlink()
    report = verify_dataset(tmp_path, expected_num_images=99)
    assert not report.ok
    assert any("no image file" in e for e in report.errors)
    assert any("Expected 99" in e for e in report.errors)


def test_verify_reports_missing_root(tmp_path):
    report = verify_dataset(tmp_path)
    assert not report.ok and report.root is None

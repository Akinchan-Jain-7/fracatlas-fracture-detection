from pathlib import Path

import pandas as pd
import pytest

from fracdet.data.metadata import load_metadata, region_key, view_key


def _row(**flags) -> pd.Series:
    base = {"hand": 0, "leg": 0, "hip": 0, "shoulder": 0, "mixed": 0}
    base.update({"frontal": 0, "lateral": 0, "oblique": 0})
    base.update(flags)
    return pd.Series(base)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"hand": 1}, "hand"),
        ({"leg": 1, "hip": 1}, "mixed"),
        ({"hand": 1, "mixed": 1}, "mixed"),
        ({"mixed": 1}, "mixed"),
        ({}, "unknown"),
    ],
)
def test_region_key(flags, expected):
    assert region_key(_row(**flags)) == expected


@pytest.mark.parametrize(
    ("flags", "expected"),
    [({"frontal": 1}, "frontal"), ({"frontal": 1, "lateral": 1}, "multiple"), ({}, "none")],
)
def test_view_key(flags, expected):
    assert view_key(_row(**flags)) == expected


def test_load_metadata_resolves_label_folder(fake_root):
    meta = load_metadata(fake_root)
    assert len(meta) == 40
    for path, label in zip(meta["path"], meta["label"], strict=True):
        assert Path(path).is_file()
        assert Path(path).parent.name == ("Fractured" if label else "Non_fractured")
    assert set(meta["region"]) <= {"hand", "leg", "hip", "shoulder"}


def test_stale_copy_is_ignored(fake_root):
    """A leftover copy in the wrong folder must not change which file we load."""
    meta = load_metadata(fake_root)
    frac = meta[meta.label == 1].iloc[0]
    stale = fake_root / "images" / "Non_fractured" / frac.image_id
    stale.write_bytes(Path(frac.path).read_bytes())
    again = load_metadata(fake_root)
    assert again.loc[again.image_id == frac.image_id, "path"].item() == frac.path


def test_missing_file_raises(fake_root):
    meta = load_metadata(fake_root)
    Path(meta.path.iloc[0]).unlink()
    with pytest.raises(FileNotFoundError):
        load_metadata(fake_root)
    assert len(load_metadata(fake_root, check_files=False)) == 40

import numpy as np
import pandas as pd
import pytest

from fracdet.data.split import (
    SplitConfig,
    check_split,
    consecutive_run_pairs,
    find_exclusions,
    make_split,
    summarize,
)


def _synthetic(n: int = 400, seed: int = 0):
    """Metadata + hashes with known duplicates, a label conflict and corrupt files."""
    rng = np.random.default_rng(seed)
    regions = np.array(["hand", "leg", "hip", "shoulder"])
    meta = pd.DataFrame(
        {
            # gaps in numbering so not every neighbour is consecutive
            "image_id": [f"IMG{3 * i + (i % 2):07d}.jpg" for i in range(n)],
            "label": (rng.random(n) < 0.2).astype(int),
            "region": regions[rng.integers(0, 4, n)],
        }
    )
    hashes = rng.integers(0, 256, size=(n, 32), dtype=np.uint8)  # ~128 bits apart
    md5s = [f"md5_{i}" for i in range(n)]
    # 10 -> 11 byte-identical copy; 20 -> 21 re-encoded (3 bits differ)
    md5s[11] = md5s[10]
    hashes[11] = hashes[10]
    meta.loc[11, "label"] = meta.loc[10, "label"]
    hashes[21] = hashes[20]
    hashes[21, 0] ^= 0b111
    meta.loc[21, "label"] = meta.loc[20, "label"]
    # 30 -> 31 same picture, contradictory labels
    hashes[31] = hashes[30]
    meta.loc[30, "label"], meta.loc[31, "label"] = 1, 0
    # 40 -> 41 merely similar (15 bits): grouped, not deduplicated
    hashes[41] = hashes[40]
    hashes[41, :2] ^= np.array([0xFF, 0x7F], dtype=np.uint8)
    corrupt = np.zeros(n, dtype=bool)
    corrupt[[5, 6, 7]] = True
    return meta, hashes, md5s, corrupt


def test_fold_counts_validation():
    assert SplitConfig().fold_counts() == {"train": 14, "val": 3, "test": 3}
    with pytest.raises(ValueError):
        SplitConfig(train=0.72, val=0.14, test=0.14).fold_counts()
    with pytest.raises(ValueError):
        SplitConfig(train=0.70, val=0.20, test=0.15).fold_counts()


def test_consecutive_runs_require_same_label_and_region():
    meta = pd.DataFrame(
        {
            "image_id": ["IMG0000001.jpg", "IMG0000002.jpg", "IMG0000003.jpg", "IMG0000005.jpg"],
            "label": [1, 1, 0, 0],
            "region": ["hand", "hand", "hand", "hand"],
        }
    )
    assert consecutive_run_pairs(meta) == [(0, 1)]  # 2->3 label differs, 3->5 not consecutive


def test_exclusions():
    meta = pd.DataFrame({"image_id": [f"IMG{i:07d}.jpg" for i in range(6)]})
    meta["label"] = [0, 0, 1, 0, 1, 1]
    dup_group = np.array([0, 0, 2, 2, 4, 5])
    corrupt = np.array([False, False, False, False, False, True])
    reason = find_exclusions(meta, dup_group, corrupt)
    assert reason.tolist() == [
        "",
        "duplicate_of:IMG0000000.jpg",
        "label_conflict",
        "label_conflict",
        "",
        "corrupt",
    ]


def test_make_split_invariants_and_proportions():
    meta, hashes, md5s, corrupt = _synthetic()
    df = make_split(meta, hashes, md5s, corrupt, SplitConfig())
    check_split(df)  # no group straddles splits; one image per duplicate group

    reasons = df.set_index("image_id")["exclude_reason"]
    ids = meta["image_id"]
    assert reasons[ids[11]].startswith("duplicate_of:")
    assert reasons[ids[21]].startswith("duplicate_of:")
    assert reasons[ids[30]] == reasons[ids[31]] == "label_conflict"
    assert all(reasons[ids[k]] == "corrupt" for k in (5, 6, 7))
    assert reasons[ids[40]] == reasons[ids[41]] == ""  # similar, both kept
    s = df.set_index("image_id")["split"]
    assert s[ids[40]] == s[ids[41]]  # ...but in the same split

    summary = summarize(df)
    fracs = {k: v["fraction"] for k, v in summary["splits"].items()}
    assert fracs["train"] == pytest.approx(0.70, abs=0.05)
    assert fracs["val"] == pytest.approx(0.15, abs=0.04)
    assert fracs["test"] == pytest.approx(0.15, abs=0.04)


def test_split_is_deterministic_and_seed_dependent():
    meta, hashes, md5s, corrupt = _synthetic()
    a = make_split(meta, hashes, md5s, corrupt, SplitConfig(seed=1))
    b = make_split(meta, hashes, md5s, corrupt, SplitConfig(seed=1))
    c = make_split(meta, hashes, md5s, corrupt, SplitConfig(seed=2))
    assert a["split"].tolist() == b["split"].tolist()
    assert a["split"].tolist() != c["split"].tolist()


def test_check_split_catches_straddling_group():
    meta, hashes, md5s, corrupt = _synthetic()
    df = make_split(meta, hashes, md5s, corrupt, SplitConfig())
    used = df[df.split != "excluded"]
    i, j = used.index[:2]
    df.loc[j, "split_group"] = df.loc[i, "split_group"]
    df.loc[i, "split"], df.loc[j, "split"] = "train", "test"
    with pytest.raises(AssertionError, match="span more than one split"):
        check_split(df)

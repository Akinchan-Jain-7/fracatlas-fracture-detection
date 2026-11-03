import json

import pandas as pd

from fracdet.data.eda import (
    add_image_properties,
    fracture_rate_by_id,
    rate_table,
    summarize_dataset,
    yolo_box_counts,
)
from fracdet.data.metadata import load_metadata


def test_rate_table():
    df = pd.DataFrame({"g": ["a", "a", "b", "b", "b"], "label": [1, 0, 1, 1, 0]})
    t = rate_table(df, "g")
    assert t.loc["a", "fracture_rate"] == 0.5
    assert t.loc["b", "fractured"] == 2
    assert t.index.tolist() == ["b", "a"]  # largest group first


def test_summary_on_fake_dataset(fake_root):
    meta = add_image_properties(load_metadata(fake_root))
    meta["yolo_boxes"] = yolo_box_counts(fake_root, meta["image_id"])
    meta["corrupt"] = False
    s = summarize_dataset(meta)
    json.dumps(s)  # must be serialisable
    n_frac = int(meta.label.sum())
    assert s["n_fractured"] == n_frac
    assert s["yolo"]["fractured_with_boxes"] == n_frac
    assert s["yolo"]["non_fractured_empty_file"] == len(meta) - n_frac
    assert s["yolo"]["box_count_matches_fracture_count"] == n_frac
    assert set(meta.size_group) == {"small"}


def test_missing_yolo_file_is_minus_one(fake_root):
    (fake_root / "Annotations" / "YOLO" / "IMG0000000.txt").unlink()
    counts = yolo_box_counts(fake_root, pd.Series(["IMG0000000.jpg", "IMG0000001.jpg"]))
    assert counts.iloc[0] == -1


def test_fracture_rate_by_id_sorted():
    meta = pd.DataFrame({"image_id": ["IMG0000010.jpg", "IMG0000002.jpg"], "label": [1, 0]})
    df = fracture_rate_by_id(meta, window=1)
    assert df["id_num"].tolist() == [2, 10]
    assert df["rolling_rate"].tolist() == [0.0, 1.0]

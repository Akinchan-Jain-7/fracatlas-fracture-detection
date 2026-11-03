from fracdet.paths import DATA_DIR_ENV, find_dataset_root, resolve_data_dir


def test_env_var_overrides_config(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert resolve_data_dir({"paths": {"data_dir": "data/raw"}}) == tmp_path


def test_config_used_without_env(monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    assert str(resolve_data_dir({"paths": {"data_dir": "data/raw"}})) == "data/raw"


def test_finds_nested_root(tmp_path):
    root = tmp_path / "fracatlas" / "FracAtlas"
    root.mkdir(parents=True)
    (root / "dataset.csv").write_text("image_id\n")
    assert find_dataset_root(tmp_path) == root


def test_prefers_shallowest_match(tmp_path):
    (tmp_path / "dataset.csv").write_text("x\n")
    deeper = tmp_path / "copy"
    deeper.mkdir()
    (deeper / "dataset.csv").write_text("x\n")
    assert find_dataset_root(tmp_path) == tmp_path


def test_returns_none_when_missing_or_too_deep(tmp_path):
    assert find_dataset_root(tmp_path / "does-not-exist") is None
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "dataset.csv").write_text("x\n")
    assert find_dataset_root(tmp_path, max_depth=2) is None
    assert find_dataset_root(tmp_path, max_depth=3) == deep

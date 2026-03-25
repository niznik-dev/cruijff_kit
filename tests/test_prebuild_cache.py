"""Tests for tools/inspect/prebuild_cache.py"""

from unittest.mock import patch

import yaml

from cruijff_kit.tools.inspect.prebuild_cache import prebuild_cache


def _write_summary(tmp_path, tasks):
    """Helper to write a minimal experiment_summary.yaml."""
    summary = {"evaluation": {"tasks": tasks}}
    path = tmp_path / "experiment_summary.yaml"
    path.write_text(yaml.dump(summary))
    return str(path)


def _make_dataset(tmp_path, name="data.json"):
    """Helper to create a fake dataset file."""
    path = tmp_path / name
    path.write_text('{"test": [{"text": "hello"}]}')
    return str(path)


@patch("cruijff_kit.tools.inspect.prebuild_cache.load_dataset")
def test_happy_path(mock_load, tmp_path):
    """Valid YAML with 2 dataset paths → success, datasets_cached: 2."""
    d1 = _make_dataset(tmp_path, "a.json")
    d2 = _make_dataset(tmp_path, "b.json")
    summary = _write_summary(
        tmp_path,
        [
            {"name": "task1", "dataset": d1},
            {"name": "task2", "dataset": d2},
        ],
    )

    result = prebuild_cache(summary)

    assert result["status"] == "success"
    assert result["datasets_cached"] == 2
    assert result["datasets_failed"] == 0
    assert mock_load.call_count == 2


@patch("cruijff_kit.tools.inspect.prebuild_cache.load_dataset")
def test_duplicate_paths(mock_load, tmp_path):
    """Same dataset path in multiple tasks → only cached once."""
    d1 = _make_dataset(tmp_path, "shared.json")
    summary = _write_summary(
        tmp_path,
        [
            {"name": "task1", "dataset": d1},
            {"name": "task2", "dataset": d1},
        ],
    )

    result = prebuild_cache(summary)

    assert result["status"] == "success"
    assert result["datasets_cached"] == 1
    assert mock_load.call_count == 1


@patch("cruijff_kit.tools.inspect.prebuild_cache.load_dataset")
def test_missing_dataset_file(mock_load, tmp_path):
    """Path in YAML doesn't exist → continues, reports failure."""
    d1 = _make_dataset(tmp_path, "real.json")
    missing = str(tmp_path / "nonexistent.json")
    summary = _write_summary(
        tmp_path,
        [
            {"name": "task1", "dataset": d1},
            {"name": "task2", "dataset": missing},
        ],
    )

    result = prebuild_cache(summary)

    assert result["status"] == "success"
    assert result["datasets_cached"] == 1
    assert result["datasets_failed"] == 1
    assert missing in result["paths_failed"]
    assert mock_load.call_count == 1


@patch("cruijff_kit.tools.inspect.prebuild_cache.load_dataset")
def test_no_evaluation_tasks(mock_load, tmp_path):
    """YAML has no evaluation.tasks → success, datasets_cached: 0."""
    summary = _write_summary(tmp_path, [])

    result = prebuild_cache(summary)

    assert result["status"] == "success"
    assert result["datasets_cached"] == 0
    assert mock_load.call_count == 0


def test_missing_yaml_file():
    """Bad path → returns error dict."""
    result = prebuild_cache("/nonexistent/path/summary.yaml")

    assert result["status"] == "error"
    assert "not found" in result["message"].lower()


@patch("cruijff_kit.tools.inspect.prebuild_cache.load_dataset")
def test_no_dataset_key_in_tasks(mock_load, tmp_path):
    """Tasks without dataset key are skipped gracefully."""
    summary = _write_summary(
        tmp_path,
        [
            {"name": "task1"},
            {"name": "task2", "dataset": None},
        ],
    )

    result = prebuild_cache(summary)

    assert result["status"] == "success"
    assert result["datasets_cached"] == 0
    assert mock_load.call_count == 0


@patch("cruijff_kit.tools.inspect.prebuild_cache.load_dataset", None)
def test_missing_datasets_package(tmp_path):
    """Missing datasets package → returns friendly error."""
    d1 = _make_dataset(tmp_path, "a.json")
    summary = _write_summary(
        tmp_path,
        [
            {"name": "task1", "dataset": d1},
        ],
    )

    result = prebuild_cache(summary)

    assert result["status"] == "error"
    assert "datasets not installed" in result["message"]

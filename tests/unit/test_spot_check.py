"""Tests for cruijff_kit.utils.spot_check.

Only tests the data loading function (load_data), since spot_check() and main()
require GPU access via llm_utils.
"""

import json
import tempfile
from pathlib import Path

import pytest

from cruijff_kit.utils.spot_check import load_data


@pytest.fixture
def flat_data_file(tmp_path):
    """Create a flat-format JSON data file."""
    data = [
        {"input": "hello", "output": "Hello"},
        {"input": "world", "output": "World"},
        {"input": "test", "output": "Test"},
        {"input": "data", "output": "Data"},
        {"input": "five", "output": "Five"},
    ]
    path = tmp_path / "flat.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def nested_data_file(tmp_path):
    """Create a nested-format JSON data file with train/validation splits."""
    data = {
        "train": [
            {"input": "train1", "output": "Train1"},
            {"input": "train2", "output": "Train2"},
        ],
        "validation": [
            {"input": "val1", "output": "Val1"},
            {"input": "val2", "output": "Val2"},
            {"input": "val3", "output": "Val3"},
        ],
    }
    path = tmp_path / "nested.json"
    path.write_text(json.dumps(data))
    return str(path)


class TestLoadData:
    """Tests for load_data function."""

    def test_loads_flat_format(self, flat_data_file):
        prompts, targets = load_data(flat_data_file, n=3)
        assert len(prompts) == 3
        assert len(targets) == 3
        assert prompts[0] == "hello"
        assert targets[0] == "Hello"

    def test_respects_n_limit(self, flat_data_file):
        prompts, targets = load_data(flat_data_file, n=2)
        assert len(prompts) == 2

    def test_n_larger_than_data(self, flat_data_file):
        prompts, targets = load_data(flat_data_file, n=100)
        assert len(prompts) == 5

    def test_nested_with_explicit_split(self, nested_data_file):
        prompts, targets = load_data(nested_data_file, n=10, split="train")
        assert len(prompts) == 2
        assert prompts[0] == "train1"

    def test_nested_with_validation_split(self, nested_data_file):
        prompts, targets = load_data(nested_data_file, n=10, split="validation")
        assert len(prompts) == 3
        assert prompts[0] == "val1"

    def test_nested_auto_detects_validation(self, nested_data_file):
        """Without explicit split, should auto-detect and use validation."""
        prompts, targets = load_data(nested_data_file, n=10)
        assert len(prompts) == 3
        assert prompts[0] == "val1"

    def test_invalid_split_raises(self, nested_data_file):
        with pytest.raises(ValueError, match="Split 'nonexistent' not found"):
            load_data(nested_data_file, n=10, split="nonexistent")

    def test_nested_train_only_falls_back_to_train(self, tmp_path):
        """Nested JSON with 'train' but no 'validation' should fall back to 'train'."""
        data = {
            "train": [
                {"input": "t1", "output": "T1"},
                {"input": "t2", "output": "T2"},
            ],
        }
        path = tmp_path / "train_only.json"
        path.write_text(json.dumps(data))

        prompts, targets = load_data(str(path), n=10)
        assert len(prompts) == 2
        assert prompts[0] == "t1"

    def test_returns_correct_types(self, flat_data_file):
        prompts, targets = load_data(flat_data_file, n=2)
        assert all(isinstance(p, str) for p in prompts)
        assert all(isinstance(t, str) for t in targets)

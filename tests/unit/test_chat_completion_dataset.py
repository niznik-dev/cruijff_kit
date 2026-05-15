"""Regression: chat_completion_dataset must honor max_samples for training-set slicing (#447)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from cruijff_kit.tools.torchtune.datasets.chat_completion import chat_completion_dataset


@pytest.fixture
def small_json(tmp_path):
    path = tmp_path / "data.json"
    rows = {
        "train": [{"input": str(i), "output": str(i)} for i in range(10)],
        "validation": [{"input": "v", "output": "v"}],
    }
    path.write_text(json.dumps(rows))
    return path


@patch("transformers.AutoTokenizer.from_pretrained")
def test_max_samples_slices_train(mock_tok, small_json):
    mock_tok.return_value = MagicMock()
    ds = chat_completion_dataset(
        tokenizer=None,
        source="json",
        data_files=str(small_json),
        model_path="/fake/model",
        split="train",
        max_samples=3,
    )
    assert len(ds) == 3


@patch("transformers.AutoTokenizer.from_pretrained")
def test_max_samples_none_loads_all(mock_tok, small_json):
    mock_tok.return_value = MagicMock()
    ds = chat_completion_dataset(
        tokenizer=None,
        source="json",
        data_files=str(small_json),
        model_path="/fake/model",
        split="train",
    )
    assert len(ds) == 10


@patch("transformers.AutoTokenizer.from_pretrained")
def test_max_samples_larger_than_data_returns_full(mock_tok, small_json):
    """Python slice semantics: rows[:999] on a 10-row list returns 10 rows."""
    mock_tok.return_value = MagicMock()
    ds = chat_completion_dataset(
        tokenizer=None,
        source="json",
        data_files=str(small_json),
        model_path="/fake/model",
        split="train",
        max_samples=999,
    )
    assert len(ds) == 10

"""Tests for cruijff_kit.tools.inspect.summary_binary."""

import json
import zipfile
from pathlib import Path

import pytest

from cruijff_kit.tools.inspect.summary_binary import (
    load_samples,
    get_prediction,
    compute_metrics,
)


def _make_sample(target: str, prediction: str) -> dict:
    """Helper to create a sample dict matching inspect-ai format."""
    return {
        "target": target,
        "messages": [
            {"role": "user", "content": "What is the answer?"},
            {"role": "assistant", "content": prediction},
        ],
    }


def _write_eval_file(path: Path, samples: list[dict]):
    """Write samples into a fake .eval zip archive."""
    with zipfile.ZipFile(path, "w") as z:
        for i, sample in enumerate(samples):
            z.writestr(f"samples/{i}.json", json.dumps(sample))


@pytest.fixture
def perfect_eval(tmp_path):
    """An eval file where the model gets everything right."""
    samples = [
        _make_sample("1", "1"),
        _make_sample("1", "1"),
        _make_sample("0", "0"),
        _make_sample("0", "0"),
    ]
    path = tmp_path / "perfect.eval"
    _write_eval_file(path, samples)
    return path


@pytest.fixture
def mixed_eval(tmp_path):
    """An eval file with a mix of correct and incorrect predictions."""
    samples = [
        _make_sample("1", "1"),  # TP
        _make_sample("1", "0"),  # FN
        _make_sample("0", "0"),  # TN
        _make_sample("0", "1"),  # FP
    ]
    path = tmp_path / "mixed.eval"
    _write_eval_file(path, samples)
    return path


@pytest.fixture
def all_wrong_eval(tmp_path):
    """An eval file where the model gets everything wrong."""
    samples = [
        _make_sample("1", "0"),
        _make_sample("1", "0"),
        _make_sample("0", "1"),
        _make_sample("0", "1"),
    ]
    path = tmp_path / "wrong.eval"
    _write_eval_file(path, samples)
    return path


@pytest.fixture
def eval_with_other(tmp_path):
    """An eval file with some non-binary predictions."""
    samples = [
        _make_sample("1", "1"),
        _make_sample("0", "0"),
        _make_sample("1", "maybe"),
        _make_sample("0", "I don't know"),
    ]
    path = tmp_path / "other.eval"
    _write_eval_file(path, samples)
    return path


@pytest.fixture
def empty_eval(tmp_path):
    """An eval file with no samples."""
    path = tmp_path / "empty.eval"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("metadata.json", "{}")
    return path


class TestLoadSamples:
    """Tests for load_samples."""

    def test_loads_correct_count(self, perfect_eval):
        samples = load_samples(perfect_eval)
        assert len(samples) == 4

    def test_returns_list_of_dicts(self, perfect_eval):
        samples = load_samples(perfect_eval)
        assert all(isinstance(s, dict) for s in samples)

    def test_empty_file(self, empty_eval):
        samples = load_samples(empty_eval)
        assert len(samples) == 0


class TestGetPrediction:
    """Tests for get_prediction."""

    def test_extracts_assistant_message(self):
        sample = _make_sample("1", "0")
        assert get_prediction(sample) == "0"

    def test_strips_whitespace(self):
        sample = {
            "messages": [
                {"role": "assistant", "content": "  1  "},
            ]
        }
        assert get_prediction(sample) == "1"

    def test_uses_last_assistant_message(self):
        sample = {
            "messages": [
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "try again"},
                {"role": "assistant", "content": "second"},
            ]
        }
        assert get_prediction(sample) == "second"

    def test_no_assistant_message(self):
        sample = {"messages": [{"role": "user", "content": "hello"}]}
        assert get_prediction(sample) == ""

    def test_empty_messages(self):
        sample = {"messages": []}
        assert get_prediction(sample) == ""

    def test_no_messages_key(self):
        sample = {}
        assert get_prediction(sample) == ""


class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_perfect_accuracy(self, perfect_eval):
        result = compute_metrics(perfect_eval)
        assert result["status"] == "success"
        assert result["accuracy"] == 1.0
        assert result["balanced_accuracy"] == 1.0
        assert result["f1"] == 1.0

    def test_all_wrong(self, all_wrong_eval):
        result = compute_metrics(all_wrong_eval)
        assert result["status"] == "success"
        assert result["accuracy"] == 0.0
        assert result["balanced_accuracy"] == 0.0
        assert result["f1"] == 0.0

    def test_mixed_results(self, mixed_eval):
        result = compute_metrics(mixed_eval)
        assert result["status"] == "success"
        assert result["accuracy"] == 0.5
        assert result["samples"] == 4
        cm = result["confusion_matrix"]
        assert cm["tp"] == 1
        assert cm["tn"] == 1
        assert cm["fp"] == 1
        assert cm["fn"] == 1

    def test_other_predictions_tracked(self, eval_with_other):
        result = compute_metrics(eval_with_other)
        assert result["status"] == "success"
        assert result["confusion_matrix"]["other"] == 2

    def test_empty_file_returns_error(self, empty_eval):
        result = compute_metrics(empty_eval)
        assert result["status"] == "error"

    def test_sample_count(self, mixed_eval):
        result = compute_metrics(mixed_eval)
        assert result["samples"] == 4

    def test_path_in_result(self, mixed_eval):
        result = compute_metrics(mixed_eval)
        assert result["path"] == str(mixed_eval)

    def test_all_positive_predictions(self, tmp_path):
        """Model always predicts 1."""
        samples = [
            _make_sample("1", "1"),
            _make_sample("1", "1"),
            _make_sample("0", "1"),
            _make_sample("0", "1"),
        ]
        path = tmp_path / "all_pos.eval"
        _write_eval_file(path, samples)
        result = compute_metrics(path)
        assert result["recall_1"] == 1.0
        assert result["recall_0"] == 0.0
        assert result["precision_1"] == 0.5

    def test_all_negative_predictions(self, tmp_path):
        """Model always predicts 0."""
        samples = [
            _make_sample("1", "0"),
            _make_sample("1", "0"),
            _make_sample("0", "0"),
            _make_sample("0", "0"),
        ]
        path = tmp_path / "all_neg.eval"
        _write_eval_file(path, samples)
        result = compute_metrics(path)
        assert result["recall_1"] == 0.0
        assert result["recall_0"] == 1.0
        assert result["precision_1"] == 0.0
        assert result["f1"] == 0.0

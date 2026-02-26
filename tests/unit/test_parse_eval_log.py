"""Tests for cruijff_kit.tools.inspect.parse_eval_log."""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from cruijff_kit.tools.inspect.parse_eval_log import parse_eval_log

# The function does a lazy `from inspect_ai.log import read_eval_log` inside
# the function body, so we patch it at the source module.
PATCH_TARGET = "inspect_ai.log.read_eval_log"


def _make_metric(value):
    """Create a mock Metric object with a .value attribute."""
    return SimpleNamespace(value=value)


def _make_log(
    task="capitalization",
    model="meta-llama/Llama-3.2-1B-Instruct",
    n_samples=100,
    scorer_name="exact_match",
    metrics=None,
):
    """Create a mock inspect-ai EvalLog object."""
    if metrics is None:
        metrics = {"accuracy": _make_metric(0.85)}

    score = SimpleNamespace(name=scorer_name, metrics=metrics)
    results = SimpleNamespace(scores=[score])
    samples = [SimpleNamespace() for _ in range(n_samples)]
    eval_info = SimpleNamespace(task=task, model=model)

    return SimpleNamespace(eval=eval_info, results=results, samples=samples)


class TestParseEvalLog:
    """Tests for parse_eval_log."""

    @patch(PATCH_TARGET)
    def test_success_basic(self, mock_read, tmp_path):
        log_file = tmp_path / "test.eval"
        log_file.touch()
        mock_read.return_value = _make_log()

        result = parse_eval_log(str(log_file))

        assert result["status"] == "success"
        assert result["task"] == "capitalization"
        assert result["model"] == "meta-llama/Llama-3.2-1B-Instruct"
        assert result["samples"] == 100
        assert result["scorer"] == "exact_match"

    @patch(PATCH_TARGET)
    def test_accuracy_promoted_to_top_level(self, mock_read, tmp_path):
        log_file = tmp_path / "test.eval"
        log_file.touch()
        mock_read.return_value = _make_log(
            metrics={"accuracy": _make_metric(0.92)}
        )

        result = parse_eval_log(str(log_file))

        assert result["accuracy"] == 0.92
        assert result["metrics"]["accuracy"] == 0.92

    @patch(PATCH_TARGET)
    def test_multiple_metrics(self, mock_read, tmp_path):
        log_file = tmp_path / "test.eval"
        log_file.touch()
        mock_read.return_value = _make_log(
            metrics={
                "accuracy": _make_metric(0.85),
                "f1": _make_metric(0.80),
                "recall": _make_metric(0.75),
            }
        )

        result = parse_eval_log(str(log_file))

        assert result["metrics"]["accuracy"] == 0.85
        assert result["metrics"]["f1"] == 0.80
        assert result["metrics"]["recall"] == 0.75

    @patch(PATCH_TARGET)
    def test_metric_without_value_attribute(self, mock_read, tmp_path):
        """Metrics that are plain values (not Metric objects) should work."""
        log_file = tmp_path / "test.eval"
        log_file.touch()
        mock_read.return_value = _make_log(
            metrics={"accuracy": _make_metric(0.85), "count": 42}
        )

        result = parse_eval_log(str(log_file))

        assert result["metrics"]["count"] == 42

    @patch(PATCH_TARGET)
    def test_no_accuracy_not_promoted(self, mock_read, tmp_path):
        """If no accuracy metric, top-level accuracy key shouldn't exist."""
        log_file = tmp_path / "test.eval"
        log_file.touch()
        mock_read.return_value = _make_log(
            metrics={"f1": _make_metric(0.80)}
        )

        result = parse_eval_log(str(log_file))

        assert "accuracy" not in result
        assert result["metrics"]["f1"] == 0.80

    @patch(PATCH_TARGET)
    def test_no_results(self, mock_read, tmp_path):
        log_file = tmp_path / "test.eval"
        log_file.touch()
        log = _make_log()
        log.results = None
        mock_read.return_value = log

        result = parse_eval_log(str(log_file))

        assert result["status"] == "success"
        assert result["metrics"] == {}
        assert "scorer" not in result

    @patch(PATCH_TARGET)
    def test_no_samples(self, mock_read, tmp_path):
        log_file = tmp_path / "test.eval"
        log_file.touch()
        log = _make_log()
        log.samples = None
        mock_read.return_value = log

        result = parse_eval_log(str(log_file))

        assert result["samples"] == 0

    @patch(PATCH_TARGET)
    def test_empty_scores_list(self, mock_read, tmp_path):
        log_file = tmp_path / "test.eval"
        log_file.touch()
        log = _make_log()
        log.results.scores = []
        mock_read.return_value = log

        result = parse_eval_log(str(log_file))

        assert result["status"] == "success"
        assert result["metrics"] == {}

    def test_file_not_found(self):
        result = parse_eval_log("/nonexistent/path/fake.eval")

        assert result["status"] == "error"
        assert "not found" in result["message"].lower() or "File not found" in result["message"]

    @patch(PATCH_TARGET)
    def test_read_eval_log_exception(self, mock_read, tmp_path):
        log_file = tmp_path / "test.eval"
        log_file.touch()
        mock_read.side_effect = RuntimeError("corrupted file")

        result = parse_eval_log(str(log_file))

        assert result["status"] == "error"
        assert "corrupted file" in result["message"]

    @patch(PATCH_TARGET)
    def test_path_in_result(self, mock_read, tmp_path):
        log_file = tmp_path / "test.eval"
        log_file.touch()
        mock_read.return_value = _make_log()

        result = parse_eval_log(str(log_file))

        assert result["path"] == str(log_file)

    def test_path_in_error_result(self):
        path = "/nonexistent/fake.eval"
        result = parse_eval_log(path)
        assert result["path"] == path

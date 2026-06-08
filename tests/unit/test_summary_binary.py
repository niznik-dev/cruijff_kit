"""Tests for cruijff_kit.tools.inspect.summary_binary.

Fixtures build genuine inspect-ai eval logs via write_eval_log so the code under
test exercises the real read_eval_log() path rather than a hand-rolled zip (#317).
"""

from pathlib import Path

import pytest
from inspect_ai.log import EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelOutput

from cruijff_kit.tools.inspect.summary_binary import (
    compute_metrics,
    get_prediction,
    load_samples,
)

_CREATED = "2026-01-01T00:00:00"


def _make_sample(target: str, prediction: str) -> EvalSample:
    """A single eval sample: the model emitted `prediction` for `target`."""
    return EvalSample(
        id=0,
        epoch=1,
        input="What is the answer?",
        target=target,
        messages=[
            ChatMessageUser(content="What is the answer?"),
            ChatMessageAssistant(content=prediction),
        ],
        output=ModelOutput.from_content("mockmodel", prediction),
    )


def _write_eval_file(path: Path, samples: list[EvalSample]):
    """Write samples into a real inspect-ai .eval log."""
    for i, sample in enumerate(samples):
        sample.id = i  # ids must be unique within a log
    log = EvalLog(
        eval=EvalSpec(
            created=_CREATED,
            task="binary_test",
            dataset={},
            model="mockmodel",
            config={},
        ),
        samples=samples,
    )
    write_eval_log(log, str(path))


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
    """A header-only eval file with no samples."""
    path = tmp_path / "empty.eval"
    log = EvalLog(
        eval=EvalSpec(
            created=_CREATED,
            task="binary_test",
            dataset={},
            model="mockmodel",
            config={},
        )
    )
    write_eval_log(log, str(path))
    return path


class TestLoadSamples:
    """Tests for load_samples."""

    def test_loads_correct_count(self, perfect_eval):
        samples = load_samples(perfect_eval)
        assert len(samples) == 4

    def test_returns_list_of_samples(self, perfect_eval):
        samples = load_samples(perfect_eval)
        assert all(isinstance(s, EvalSample) for s in samples)

    def test_empty_file(self, empty_eval):
        samples = load_samples(empty_eval)
        assert len(samples) == 0


class TestGetPrediction:
    """Tests for get_prediction."""

    def test_extracts_completion(self):
        assert get_prediction(_make_sample("1", "0")) == "0"

    def test_strips_whitespace(self):
        assert get_prediction(_make_sample("1", "  1  ")) == "1"

    def test_falls_back_to_last_assistant_message(self):
        """With no model output, walk messages and take the last assistant turn."""
        sample = EvalSample(
            id=0,
            epoch=1,
            input="q",
            target="1",
            messages=[
                ChatMessageAssistant(content="first"),
                ChatMessageUser(content="try again"),
                ChatMessageAssistant(content="second"),
            ],
        )
        assert get_prediction(sample) == "second"

    def test_no_assistant_message(self):
        sample = EvalSample(
            id=0,
            epoch=1,
            input="q",
            target="1",
            messages=[ChatMessageUser(content="hello")],
        )
        assert get_prediction(sample) == ""

    def test_empty_messages(self):
        sample = EvalSample(id=0, epoch=1, input="q", target="1", messages=[])
        assert get_prediction(sample) == ""

    def test_no_messages_and_no_output(self):
        sample = EvalSample(id=0, epoch=1, input="q", target="1")
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

    def test_non_binary_targets_silently_skipped(self, tmp_path):
        """Samples with targets outside {0, 1} are silently ignored in the matrix."""
        samples = [
            _make_sample("1", "1"),  # counted
            _make_sample("0", "0"),  # counted
            _make_sample("2", "1"),  # target "2" — not in matrix, skipped
            _make_sample("3", "0"),  # target "3" — not in matrix, skipped
        ]
        path = tmp_path / "non_binary.eval"
        _write_eval_file(path, samples)
        result = compute_metrics(path)
        assert result["status"] == "success"
        assert result["samples"] == 4
        # Only the 2 valid samples contribute to the confusion matrix
        cm = result["confusion_matrix"]
        assert cm["tp"] == 1
        assert cm["tn"] == 1
        assert cm["fp"] == 0
        assert cm["fn"] == 0


class TestClassBalance:
    """Class balance is eval-set provenance — the actual label split, computed
    from the confusion matrix's actual-class totals. Reported by summarize as a
    neutral fact, not framed as a performance floor.
    """

    def test_balanced_split(self, perfect_eval):
        """2 class-1, 2 class-0 → 50/50."""
        cb = compute_metrics(perfect_eval)["class_balance"]
        assert cb == {"frac_1": 0.5, "frac_0": 0.5, "n_1": 2, "n_0": 2}

    def test_imbalanced_split(self, tmp_path):
        """3 class-1, 1 class-0 → 75/25, independent of correctness."""
        samples = [
            _make_sample("1", "0"),  # wrong predictions must not move the balance
            _make_sample("1", "1"),
            _make_sample("1", "1"),
            _make_sample("0", "1"),
        ]
        path = tmp_path / "imbalanced.eval"
        _write_eval_file(path, samples)
        cb = compute_metrics(path)["class_balance"]
        assert cb == {"frac_1": 0.75, "frac_0": 0.25, "n_1": 3, "n_0": 1}

    def test_unparsed_predictions_count_toward_actual_class(self, eval_with_other):
        """Targets 1,0,1,0 with two unparsed predictions: the actual-label split
        is still 50/50 — an unparseable output doesn't drop its sample from the
        denominator."""
        cb = compute_metrics(eval_with_other)["class_balance"]
        assert cb == {"frac_1": 0.5, "frac_0": 0.5, "n_1": 2, "n_0": 2}

    def test_denominator_excludes_non_binary_targets(self, tmp_path):
        """Only {0,1}-labeled samples enter the balance; "2"/"3" targets don't."""
        samples = [
            _make_sample("1", "1"),
            _make_sample("0", "0"),
            _make_sample("2", "1"),  # excluded
            _make_sample("3", "0"),  # excluded
        ]
        path = tmp_path / "non_binary_balance.eval"
        _write_eval_file(path, samples)
        cb = compute_metrics(path)["class_balance"]
        assert cb == {"frac_1": 0.5, "frac_0": 0.5, "n_1": 1, "n_0": 1}

    def test_all_non_binary_targets_balance_is_zero(self, tmp_path):
        """No {0,1} targets at all → n_labeled is 0, so the fractions report 0
        rather than raising ZeroDivisionError."""
        samples = [
            _make_sample("2", "1"),
            _make_sample("3", "0"),
        ]
        path = tmp_path / "no_binary.eval"
        _write_eval_file(path, samples)
        cb = compute_metrics(path)["class_balance"]
        assert cb == {"frac_1": 0, "frac_0": 0, "n_1": 0, "n_0": 0}

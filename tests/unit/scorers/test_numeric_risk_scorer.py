"""Unit tests for tools/inspect/scorers/numeric_risk_scorer.py

Tests the numeric risk scorer that parses probabilities from model text output:
- Parsing: valid decimals, whitespace, out-of-range, non-numeric
- Binary scoring: risk_score, option_probs structure, correctness thresholding
- Custom labels
- Edge cases: empty/malformed completions
- Metric compatibility: synthetic option_probs work with calibration metrics
"""

import math
import asyncio
import pytest
from unittest.mock import MagicMock

from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput
from inspect_ai.model._chat_message import ChatMessageAssistant
from inspect_ai.scorer import Score, CORRECT, INCORRECT, Target

from cruijff_kit.tools.inspect.scorers.numeric_risk_scorer import (
    numeric_risk_scorer,
    _parse_risk_score,
)
from cruijff_kit.tools.inspect.scorers.calibration_metrics import (
    expected_calibration_error,
    risk_calibration_error,
    brier_score,
    auc_score,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_state(completion: str):
    """Build a mock TaskState with the given text completion (no logprobs needed)."""
    message = ChatMessageAssistant(content=completion)
    choice = ChatCompletionChoice(message=message, logprobs=None)
    output = ModelOutput(choices=[choice], completion=completion)

    state = MagicMock()
    state.output = output
    return state


def _make_target(text: str) -> Target:
    """Build a Target with the given text."""
    return Target([text])


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.new_event_loop().run_until_complete(coro)


# =============================================================================
# _parse_risk_score tests
# =============================================================================

class TestParseRiskScore:
    """Tests for the _parse_risk_score helper function."""

    def test_valid_decimal(self):
        assert _parse_risk_score("0.73") == pytest.approx(0.73)

    def test_zero(self):
        assert _parse_risk_score("0.0") == pytest.approx(0.0)

    def test_one(self):
        assert _parse_risk_score("1.0") == pytest.approx(1.0)

    def test_integer_zero(self):
        assert _parse_risk_score("0") == pytest.approx(0.0)

    def test_integer_one(self):
        assert _parse_risk_score("1") == pytest.approx(1.0)

    def test_leading_trailing_whitespace(self):
        assert _parse_risk_score("  0.42  \n") == pytest.approx(0.42)

    def test_negative_returns_none(self):
        assert _parse_risk_score("-0.1") is None

    def test_above_one_returns_none(self):
        assert _parse_risk_score("1.01") is None

    def test_non_numeric_returns_none(self):
        assert _parse_risk_score("hello") is None

    def test_empty_string_returns_none(self):
        assert _parse_risk_score("") is None

    def test_none_input_returns_none(self):
        assert _parse_risk_score(None) is None


# =============================================================================
# Binary scoring tests
# =============================================================================

class TestBinaryScoring:
    """Tests for binary classification with default labels ("0", "1")."""

    def test_high_risk_correct(self):
        """Model outputs 0.8, target is "1" (positive class) -> CORRECT."""
        state = _make_state("0.8")
        target = _make_target("1")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT
        assert result.metadata["risk_score"] == pytest.approx(0.8)
        assert result.metadata["option_probs"]["0"] == pytest.approx(0.2)
        assert result.metadata["option_probs"]["1"] == pytest.approx(0.8)
        assert result.metadata["target"] == "1"

    def test_low_risk_correct(self):
        """Model outputs 0.2, target is "0" (negative class) -> CORRECT."""
        state = _make_state("0.2")
        target = _make_target("0")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT
        assert result.metadata["risk_score"] == pytest.approx(0.2)

    def test_high_risk_incorrect(self):
        """Model outputs 0.8, target is "0" -> INCORRECT."""
        state = _make_state("0.8")
        target = _make_target("0")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT

    def test_low_risk_incorrect(self):
        """Model outputs 0.2, target is "1" -> INCORRECT."""
        state = _make_state("0.2")
        target = _make_target("1")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT

    def test_exactly_0_5_predicts_positive(self):
        """At threshold 0.5 exactly, predicted label is positive (last label)."""
        state = _make_state("0.5")
        target = _make_target("1")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT
        assert result.metadata["risk_score"] == pytest.approx(0.5)

    def test_answer_is_string_risk_score(self):
        """Score.answer should be the string representation of the parsed risk score."""
        state = _make_state("  0.73  ")
        target = _make_target("1")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.answer == "0.73"

    def test_option_probs_sum_to_one(self):
        """Synthetic option_probs should always sum to 1."""
        state = _make_state("0.37")
        target = _make_target("0")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        total = sum(result.metadata["option_probs"].values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_risk_zero(self):
        """Risk score of 0.0 -> all probability on negative class."""
        state = _make_state("0.0")
        target = _make_target("0")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT
        assert result.metadata["risk_score"] == pytest.approx(0.0)
        assert result.metadata["option_probs"]["0"] == pytest.approx(1.0)
        assert result.metadata["option_probs"]["1"] == pytest.approx(0.0)

    def test_risk_one(self):
        """Risk score of 1.0 -> all probability on positive class."""
        state = _make_state("1.0")
        target = _make_target("1")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT
        assert result.metadata["risk_score"] == pytest.approx(1.0)
        assert result.metadata["option_probs"]["0"] == pytest.approx(0.0)
        assert result.metadata["option_probs"]["1"] == pytest.approx(1.0)


# =============================================================================
# Custom labels tests
# =============================================================================

class TestCustomLabels:
    """Tests with non-default label names."""

    def test_custom_labels(self):
        """Labels ("A", "B") should work correctly."""
        state = _make_state("0.6")
        target = _make_target("B")

        score_fn = numeric_risk_scorer(labels=("A", "B"))
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT
        assert result.metadata["option_probs"]["A"] == pytest.approx(0.4)
        assert result.metadata["option_probs"]["B"] == pytest.approx(0.6)

    def test_custom_labels_negative_correct(self):
        """Low risk with custom labels, target is negative class."""
        state = _make_state("0.3")
        target = _make_target("A")

        score_fn = numeric_risk_scorer(labels=("A", "B"))
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT
        assert result.metadata["risk_score"] == pytest.approx(0.3)


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for error handling and malformed input."""

    def test_non_numeric_completion(self):
        """Non-numeric text -> INCORRECT with risk_score=None."""
        state = _make_state("I think the answer is about 0.5")
        target = _make_target("1")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT
        assert result.metadata["risk_score"] is None
        assert result.metadata["option_probs"] is None
        assert result.metadata["target"] == "1"

    def test_empty_completion(self):
        """Empty string -> INCORRECT with risk_score=None."""
        state = _make_state("")
        target = _make_target("0")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT
        assert result.metadata["risk_score"] is None

    def test_out_of_range_completion(self):
        """Value > 1 -> INCORRECT with risk_score=None."""
        state = _make_state("1.5")
        target = _make_target("1")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT
        assert result.metadata["risk_score"] is None

    def test_negative_completion(self):
        """Negative value -> INCORRECT with risk_score=None."""
        state = _make_state("-0.3")
        target = _make_target("0")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT
        assert result.metadata["risk_score"] is None

    def test_whitespace_only(self):
        """Whitespace-only completion -> INCORRECT."""
        state = _make_state("   \n  ")
        target = _make_target("1")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT
        assert result.metadata["risk_score"] is None

    def test_none_completion(self):
        """None completion -> INCORRECT without AttributeError.

        The source does `state.output.completion.strip()` in the fallback path,
        so None must be handled gracefully by _parse_risk_score's except clause.
        """
        state = _make_state("placeholder")
        # Override completion to None after construction
        state.output = MagicMock()
        state.output.completion = None

        target = _make_target("1")

        score_fn = numeric_risk_scorer()
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT
        assert result.metadata["risk_score"] is None
        assert result.answer == ""


# =============================================================================
# Metric compatibility tests
# =============================================================================

class TestMetricCompatibility:
    """Verify synthetic option_probs work with all calibration metrics."""

    def _make_scores(self) -> list[Score]:
        """Build a set of Score objects matching numeric_risk_scorer output format."""
        return [
            Score(value=CORRECT, metadata={
                "risk_score": 0.8,
                "option_probs": {"0": 0.2, "1": 0.8},
                "target": "1",
            }),
            Score(value=CORRECT, metadata={
                "risk_score": 0.3,
                "option_probs": {"0": 0.7, "1": 0.3},
                "target": "0",
            }),
            Score(value=INCORRECT, metadata={
                "risk_score": 0.6,
                "option_probs": {"0": 0.4, "1": 0.6},
                "target": "0",
            }),
            Score(value=CORRECT, metadata={
                "risk_score": 0.9,
                "option_probs": {"0": 0.1, "1": 0.9},
                "target": "1",
            }),
        ]

    def test_expected_calibration_error_works(self):
        """ECE metric runs without error on synthetic option_probs."""
        metric_fn = expected_calibration_error()
        result = metric_fn(self._make_scores())
        assert isinstance(result, float)
        assert not math.isnan(result)

    def test_risk_calibration_error_works(self):
        """Risk ECE metric runs without error on synthetic option_probs."""
        metric_fn = risk_calibration_error()
        result = metric_fn(self._make_scores())
        assert isinstance(result, float)
        assert not math.isnan(result)

    def test_brier_score_works(self):
        """Brier score metric runs without error on synthetic option_probs."""
        metric_fn = brier_score()
        result = metric_fn(self._make_scores())
        assert isinstance(result, float)
        assert not math.isnan(result)

    def test_auc_score_works(self):
        """AUC metric runs without error on synthetic option_probs."""
        metric_fn = auc_score()
        result = metric_fn(self._make_scores())
        assert isinstance(result, float)
        assert not math.isnan(result)

    def test_scores_with_none_risk_handled(self):
        """Metrics gracefully skip samples with risk_score=None."""
        scores = self._make_scores()
        scores.append(Score(value=INCORRECT, metadata={
            "risk_score": None,
            "option_probs": None,
            "target": "1",
        }))
        metric_fn = brier_score()
        result = metric_fn(scores)
        assert isinstance(result, float)
        assert not math.isnan(result)

"""Unit tests for tools/inspect/scorers/continuous_scorer.py

Tests the continuous (regression) scorer that parses numeric output and
computes regression metrics:
- _parse_number: integers, decimals, negatives, formatted ($, %, comma), bad input
- continuous_scorer score function: prediction, target, error in metadata
- Aggregate metrics: mae, rmse, r_squared, parse_rate
- Registry wiring: build_scorers("continuous_scorer") returns a usable scorer
"""

import asyncio
import math

import pytest
from inspect_ai.model._chat_message import ChatMessageAssistant
from inspect_ai.model._model_output import ChatCompletionChoice, ModelOutput
from inspect_ai.scorer import Score, Target
from unittest.mock import MagicMock

from cruijff_kit.tools.inspect.scorers import build_scorers
from cruijff_kit.tools.inspect.scorers.continuous_scorer import (
    _parse_number,
    continuous_scorer,
    mae,
    parse_rate,
    r_squared,
    rmse,
)


# ----------------------------- helpers --------------------------------


def _make_state(completion: str):
    message = ChatMessageAssistant(content=completion)
    choice = ChatCompletionChoice(message=message, logprobs=None)
    output = ModelOutput(choices=[choice], completion=completion)
    state = MagicMock()
    state.output = output
    return state


def _make_target(text: str) -> Target:
    return Target([text])


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _score(prediction, target_value, error):
    """Build a Score with the metadata shape continuous_scorer produces."""
    return Score(
        value=abs(error) if error is not None else 0,
        answer=str(prediction) if prediction is not None else "",
        metadata={
            "prediction": prediction,
            "target_value": target_value,
            "error": error,
        },
    )


# ----------------------------- _parse_number --------------------------


class TestParseNumber:
    def test_integer(self):
        assert _parse_number("42") == 42.0

    def test_decimal(self):
        assert _parse_number("3.14") == pytest.approx(3.14)

    def test_negative(self):
        assert _parse_number("-5") == -5.0

    def test_strip_dollar(self):
        assert _parse_number("$50000") == 50000.0

    def test_strip_comma(self):
        assert _parse_number("50,000") == 50000.0

    def test_strip_percent(self):
        assert _parse_number("42%") == 42.0

    def test_combined_format(self):
        assert _parse_number("$1,250.50") == pytest.approx(1250.50)

    def test_extracts_first_number_from_text(self):
        # Model may produce extra text; we want the first number
        assert _parse_number("about 42 years") == 42.0

    def test_whitespace(self):
        assert _parse_number("  17  ") == 17.0

    def test_empty(self):
        assert _parse_number("") is None

    def test_none(self):
        assert _parse_number(None) is None

    def test_no_number_in_text(self):
        assert _parse_number("hello") is None


# ----------------------------- score() --------------------------------


class TestScoreFn:
    def test_exact_prediction(self):
        score_fn = continuous_scorer()
        result = _run(score_fn(_make_state("42"), _make_target("42")))
        assert result.metadata["prediction"] == 42.0
        assert result.metadata["target_value"] == 42.0
        assert result.metadata["error"] == 0.0

    def test_overpredict(self):
        score_fn = continuous_scorer()
        result = _run(score_fn(_make_state("50"), _make_target("40")))
        assert result.metadata["error"] == 10.0
        assert result.value == 10

    def test_underpredict_error_is_negative(self):
        score_fn = continuous_scorer()
        result = _run(score_fn(_make_state("30"), _make_target("40")))
        assert result.metadata["error"] == -10.0
        # Per-sample score value is absolute error
        assert result.value == 10

    def test_unparseable_completion(self):
        score_fn = continuous_scorer()
        result = _run(score_fn(_make_state("not a number"), _make_target("40")))
        assert result.metadata["prediction"] is None
        assert result.metadata["error"] is None

    def test_formatted_number(self):
        score_fn = continuous_scorer()
        result = _run(score_fn(_make_state("$65,000"), _make_target("60000")))
        assert result.metadata["prediction"] == 65000.0
        assert result.metadata["error"] == 5000.0


# ----------------------------- aggregate metrics ----------------------


class TestMetrics:
    def test_mae(self):
        scores = [
            _score(50, 40, 10),
            _score(35, 40, -5),
            _score(45, 40, 5),
        ]
        # MAE = mean(|10|, |-5|, |5|) = 20/3
        assert mae()(scores) == pytest.approx(20 / 3)

    def test_rmse(self):
        scores = [_score(50, 40, 10), _score(30, 40, -10)]
        # RMSE = sqrt(mean(100, 100)) = 10
        assert rmse()(scores) == pytest.approx(10.0)

    def test_r_squared_perfect(self):
        # ss_res = 0 -> R² = 1
        scores = [_score(10, 10, 0), _score(20, 20, 0), _score(30, 30, 0)]
        # ss_tot != 0 because targets vary
        assert r_squared()(scores) == pytest.approx(1.0)

    def test_r_squared_zero_when_predicting_mean(self):
        # Predict the target mean for every sample -> R² = 0
        # targets = [10, 20, 30], mean=20; predict 20 each
        scores = [
            _score(20, 10, 10),
            _score(20, 20, 0),
            _score(20, 30, -10),
        ]
        assert r_squared()(scores) == pytest.approx(0.0)

    def test_r_squared_negative_when_worse_than_mean(self):
        # Anti-correlated predictions should yield negative R²
        scores = [
            _score(30, 10, 20),
            _score(20, 20, 0),
            _score(10, 30, -20),
        ]
        val = r_squared()(scores)
        assert val < 0

    def test_r_squared_constant_targets_returns_nan(self):
        # ss_tot = 0 when all targets identical
        scores = [_score(5, 10, -5), _score(15, 10, 5)]
        val = r_squared()(scores)
        assert math.isnan(val)

    def test_parse_rate_full(self):
        scores = [_score(1, 1, 0), _score(2, 2, 0)]
        assert parse_rate()(scores) == 1.0

    def test_parse_rate_mixed(self):
        scores = [
            _score(1, 1, 0),
            _score(None, 1, None),
            _score(3, 1, 2),
            _score(None, 1, None),
        ]
        assert parse_rate()(scores) == 0.5

    def test_parse_rate_empty_returns_nan(self):
        val = parse_rate()([])
        assert math.isnan(val)

    def test_mae_skips_unparseable(self):
        scores = [
            _score(50, 40, 10),
            _score(None, 40, None),
            _score(45, 40, 5),
        ]
        # Only two scored: mean(|10|, |5|) = 7.5
        assert mae()(scores) == pytest.approx(7.5)


# ----------------------------- registry ------------------------------


class TestRegistry:
    def test_build_scorers_continuous(self):
        scorers = build_scorers({"scorer": [{"name": "continuous_scorer"}]})
        assert len(scorers) == 1
        # Smoke-check that it returns something callable-like (an inspect Scorer)
        assert scorers[0] is not None

    def test_build_scorers_continuous_with_empty_params(self):
        scorers = build_scorers(
            {"scorer": [{"name": "continuous_scorer", "params": {}}]}
        )
        assert len(scorers) == 1

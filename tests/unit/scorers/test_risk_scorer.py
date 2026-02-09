"""Unit tests for tools/inspect/scorers/risk_scorer.py

Tests the risk scorer against theoretical logprobs to verify that:
- Normalized probabilities are computed correctly from logprobs
- Binary risk scores match expected P(positive class)
- Multiclass returns option_probs but no scalar risk_score
- Edge cases (missing tokens, no logprobs) are handled gracefully
- The mean_risk_score metric aggregates correctly
"""

import math
import asyncio
import pytest
from unittest.mock import MagicMock

from inspect_ai.model._model_output import TopLogprob, Logprob, Logprobs, ChatCompletionChoice, ModelOutput
from inspect_ai.model._chat_message import ChatMessageAssistant
from inspect_ai.scorer import Score, CORRECT, INCORRECT, Target

from cruijff_kit.tools.inspect.scorers.risk_scorer import risk_scorer, mean_risk_score

# =============================================================================
# Helpers
# =============================================================================

def _make_state(completion: str, token_logprobs: dict[str, float], generated_token: str | None = None):
    """Build a mock TaskState with logprobs for the first generated token.

    Args:
        completion: The model's text completion (e.g., "0" or "1").
        token_logprobs: Dict of {token_str: logprob} to include in top_logprobs.
        generated_token: The actually-generated token. Defaults to completion.
    """
    if generated_token is None:
        generated_token = completion

    top_logprobs = [TopLogprob(token=t, logprob=lp) for t, lp in token_logprobs.items()]

    first_token = Logprob(
        token=generated_token,
        logprob=token_logprobs.get(generated_token, -10.0),
        top_logprobs=top_logprobs,
    )

    # Set message content = completion so ModelOutput.completion resolves
    # correctly whether it uses the field value or the set_completion validator
    message = ChatMessageAssistant(content=completion)

    logprobs = Logprobs(content=[first_token])
    choice = ChatCompletionChoice(
        message=message,
        logprobs=logprobs,
    )
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
# Binary classification tests
# =============================================================================

class TestBinaryRiskScore:
    """Tests for binary (2-class) risk scoring."""

    def test_confident_correct_prediction(self):
        """Model is 90% confident in class 0, and 0 is correct."""
        # P(0)=0.9, P(1)=0.1 in full vocab; logprobs reflect this
        lp_0 = math.log(0.9)   # -0.1054
        lp_1 = math.log(0.1)   # -2.3026
        state = _make_state("0", {"0": lp_0, "1": lp_1})
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT
        assert result.metadata["risk_score"] == pytest.approx(0.9, abs=1e-4)
        assert result.metadata["option_probs"]["0"] == pytest.approx(0.9, abs=1e-4)
        assert result.metadata["option_probs"]["1"] == pytest.approx(0.1, abs=1e-4)
        assert result.metadata["target"] == "0"

    def test_confident_incorrect_prediction(self):
        """Model is 90% confident in class 1, but 0 is correct."""
        lp_0 = math.log(0.1)
        lp_1 = math.log(0.9)
        state = _make_state("1", {"0": lp_0, "1": lp_1})
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT
        # risk_score = P(first option token) = P("0") ≈ 0.1
        assert result.metadata["risk_score"] == pytest.approx(0.1, abs=1e-4)

    def test_equal_logprobs_gives_50_50(self):
        """Equal logprobs for both classes -> risk_score = 0.5."""
        lp = math.log(0.5)
        state = _make_state("0", {"0": lp, "1": lp})
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT
        assert result.metadata["risk_score"] == pytest.approx(0.5, abs=1e-6)
        assert result.metadata["option_probs"]["0"] == pytest.approx(0.5, abs=1e-6)
        assert result.metadata["option_probs"]["1"] == pytest.approx(0.5, abs=1e-6)

    def test_renormalization_over_option_tokens(self):
        """Probabilities are renormalized over just the option tokens.

        If the model puts most mass on non-option tokens, the scorer
        should still produce a valid distribution over option tokens.
        E.g., P(0)=0.3, P(1)=0.1 in full vocab -> renormalized P(0)=0.75, P(1)=0.25
        """
        lp_0 = math.log(0.3)
        lp_1 = math.log(0.1)
        state = _make_state("0", {"0": lp_0, "1": lp_1, "other": math.log(0.6)})
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.metadata["risk_score"] == pytest.approx(0.75, abs=1e-4)
        assert result.metadata["option_probs"]["0"] == pytest.approx(0.75, abs=1e-4)
        assert result.metadata["option_probs"]["1"] == pytest.approx(0.25, abs=1e-4)

    def test_extreme_confidence(self):
        """Near-certain prediction: P(0) ≈ 0.9999."""
        lp_0 = math.log(0.9999)
        lp_1 = math.log(0.0001)
        state = _make_state("0", {"0": lp_0, "1": lp_1})
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.metadata["risk_score"] == pytest.approx(0.9999, abs=1e-4)

    def test_risk_score_is_always_first_option_token(self):
        """risk_score = P(first option token), regardless of which is 'correct'."""
        lp_0 = math.log(0.3)
        lp_1 = math.log(0.7)
        state = _make_state("1", {"0": lp_0, "1": lp_1})
        target = _make_target("1")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT
        # risk_score = P("0") = 0.3, not P(correct class)
        assert result.metadata["risk_score"] == pytest.approx(0.3, abs=1e-4)

    def test_probs_sum_to_one(self):
        """Normalized option_probs should always sum to 1."""
        lp_0 = math.log(0.6)
        lp_1 = math.log(0.25)
        state = _make_state("0", {"0": lp_0, "1": lp_1, "2": math.log(0.15)})
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        total = sum(result.metadata["option_probs"].values())
        assert total == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# Multiclass tests
# =============================================================================

class TestMulticlassRiskScore:
    """Tests for multiclass (>2 classes) scoring."""

    def test_multiclass_risk_score_is_none(self):
        """Multiclass tasks should return risk_score=None."""
        tokens = {"A": math.log(0.5), "B": math.log(0.3), "C": math.log(0.15), "D": math.log(0.05)}
        state = _make_state("A", tokens)
        target = _make_target("A")

        score_fn = risk_scorer(option_tokens=["A", "B", "C", "D"])
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT
        assert result.metadata["risk_score"] is None

    def test_multiclass_option_probs_correct(self):
        """Multiclass option_probs should be correctly renormalized."""
        tokens = {"A": math.log(0.5), "B": math.log(0.3), "C": math.log(0.2)}
        state = _make_state("A", tokens)
        target = _make_target("A")

        score_fn = risk_scorer(option_tokens=["A", "B", "C"])
        result: Score = _run(score_fn(state, target))

        assert result.metadata["option_probs"]["A"] == pytest.approx(0.5, abs=1e-4)
        assert result.metadata["option_probs"]["B"] == pytest.approx(0.3, abs=1e-4)
        assert result.metadata["option_probs"]["C"] == pytest.approx(0.2, abs=1e-4)

    def test_multiclass_probs_sum_to_one(self):
        """Multiclass option_probs should sum to 1."""
        tokens = {"A": math.log(0.4), "B": math.log(0.3), "C": math.log(0.2), "D": math.log(0.1)}
        state = _make_state("B", tokens)
        target = _make_target("A")

        score_fn = risk_scorer(option_tokens=["A", "B", "C", "D"])
        result: Score = _run(score_fn(state, target))

        total = sum(result.metadata["option_probs"].values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_multiclass_uniform_distribution(self):
        """Equal logprobs -> uniform distribution."""
        lp = math.log(0.25)
        tokens = {"A": lp, "B": lp, "C": lp, "D": lp}
        state = _make_state("A", tokens)
        target = _make_target("A")

        score_fn = risk_scorer(option_tokens=["A", "B", "C", "D"])
        result: Score = _run(score_fn(state, target))

        for token in ["A", "B", "C", "D"]:
            assert result.metadata["option_probs"][token] == pytest.approx(0.25, abs=1e-6)

    def test_multiclass_renormalization(self):
        """Multiclass renormalization when option tokens are a subset of vocab.

        Full vocab: P(A)=0.2, P(B)=0.1, P(C)=0.1, P(other)=0.6
        Renormalized: P(A)=0.5, P(B)=0.25, P(C)=0.25
        """
        tokens = {"A": math.log(0.2), "B": math.log(0.1), "C": math.log(0.1), "other": math.log(0.6)}
        state = _make_state("A", tokens)
        target = _make_target("A")

        score_fn = risk_scorer(option_tokens=["A", "B", "C"])
        result: Score = _run(score_fn(state, target))

        assert result.metadata["option_probs"]["A"] == pytest.approx(0.5, abs=1e-4)
        assert result.metadata["option_probs"]["B"] == pytest.approx(0.25, abs=1e-4)
        assert result.metadata["option_probs"]["C"] == pytest.approx(0.25, abs=1e-4)


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for error handling and edge cases."""

    def test_missing_option_token(self):
        """When an option token is not in top logprobs, return INCORRECT."""
        # Only "0" appears; "1" is missing from top logprobs
        state = _make_state("0", {"0": math.log(0.8), "foo": math.log(0.2)})
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT
        assert result.metadata["risk_score"] is None
        assert result.metadata["option_probs"] is None
        assert "1" in result.metadata["missing_tokens"]
        assert result.metadata["target"] == "0"

    def test_no_logprobs(self):
        """When no logprobs data is available, return INCORRECT."""
        choice = ChatCompletionChoice(message=ChatMessageAssistant(content="0"), logprobs=None)
        output = ModelOutput(choices=[choice], completion="0")

        state = MagicMock()
        state.output = output
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT
        assert result.metadata["risk_score"] is None
        assert result.metadata["target"] == "0"
        assert "No logprobs" in result.explanation

    def test_empty_logprobs_content(self):
        """When logprobs.content is empty, return INCORRECT."""
        logprobs = Logprobs(content=[])
        choice = ChatCompletionChoice(message=ChatMessageAssistant(content="0"), logprobs=logprobs)
        output = ModelOutput(choices=[choice], completion="0")

        state = MagicMock()
        state.output = output
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT
        assert result.metadata["risk_score"] is None

    def test_no_choices(self):
        """When output has no choices, return INCORRECT."""
        output = ModelOutput(choices=[], completion="0")

        state = MagicMock()
        state.output = output
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT
        assert result.metadata["risk_score"] is None

    def test_whitespace_token_stripping(self):
        """Tokens with whitespace should be stripped to match option tokens."""
        # Some models return " 0" instead of "0"
        top_logprobs = [
            TopLogprob(token=" 0", logprob=math.log(0.7)),
            TopLogprob(token=" 1", logprob=math.log(0.3)),
        ]
        first_token = Logprob(token=" 0", logprob=math.log(0.7), top_logprobs=top_logprobs)
        logprobs = Logprobs(content=[first_token])
        choice = ChatCompletionChoice(message=ChatMessageAssistant(content="0"), logprobs=logprobs)
        output = ModelOutput(choices=[choice], completion="0")

        state = MagicMock()
        state.output = output
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.metadata["risk_score"] == pytest.approx(0.7, abs=1e-4)

    def test_all_option_tokens_missing(self):
        """When no option tokens appear in top logprobs."""
        state = _make_state("X", {"X": math.log(0.5), "Y": math.log(0.5)})
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT
        assert result.metadata["risk_score"] is None
        assert set(result.metadata["missing_tokens"]) == {"0", "1"}


# =============================================================================
# Correctness logic
# =============================================================================

class TestCorrectnessScoring:
    """Tests that CORRECT/INCORRECT is based on completion matching target."""

    def test_correct_when_completion_matches_target(self):
        """CORRECT when model output matches target text."""
        state = _make_state("1", {"0": math.log(0.3), "1": math.log(0.7)})
        target = _make_target("1")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT

    def test_incorrect_when_completion_differs_from_target(self):
        """INCORRECT when model output doesn't match target."""
        state = _make_state("0", {"0": math.log(0.6), "1": math.log(0.4)})
        target = _make_target("1")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.value == INCORRECT

    def test_correctness_independent_of_risk_score(self):
        """Correctness is about the completion, not the probability."""
        # Model generates "0" (correct) but P("0") is only 0.3
        # (it generated the less-likely token)
        state = _make_state("0", {"0": math.log(0.3), "1": math.log(0.7)})
        target = _make_target("0")

        score_fn = risk_scorer(option_tokens=["0", "1"])
        result: Score = _run(score_fn(state, target))

        assert result.value == CORRECT
        assert result.metadata["risk_score"] == pytest.approx(0.3, abs=1e-4)


# =============================================================================
# mean_risk_score metric
# =============================================================================

class TestMeanRiskScoreMetric:
    """Tests for the mean_risk_score aggregation metric."""

    def test_basic_average(self):
        """Mean of several risk scores."""
        scores = [
            Score(value=CORRECT, metadata={"risk_score": 0.8}),
            Score(value=CORRECT, metadata={"risk_score": 0.6}),
            Score(value=INCORRECT, metadata={"risk_score": 0.4}),
        ]
        metric_fn = mean_risk_score()
        result = metric_fn(scores)
        assert result == pytest.approx(0.6, abs=1e-6)

    def test_skips_none_values(self):
        """None risk scores (from missing logprobs) are excluded."""
        scores = [
            Score(value=CORRECT, metadata={"risk_score": 0.8}),
            Score(value=INCORRECT, metadata={"risk_score": None}),
            Score(value=CORRECT, metadata={"risk_score": 0.6}),
        ]
        metric_fn = mean_risk_score()
        result = metric_fn(scores)
        assert result == pytest.approx(0.7, abs=1e-6)

    def test_all_none_returns_nan(self):
        """When all risk scores are None, return NaN."""
        scores = [
            Score(value=INCORRECT, metadata={"risk_score": None}),
            Score(value=INCORRECT, metadata={"risk_score": None}),
        ]
        metric_fn = mean_risk_score()
        result = metric_fn(scores)
        assert math.isnan(result)

    def test_single_score(self):
        """Single valid score returns that value."""
        scores = [Score(value=CORRECT, metadata={"risk_score": 0.73})]
        metric_fn = mean_risk_score()
        result = metric_fn(scores)
        assert result == pytest.approx(0.73, abs=1e-6)

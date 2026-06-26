"""Unit tests for tools/inspect/scorers/reasoning_risk_scorer.py

The reasoning_risk_scorer is risk_scorer made robust to <think>...</think>
reasoning blocks. These tests verify that:
- A direct answer (no think block) is scored at position 0, matching risk_scorer.
- A reasoning trace is skipped: logprobs are read at the first option token
  after the generated </think>, not at <think>.
- A digit appearing *inside* the think block is not mistaken for the answer.
- <think>...</think> is stripped from the completion before the accuracy check.
- Edge cases (no option token generated, no logprobs) are handled gracefully.
"""

import math
import asyncio
from unittest.mock import MagicMock

from inspect_ai.model._model_output import (
    TopLogprob,
    Logprob,
    Logprobs,
    ChatCompletionChoice,
    ModelOutput,
)
from inspect_ai.model._chat_message import ChatMessageAssistant
from inspect_ai.scorer import Score, CORRECT, INCORRECT, Target

import pytest

from cruijff_kit.tools.inspect.scorers.reasoning_risk_scorer import (
    reasoning_risk_scorer,
    _strip_think,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_state(completion: str, content_tokens: list[tuple[str, dict[str, float]]]):
    """Build a mock TaskState whose logprob stream is a list of generated tokens.

    Args:
        completion: The model's full text completion (may include a think block).
        content_tokens: Ordered list of (generated_token, {token: logprob}) — one
            entry per generated token. The generated token's own logprob is read
            from its top-logprob dict (default -10.0 if absent).
    """
    content = []
    for gen_tok, top in content_tokens:
        top_lps = [TopLogprob(token=t, logprob=lp) for t, lp in top.items()]
        content.append(
            Logprob(
                token=gen_tok,
                logprob=top.get(gen_tok, -10.0),
                top_logprobs=top_lps,
            )
        )

    message = ChatMessageAssistant(content=completion)
    logprobs = Logprobs(content=content)
    choice = ChatCompletionChoice(message=message, logprobs=logprobs)
    output = ModelOutput(choices=[choice], completion=completion)

    state = MagicMock()
    state.output = output
    return state


def _make_target(text: str) -> Target:
    return Target([text])


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# =============================================================================
# _strip_think helper
# =============================================================================


class TestStripThink:
    def test_strips_block_and_whitespace(self):
        assert _strip_think("<think>reasoning here</think>\n\n1") == "1"

    def test_strips_multiline_block(self):
        assert _strip_think("<think>\nline 1\nline 2\n</think>0") == "0"

    def test_no_block_is_unchanged(self):
        assert _strip_think("1") == "1"

    def test_empty_primed_block(self):
        # The block a thinking-off prompt primes, if echoed by the model.
        assert _strip_think("<think>\n\n</think>\n\n1") == "1"


# =============================================================================
# Direct answer (thinking off) — collapses to risk_scorer behavior
# =============================================================================


class TestDirectAnswer:
    def test_position_zero_when_no_think(self):
        """No think block: answer is the first token, like risk_scorer."""
        state = _make_state("0", [("0", {"0": math.log(0.9), "1": math.log(0.1)})])
        result: Score = _run(
            reasoning_risk_scorer(["0", "1"])(state, _make_target("0"))
        )

        assert result.value == CORRECT
        assert result.metadata["risk_score"] == pytest.approx(0.1, abs=1e-4)
        assert result.metadata["option_probs"]["0"] == pytest.approx(0.9, abs=1e-4)
        assert result.answer == "0"

    def test_whitespace_token_stripping(self):
        """Tokens like ' 1' still match option token '1'."""
        state = _make_state("1", [(" 1", {" 0": math.log(0.3), " 1": math.log(0.7)})])
        result: Score = _run(
            reasoning_risk_scorer(["0", "1"])(state, _make_target("1"))
        )

        assert result.value == CORRECT
        assert result.metadata["risk_score"] == pytest.approx(0.7, abs=1e-4)


# =============================================================================
# Reasoning trace (thinking on) — answer after </think>
# =============================================================================


class TestThinkBlockSkipping:
    def test_reads_answer_after_think(self):
        """Logprobs are read at the option token after </think>, not at <think>."""
        completion = "<think>profile suggests yes</think>\n\n1"
        state = _make_state(
            completion,
            [
                ("<think>", {"<think>": math.log(0.99)}),
                ("profile", {"profile": math.log(0.5)}),
                ("</think>", {"</think>": math.log(0.99)}),
                ("\n\n", {"\n\n": math.log(0.9)}),
                ("1", {"0": math.log(0.2), "1": math.log(0.8)}),
            ],
        )
        result: Score = _run(
            reasoning_risk_scorer(["0", "1"])(state, _make_target("1"))
        )

        assert result.value == CORRECT
        # risk_score read at the post-</think> answer token, P("1") = 0.8
        assert result.metadata["risk_score"] == pytest.approx(0.8, abs=1e-4)
        # completion text has the think block stripped
        assert result.answer == "1"

    def test_digit_inside_think_is_ignored(self):
        """A digit inside the reasoning trace must not be mistaken for the answer."""
        completion = "<think>maybe 1 child</think>0"
        state = _make_state(
            completion,
            [
                ("<think>", {"<think>": math.log(0.99)}),
                ("1", {"1": math.log(0.6)}),  # decoy digit inside the trace
                ("</think>", {"</think>": math.log(0.99)}),
                ("0", {"0": math.log(0.7), "1": math.log(0.3)}),
            ],
        )
        result: Score = _run(
            reasoning_risk_scorer(["0", "1"])(state, _make_target("0"))
        )

        assert result.value == CORRECT
        # Must read the post-</think> "0" token: P("1") = 0.3 (not the decoy)
        assert result.metadata["risk_score"] == pytest.approx(0.3, abs=1e-4)
        assert result.answer == "0"


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    def test_no_option_token_generated(self):
        """Reasoning truncated before answering: no option token -> INCORRECT."""
        state = _make_state(
            "<think>thinking...",
            [
                ("<think>", {"<think>": math.log(0.99)}),
                ("thinking", {"thinking": math.log(0.5)}),
            ],
        )
        result: Score = _run(
            reasoning_risk_scorer(["0", "1"])(state, _make_target("1"))
        )

        assert result.value == INCORRECT
        assert result.metadata["risk_score"] is None
        assert "No option token" in result.explanation

    def test_missing_option_token_at_answer_position(self):
        """Answer token present but the other option missing from top logprobs."""
        state = _make_state("0", [("0", {"0": math.log(0.8), "foo": math.log(0.2)})])
        result: Score = _run(
            reasoning_risk_scorer(["0", "1"])(state, _make_target("0"))
        )

        assert result.value == INCORRECT
        assert "1" in result.metadata["missing_tokens"]

    def test_no_logprobs(self):
        choice = ChatCompletionChoice(
            message=ChatMessageAssistant(content="0"), logprobs=None
        )
        output = ModelOutput(choices=[choice], completion="0")
        state = MagicMock()
        state.output = output

        result: Score = _run(
            reasoning_risk_scorer(["0", "1"])(state, _make_target("0"))
        )

        assert result.value == INCORRECT
        assert "No logprobs" in result.explanation

    def test_correctness_uses_stripped_completion(self):
        """Accuracy compares the think-stripped text to target."""
        completion = "<think>long winded reasoning</think>1"
        state = _make_state(
            completion,
            [
                ("<think>", {"<think>": math.log(0.99)}),
                ("</think>", {"</think>": math.log(0.99)}),
                ("1", {"0": math.log(0.4), "1": math.log(0.6)}),
            ],
        )
        result: Score = _run(
            reasoning_risk_scorer(["0", "1"])(state, _make_target("1"))
        )

        assert result.value == CORRECT
        assert result.answer == "1"


# =============================================================================
# Registry wiring
# =============================================================================


class TestRegistryWiring:
    def test_registered_and_requires_logprobs(self):
        from cruijff_kit.tools.inspect.scorers import (
            SCORER_REGISTRY,
            SCORER_FACTORIES,
            configured_scorers_require_logprobs,
        )

        assert "reasoning_risk_scorer" in SCORER_REGISTRY
        assert SCORER_FACTORIES["reasoning_risk_scorer"].requires_logprobs is True
        # build path opts the task into logprob capture
        cfg = {"scorers": [{"name": "reasoning_risk_scorer"}]}
        assert configured_scorers_require_logprobs(cfg) is True

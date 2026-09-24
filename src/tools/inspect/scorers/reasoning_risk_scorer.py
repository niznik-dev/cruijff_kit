"""
inspect-ai risk scorer robust to reasoning ("thinking") models.

Reasoning models such as Qwen3 may emit a ``<think>...</think>`` block before
the actual answer. The plain ``risk_scorer`` reads the *first* generated token,
which for such models is ``<think>`` rather than the answer, so its probabilities
and exact-match accuracy are both wrong.

This scorer is identical to ``risk_scorer`` in output (same metric suite, same
``Score.metadata`` keys) but locates the answer token before reading logprobs:

- If a ``</think>`` token was generated, the answer is the first option token
  *after* it (the reasoning trace is skipped).
- Otherwise the answer is the first option token from the start, which collapses
  to position 0 for a direct answer.

The completion text is likewise stripped of ``<think>...</think>`` before the
exact-match accuracy check. One scorer therefore handles both thinking-on and
thinking-off generations.
"""

import math
import re

from inspect_ai.scorer import scorer, Score, CORRECT, INCORRECT, Target
from inspect_ai.solver import TaskState

from .risk_scorer import risk_metric_suite

# A full <think>...</think> block including the tags, non-greedy and across
# newlines, plus any trailing whitespace. Also matches the empty block a
# thinking-off prompt primes, should the model echo it.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove any <think>...</think> block(s) and surrounding whitespace."""
    return _THINK_BLOCK.sub("", text).strip()


def _answer_index(content, option_tokens) -> int | None:
    """Index of the answer token in the generated logprob stream.

    Skips a reasoning trace by starting the search after the last generated
    ``</think>`` token (if any), then returns the first token whose stripped
    text is an option token. Returns None if no option token was generated.
    """
    option_set = {t.strip() for t in option_tokens}

    start = 0
    for i, entry in enumerate(content):
        if "</think>" in entry.token:
            start = i + 1

    for i in range(start, len(content)):
        if content[i].token.strip() in option_set:
            return i
    return None


@scorer(metrics=risk_metric_suite())
def reasoning_risk_scorer(option_tokens: list[str] = ("0", "1")):
    """Risk scorer that tolerates <think> reasoning blocks. See module docstring.

    - Requires GenerateConfig(logprobs=True, top_logprobs>=2) on the Task.
    - Publishes the identical metric suite as risk_scorer.

    Args:
        option_tokens: The target answer tokens (e.g., ["0", "1"] or ["A", "B", "C", "D"]).
    """

    async def score(state: TaskState, target: Target) -> Score:
        answer = _strip_think(state.output.completion)

        choice = state.output.choices[0] if state.output.choices else None
        logprobs_data = choice.logprobs if choice else None

        if not logprobs_data or not logprobs_data.content:
            return Score(
                value=INCORRECT,
                answer=answer,
                explanation="No logprobs available",
                metadata={"risk_score": None, "target": target.text},
            )

        idx = _answer_index(logprobs_data.content, option_tokens)
        if idx is None:
            return Score(
                value=INCORRECT,
                answer=answer,
                explanation="No option token found in generated output",
                metadata={
                    "risk_score": None,
                    "option_probs": None,
                    "target": target.text,
                },
            )

        answer_token_logprob = logprobs_data.content[idx]

        # Build map: token_str -> logprob from top_logprobs + the chosen token itself
        token_logprob_map = {}
        if answer_token_logprob.top_logprobs:
            for tlp in answer_token_logprob.top_logprobs:
                token_logprob_map[tlp.token.strip()] = tlp.logprob
        token_logprob_map[answer_token_logprob.token.strip()] = (
            answer_token_logprob.logprob
        )

        option_logprobs = {}
        missing = []
        for token in option_tokens:
            lp = token_logprob_map.get(token)
            if lp is None:
                missing.append(token)
            else:
                option_logprobs[token] = lp

        if missing:
            return Score(
                value=INCORRECT,
                answer=answer,
                explanation=f"Option token(s) {missing} not in top logprobs",
                metadata={
                    "risk_score": None,
                    "option_probs": None,
                    "missing_tokens": missing,
                    "available_tokens": list(token_logprob_map.keys()),
                    "target": target.text,
                },
            )

        # Softmax over option tokens
        max_lp = max(option_logprobs.values())
        exp_values = {t: math.exp(lp - max_lp) for t, lp in option_logprobs.items()}
        total = sum(exp_values.values())
        probs = {t: v / total for t, v in exp_values.items()}

        # Risk score: P(last option token) — the positive class probability
        risk_score = probs[option_tokens[-1]] if len(option_tokens) == 2 else None

        correct = answer == target.text if target.text else False

        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=answer,
            metadata={
                "risk_score": risk_score,
                "option_probs": probs,
                "target": target.text,
            },
        )

    return score


# Capability marker: the unified task auto-enables logprobs capture when a
# configured scorer declares this attribute. Set on the factory (not the inner
# score function) so the task can introspect by name lookup without instantiating.
reasoning_risk_scorer.requires_logprobs = True

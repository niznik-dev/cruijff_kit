"""
inspect-ai scorer that computes risk scores from logprobs.

Computes normalized probabilities over option tokens from the first generated
token's top_logprobs. For binary classification, also computes a risk score
= P(positive token) where the positive class is the last in option_tokens
(e.g., "1" in ["0", "1"]).
"""
import math
from inspect_ai.scorer import scorer, Score, metric, Metric, CORRECT, INCORRECT
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Target
from .calibration_metrics import expected_calibration_error, risk_calibration_error, brier_score, auc_score

@metric
def mean_risk_score() -> Metric:
    """Average risk score across all samples."""
    def compute(scores: list[Score]) -> float:
        values = [s.metadata["risk_score"] for s in scores if s.metadata.get("risk_score") is not None]
        return sum(values) / len(values) if values else float("nan")
    return compute

@scorer(metrics=[mean_risk_score(), expected_calibration_error(), risk_calibration_error(), brier_score(), auc_score()])
def risk_scorer(option_tokens: list[str] = ("0", "1")):
    """
    Scorer that extracts risk scores from logprobs of the first generated token.
    - Requires GenerateConfig(logprobs=True, top_logprobs=20) on the Task.
    - Supports any number of option tokens (binary or multiclass).
    - For binary tasks, returns a risk_score = P(last/positive option token).
    - For all tasks, returns normalized probabilities over all option tokens.

    Args:
        option_tokens: The target answer tokens (e.g., ["0", "1"] or ["A", "B", "C", "D"]).
    """
    async def score(state: TaskState, target: Target) -> Score:
        # Extract logprobs from first generated token
        choice = state.output.choices[0] if state.output.choices else None
        logprobs_data = choice.logprobs if choice else None

        if not logprobs_data or not logprobs_data.content:
            return Score(
                value=INCORRECT,
                answer=state.output.completion.strip(),
                explanation="No logprobs available",
                metadata={"risk_score": None, "target": target.text}
            )

        first_token_logprob = logprobs_data.content[0]

        # Build map: token_str -> logprob from top_logprobs + the chosen token itself
        token_logprob_map = {}
        if first_token_logprob.top_logprobs:
            for tlp in first_token_logprob.top_logprobs:
                token_logprob_map[tlp.token.strip()] = tlp.logprob
        # Also include the actually-generated token
        token_logprob_map[first_token_logprob.token.strip()] = first_token_logprob.logprob

        option_logprobs = {}
        missing = []
        for token in option_tokens: 
            # get the log probabilities
            lp = token_logprob_map.get(token)
            if lp is None:
                missing.append(token)
            else:
                option_logprobs[token] = lp
        
        if missing:
            return Score(
                value=INCORRECT,
                answer=state.output.completion.strip(),
                explanation=f"Option token(s) {missing} not in top logprobs",
                metadata={
                    "risk_score": None,
                    "option_probs": None,
                    "missing_tokens": missing,
                    "available_tokens": list(token_logprob_map.keys()),
                    "target": target.text,
                }
            )

        # Softmax over option tokens
        max_lp = max(option_logprobs.values())
        exp_values = {t: math.exp(lp - max_lp) for t, lp in option_logprobs.items()}
        total = sum(exp_values.values())
        probs = {t: v / total for t, v in exp_values.items()}

        # Risk score: P(last option token) — the positive class probability
        # option_tokens=("0","1"), risk = P("1")
        risk_score = probs[option_tokens[-1]] if len(option_tokens) == 2 else None

        # Check correctness (match target)
        answer = state.output.completion.strip()
        correct = answer == target.text if target.text else False

        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=answer,
            metadata={
                "risk_score": risk_score,
                "option_probs": probs,
                "target": target.text,
            }
        )

    return score
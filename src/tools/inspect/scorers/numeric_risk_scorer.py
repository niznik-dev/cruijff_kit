"""
inspect-ai scorer that parses a numeric risk score from model text output.

For tasks where the model outputs a probability directly as text (e.g., "0.73"),
this scorer parses the value into a float and constructs synthetic option_probs
so that downstream calibration metrics work identically to risk_scorer.

Correctness is determined by thresholding at 0.5: if risk_score >= 0.5 the
predicted label is the positive (last) label, otherwise the negative (first).
"""

from inspect_ai.scorer import scorer, Score, CORRECT, INCORRECT
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Target
from .risk_scorer import mean_risk_score
from .calibration_metrics import (
    expected_calibration_error,
    risk_calibration_error,
    brier_score,
    auc_score,
)


def _parse_risk_score(text: str) -> float | None:
    """Parse a risk score from model output text.

    Strips whitespace, attempts float conversion, and validates the value
    is in [0, 1]. Returns None if parsing fails or value is out of range.
    """
    try:
        value = float(text.strip())
    except (ValueError, AttributeError):
        return None
    if value < 0.0 or value > 1.0:
        return None
    return value


@scorer(
    metrics=[
        mean_risk_score(),
        expected_calibration_error(),
        risk_calibration_error(),
        brier_score(),
        auc_score(),
    ]
)
def numeric_risk_scorer(labels: tuple[str, str] = ("0", "1")):
    """
    Scorer that parses a numeric probability from the model's text output.

    The model is expected to output a single float in [0, 1] representing
    P(positive class). The positive class is labels[-1] (default "1").

    Constructs synthetic option_probs = {labels[0]: 1-risk, labels[-1]: risk}
    so that all calibration metrics (ECE, Brier, AUC) work identically to
    risk_scorer.

    Args:
        labels: Tuple of (negative_label, positive_label). Default ("0", "1").
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        risk_score = _parse_risk_score(completion)

        if risk_score is None:
            return Score(
                value=INCORRECT,
                answer=completion.strip() if completion else "",
                explanation="Could not parse a numeric risk score from output",
                metadata={
                    "risk_score": None,
                    "option_probs": None,
                    "target": target.text,
                },
            )

        # Build synthetic option_probs
        option_probs = {labels[0]: 1.0 - risk_score, labels[-1]: risk_score}

        # Threshold at 0.5 to determine predicted label
        predicted_label = labels[-1] if risk_score >= 0.5 else labels[0]
        correct = predicted_label == target.text

        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=str(risk_score),
            metadata={
                "risk_score": risk_score,
                "option_probs": option_probs,
                "target": target.text,
            },
        )

    return score

"""
inspect-ai scorer for continuous/regression predictions.

For tasks where the model outputs a numeric value (e.g., age, income, hours),
this scorer parses the number and computes regression metrics across samples.

Metrics:
    - mae: Mean Absolute Error
    - rmse: Root Mean Squared Error
    - r_squared: Coefficient of determination (R²=0 means predicting the mean)
    - parse_rate: Fraction of outputs successfully parsed as numbers
"""
import math
import re
from inspect_ai.scorer import scorer, Score, metric, Metric, CORRECT, INCORRECT
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Target


def _parse_number(text: str) -> float | None:
    """Parse a number from model output text.

    Strips whitespace, commas, dollar signs, percent signs, then extracts
    the first number found. Returns None if parsing fails.
    """
    if not text:
        return None
    text = text.strip().replace(",", "").replace("$", "").replace("%", "")
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        return float(match.group())
    return None


@metric
def mae() -> Metric:
    """Mean Absolute Error across all scored samples."""
    def compute(scores: list[Score]) -> float:
        errors = [
            abs(s.metadata["error"])
            for s in scores
            if (s.metadata or {}).get("error") is not None
        ]
        return sum(errors) / len(errors) if errors else float("nan")

    return compute


@metric
def rmse() -> Metric:
    """Root Mean Squared Error across all scored samples."""
    def compute(scores: list[Score]) -> float:
        errors = [
            s.metadata["error"]
            for s in scores
            if (s.metadata or {}).get("error") is not None
        ]
        if not errors:
            return float("nan")
        return math.sqrt(sum(e ** 2 for e in errors) / len(errors))

    return compute


@metric
def r_squared() -> Metric:
    """R-squared (coefficient of determination).

    R²=0 means the model is no better than predicting the training set mean.
    R²=1 means perfect prediction. R²<0 means worse than the mean baseline.
    """
    def compute(scores: list[Score]) -> float:
        pairs = [
            (s.metadata["target_value"], s.metadata["prediction"])
            for s in scores
            if (s.metadata or {}).get("prediction") is not None
        ]
        if len(pairs) < 2:
            return float("nan")

        targets = [t for t, _ in pairs]
        predictions = [p for _, p in pairs]
        target_mean = sum(targets) / len(targets)

        ss_res = sum((t - p) ** 2 for t, p in pairs)
        ss_tot = sum((t - target_mean) ** 2 for t in targets)

        if ss_tot == 0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    return compute


@metric
def parse_rate() -> Metric:
    """Fraction of outputs that could be parsed as a number."""
    def compute(scores: list[Score]) -> float:
        if not scores:
            return float("nan")
        parsed = sum(
            1 for s in scores
            if (s.metadata or {}).get("prediction") is not None
        )
        return parsed / len(scores)

    return compute


@scorer(metrics=[mae(), rmse(), r_squared(), parse_rate()])
def continuous_scorer():
    """
    Scorer for continuous/regression predictions.

    Parses a number from model output, computes error vs target.
    Individual errors stored in metadata; aggregate metrics (MAE, RMSE, R²)
    computed across all samples.
    """
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        prediction = _parse_number(completion)
        target_value = _parse_number(target.text)

        if prediction is None or target_value is None:
            return Score(
                value=INCORRECT,
                answer=completion.strip() if completion else "",
                explanation="Could not parse number from output",
                metadata={
                    "prediction": None,
                    "target_value": target_value,
                    "error": None,
                }
            )

        error = prediction - target_value

        return Score(
            value=abs(error),
            answer=str(prediction),
            metadata={
                "prediction": prediction,
                "target_value": target_value,
                "error": error,
            }
        )

    return score

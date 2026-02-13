"""
Calibration and discrimination metrics for inspect-ai risk scoring.

These are aggregate @metric functions that consume Score.metadata produced by
risk_scorer: risk_score (P of first option token), option_probs (full
distribution), and target (ground truth label).

Metrics:
    - expected_calibration_error: Measures calibration via binned confidence
    - brier_score: Proper scoring rule for binary probabilistic predictions
    - auc_score: Area under ROC curve for binary discrimination
"""

from inspect_ai.scorer import metric, Score, Metric, value_to_float
from sklearn.metrics import brier_score_loss, roc_auc_score


def _binned_ece(predicted: list[float], actual: list[float], n_bins: int) -> float:
    """Shared ECE computation: weighted average of |avg_predicted - avg_actual| per bin.

    Args:
        predicted: Model's predicted values (confidence or risk score).
        actual: Ground truth (correct/incorrect for confidence, Y label for risk).
        n_bins: Number of equal-width bins over [0, 1].
    """
    if not predicted:
        return float("nan")

    n = len(predicted)
    bin_width = 1.0 / n_bins
    ece = 0.0

    for i in range(n_bins):
        lo = i * bin_width
        hi = lo + bin_width
        # Include right edge in the last bin
        in_bin = [
            (p, a) for p, a in zip(predicted, actual)
            if (lo < p <= hi) or (i == 0 and p == 0.0)
        ]
        if not in_bin:
            continue
        bin_pred = sum(p for p, _ in in_bin) / len(in_bin)
        bin_actual = sum(a for _, a in in_bin) / len(in_bin)
        ece += len(in_bin) / n * abs(bin_pred - bin_actual)

    return ece


@metric
def expected_calibration_error(n_bins: int = 10) -> Metric:
    """Confidence ECE: bins by max(option_probs), checks correct/incorrect.

    "When the model is 80% confident, is it right 80% of the time?"

    Uses option_probs and Score.value for accuracy. Score.value is converted
    via value_to_float() to handle both pre-reduction ("C"/"I") and
    post-reduction (1.0/0.0) representations.
    """
    def compute(scores: list[Score]) -> float:
        vtf = value_to_float()
        confidences = []
        accuracies = []
        for s in scores:
            option_probs = (s.metadata or {}).get("option_probs")
            if option_probs is None:
                continue
            confidences.append(max(option_probs.values()))
            accuracies.append(vtf(s.value))

        return _binned_ece(confidences, accuracies, n_bins)

    return compute


@metric
def risk_calibration_error(n_bins: int = 10) -> Metric:
    """Risk ECE: bins by P(Y=1) (risk_score), checks actual Y label.

    "When the model says P(income>50K) = 0.8, is it actually >50K 80% of the time?"

    Uses risk_score and target from Score.metadata. Does not depend on
    Score.value at all, so it is immune to the inspect-ai value-reduction bug.
    Binary tasks only (requires risk_score).
    """
    def compute(scores: list[Score]) -> float:
        risks = []
        actuals = []
        for s in scores:
            meta = s.metadata or {}
            risk = meta.get("risk_score")
            target = meta.get("target")
            option_probs = meta.get("option_probs")
            if risk is None or target is None or option_probs is None:
                continue
            positive_token = next(iter(option_probs))
            risks.append(risk)
            actuals.append(1.0 if target == positive_token else 0.0)

        return _binned_ece(risks, actuals, n_bins)

    return compute


@metric
def brier_score() -> Metric:
    """Brier Score for binary predictions.

    Computes mean squared error between risk_score (predicted probability of
    first option token) and the binary indicator y_i = 1 if target matches
    first option token, else 0.

    Requires risk_score and target in Score.metadata (binary tasks only).
    Returns NaN for multiclass tasks where risk_score is not set.
    """
    def compute(scores: list[Score]) -> float:
        y_true = []
        y_prob = []
        for s in scores:
            meta = s.metadata or {}
            risk = meta.get("risk_score")
            target = meta.get("target")
            if risk is None or target is None:
                continue
            # risk_score = P(first option token), so y=1 when target IS the first option token
            # We don't have option_tokens here, but risk_score is defined as P(token_0),
            # so y=1 iff target == the token whose probability risk_score represents.
            # The option_probs keys are ordered; first key = first option token.
            option_probs = meta.get("option_probs")
            if option_probs is None:
                continue
            # First key = first option token (dict order matches risk_scorer's option_tokens list)
            positive_token = next(iter(option_probs))
            y_true.append(1.0 if target == positive_token else 0.0)
            y_prob.append(risk)

        if len(y_true) < 2:
            return float("nan")

        return float(brier_score_loss(y_true, y_prob))

    return compute


@metric
def auc_score() -> Metric:
    """Area Under ROC Curve for binary predictions.

    Uses risk_score as the predicted probability and target to derive
    binary labels (1 if target == first option token, else 0).

    Returns NaN if only one class is present, fewer than 2 samples, or
    for multiclass tasks where risk_score is not set.
    """
    def compute(scores: list[Score]) -> float:
        y_true = []
        y_score = []
        for s in scores:
            meta = s.metadata or {}
            risk = meta.get("risk_score")
            target = meta.get("target")
            if risk is None or target is None:
                continue
            option_probs = meta.get("option_probs")
            if option_probs is None:
                continue
            # First key = first option token (dict order matches risk_scorer's option_tokens list)
            positive_token = next(iter(option_probs))
            y_true.append(1.0 if target == positive_token else 0.0)
            y_score.append(risk)

        if len(y_true) < 2 or len(set(y_true)) < 2:
            return float("nan")

        return float(roc_auc_score(y_true, y_score))

    return compute

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

from inspect_ai.scorer import metric, Score, Metric, CORRECT
from sklearn.metrics import brier_score_loss, roc_auc_score


@metric
def expected_calibration_error(n_bins: int = 10) -> Metric:
    """Expected Calibration Error (ECE).

    Bins samples by confidence (max option probability), then computes the
    weighted average of |avg_confidence - avg_accuracy| per bin.

    Uses option_probs and CORRECT/INCORRECT from Score — does NOT need target.
    """
    def compute(scores: list[Score]) -> float:
        confidences = []
        accuracies = []
        for s in scores:
            option_probs = (s.metadata or {}).get("option_probs")
            if option_probs is None:
                continue
            confidences.append(max(option_probs.values()))
            accuracies.append(1.0 if s.value == CORRECT else 0.0)

        if not confidences:
            return float("nan")

        n = len(confidences)
        bin_width = 1.0 / n_bins
        ece = 0.0

        for i in range(n_bins):
            lo = i * bin_width
            hi = lo + bin_width
            # Include right edge in the last bin
            in_bin = [
                (c, a) for c, a in zip(confidences, accuracies)
                if (lo < c <= hi) or (i == 0 and c == 0.0)
            ]
            if not in_bin:
                continue
            bin_conf = sum(c for c, _ in in_bin) / len(in_bin)
            bin_acc = sum(a for _, a in in_bin) / len(in_bin)
            ece += len(in_bin) / n * abs(bin_conf - bin_acc)

        return ece

    return compute


@metric
def brier_score() -> Metric:
    """Brier Score for binary predictions.

    Computes mean squared error between risk_score (predicted probability of
    first option token) and the binary indicator y_i = 1 if target matches
    first option token, else 0.

    Requires risk_score and target in Score.metadata (binary tasks only).
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
            first_token = next(iter(option_probs))
            y_true.append(1.0 if target == first_token else 0.0)
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

    Returns NaN if only one class is present or fewer than 2 samples.
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
            first_token = next(iter(option_probs))
            y_true.append(1.0 if target == first_token else 0.0)
            y_score.append(risk)

        if len(y_true) < 2 or len(set(y_true)) < 2:
            return float("nan")

        return float(roc_auc_score(y_true, y_score))

    return compute

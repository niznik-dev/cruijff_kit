"""Unit tests for tools/inspect/scorers/calibration_metrics.py

Tests ECE, Brier Score, and AUC metrics that aggregate over Score objects
produced by risk_scorer.
"""

import math
import pytest
from inspect_ai.scorer import Score, CORRECT, INCORRECT

from cruijff_kit.tools.inspect.scorers.calibration_metrics import (
    expected_calibration_error,
    risk_calibration_error,
    brier_score,
    auc_score,
)


# =============================================================================
# Helpers
# =============================================================================

def _score(value, risk_score, option_probs, target):
    """Build a Score with risk_scorer-style metadata."""
    return Score(
        value=value,
        metadata={
            "risk_score": risk_score,
            "option_probs": option_probs,
            "target": target,
        },
    )


# =============================================================================
# Expected Calibration Error
# =============================================================================

class TestExpectedCalibrationError:

    def test_perfect_calibration(self):
        """When confidence == accuracy in every bin, ECE = 0."""
        # 10 samples, all with confidence 0.8 and 80% correct
        scores = []
        for i in range(10):
            correct = CORRECT if i < 8 else INCORRECT
            scores.append(_score(correct, 0.8, {"0": 0.8, "1": 0.2}, "0"))

        metric_fn = expected_calibration_error(n_bins=10)
        result = metric_fn(scores)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_worst_case_calibration(self):
        """All predictions confident but wrong -> high ECE."""
        # All samples: confidence=1.0 but all incorrect
        scores = [_score(INCORRECT, 1.0, {"0": 1.0, "1": 0.0}, "1") for _ in range(10)]

        metric_fn = expected_calibration_error(n_bins=10)
        result = metric_fn(scores)
        # All in bin (0.9, 1.0], conf=1.0, acc=0.0 -> ECE=1.0
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_known_ece_value(self):
        """Manually computed ECE with two distinct bins."""
        # Confidence = max(option_probs), so:
        # 5 samples: option_probs={"0": 0.7, "1": 0.3} -> conf=0.7, all correct
        #   -> bin (0.6, 0.7]: conf=0.7, acc=1.0
        # 5 samples: option_probs={"0": 0.9, "1": 0.1} -> conf=0.9, all incorrect
        #   -> bin (0.8, 0.9]: conf=0.9, acc=0.0
        scores = []
        for _ in range(5):
            scores.append(_score(CORRECT, 0.7, {"0": 0.7, "1": 0.3}, "0"))
        for _ in range(5):
            scores.append(_score(INCORRECT, 0.9, {"0": 0.9, "1": 0.1}, "1"))

        metric_fn = expected_calibration_error(n_bins=10)
        result = metric_fn(scores)
        # ECE = 5/10 * |0.7 - 1.0| + 5/10 * |0.9 - 0.0| = 0.5*0.3 + 0.5*0.9 = 0.6
        assert result == pytest.approx(0.6, abs=1e-6)

    def test_skips_none_option_probs(self):
        """Samples with None option_probs are skipped."""
        scores = [
            _score(CORRECT, 0.8, {"0": 0.8, "1": 0.2}, "0"),
            Score(value=INCORRECT, metadata={"risk_score": None, "option_probs": None, "target": "0"}),
        ]
        metric_fn = expected_calibration_error(n_bins=10)
        # Only 1 sample: conf=0.8, acc=1.0 -> in bin (0.7, 0.8], ECE = |0.8 - 1.0| = 0.2
        result = metric_fn(scores)
        assert result == pytest.approx(0.2, abs=1e-6)

    def test_empty_scores_returns_nan(self):
        """No valid scores -> NaN."""
        metric_fn = expected_calibration_error(n_bins=10)
        assert math.isnan(metric_fn([]))

    def test_post_reduction_float_values(self):
        """ECE works when Score.value is float (post-reducer) instead of string.

        inspect-ai's mean_score() reducer converts "C"->1.0 and "I"->0.0
        before passing scores to @metric functions. This test ensures ECE
        handles both representations correctly.
        """
        # Same setup as test_known_ece_value but with float values
        scores = []
        for _ in range(5):
            scores.append(_score(1.0, 0.7, {"0": 0.7, "1": 0.3}, "0"))
        for _ in range(5):
            scores.append(_score(0.0, 0.9, {"0": 0.9, "1": 0.1}, "1"))

        metric_fn = expected_calibration_error(n_bins=10)
        result = metric_fn(scores)
        # ECE = 5/10 * |0.7 - 1.0| + 5/10 * |0.9 - 0.0| = 0.6
        assert result == pytest.approx(0.6, abs=1e-6)

    def test_string_and_float_values_agree(self):
        """ECE gives same result for string ("C"/"I") and float (1.0/0.0) values."""
        string_scores = []
        float_scores = []
        for i in range(10):
            correct_str = CORRECT if i < 8 else INCORRECT
            correct_flt = 1.0 if i < 8 else 0.0
            string_scores.append(_score(correct_str, 0.8, {"0": 0.8, "1": 0.2}, "0"))
            float_scores.append(_score(correct_flt, 0.8, {"0": 0.8, "1": 0.2}, "0"))

        metric_fn = expected_calibration_error(n_bins=10)
        assert metric_fn(string_scores) == pytest.approx(metric_fn(float_scores), abs=1e-6)


# =============================================================================
# Risk Calibration Error
# =============================================================================

class TestRiskCalibrationError:

    def test_perfect_risk_calibration(self):
        """When risk_score matches actual base rate in every bin, risk ECE = 0."""
        # 10 samples with risk=0.8, 8 of which are truly positive (target="1")
        # positive_token = last key = "1", so y_true=1 when target=="1"
        scores = []
        for i in range(10):
            target = "1" if i < 8 else "0"  # 80% positive
            correct = CORRECT if target == "1" else INCORRECT
            scores.append(_score(correct, 0.8, {"0": 0.2, "1": 0.8}, target))

        metric_fn = risk_calibration_error(n_bins=10)
        result = metric_fn(scores)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_worst_case_risk_calibration(self):
        """risk=1.0 but all samples are actually negative -> risk ECE=1.0."""
        # risk=1.0 means model predicts P(positive)=1.0, but all targets are "0" (negative)
        # positive_token = "1", target="0" -> y_true=0.0 for all
        scores = [_score(INCORRECT, 1.0, {"0": 0.0, "1": 1.0}, "0") for _ in range(10)]

        metric_fn = risk_calibration_error(n_bins=10)
        result = metric_fn(scores)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_known_risk_ece_value(self):
        """Manually computed risk ECE with two distinct bins."""
        # positive_token = last key = "1"
        # 5 samples: risk=0.7, all truly positive (target="1")
        #   -> bin (0.6, 0.7]: avg_risk=0.7, avg_actual=1.0
        # 5 samples: risk=0.3, all truly negative (target="0")
        #   -> bin (0.2, 0.3]: avg_risk=0.3, avg_actual=0.0
        scores = []
        for _ in range(5):
            scores.append(_score(CORRECT, 0.7, {"0": 0.3, "1": 0.7}, "1"))
        for _ in range(5):
            scores.append(_score(CORRECT, 0.3, {"0": 0.7, "1": 0.3}, "0"))

        metric_fn = risk_calibration_error(n_bins=10)
        result = metric_fn(scores)
        # ECE = 5/10 * |0.7 - 1.0| + 5/10 * |0.3 - 0.0| = 0.15 + 0.15 = 0.3
        assert result == pytest.approx(0.3, abs=1e-6)

    def test_differs_from_confidence_ece(self):
        """Risk ECE and confidence ECE give different values on the same data."""
        # P(Y=1)=0.3 -> confidence=0.7, risk=0.3 — different bins!
        # 5 samples: risk=0.3, truly negative, model predicts 0 correctly
        # 5 samples: risk=0.7, truly positive, model predicts 0 incorrectly
        scores = []
        for _ in range(5):
            # risk=0.3, target="1" (negative), predicted "1" correctly
            scores.append(_score(CORRECT, 0.3, {"0": 0.3, "1": 0.7}, "1"))
        for _ in range(5):
            # risk=0.7, target="0" (positive), predicted "1" incorrectly
            scores.append(_score(INCORRECT, 0.7, {"0": 0.7, "1": 0.3}, "0"))

        risk_ece = risk_calibration_error(n_bins=10)(scores)
        conf_ece = expected_calibration_error(n_bins=10)(scores)
        assert risk_ece != pytest.approx(conf_ece, abs=1e-3)

    def test_does_not_use_score_value(self):
        """Risk ECE is immune to Score.value bugs — same result regardless of value."""
        base_args = dict(risk_score=0.8, option_probs={"0": 0.8, "1": 0.2}, target="0")
        scores_correct = [_score(CORRECT, **base_args) for _ in range(10)]
        scores_wrong = [_score(INCORRECT, **base_args) for _ in range(10)]
        scores_float = [_score(0.0, **base_args) for _ in range(10)]  # bogus float

        metric_fn = risk_calibration_error(n_bins=10)
        assert metric_fn(scores_correct) == pytest.approx(metric_fn(scores_wrong), abs=1e-6)
        assert metric_fn(scores_correct) == pytest.approx(metric_fn(scores_float), abs=1e-6)

    def test_empty_scores_returns_nan(self):
        metric_fn = risk_calibration_error(n_bins=10)
        assert math.isnan(metric_fn([]))

    def test_skips_none_risk_score(self):
        """Samples with None risk_score are skipped."""
        # positive_token = "1", target="1" -> y_true=1.0
        scores = [
            _score(CORRECT, 0.8, {"0": 0.2, "1": 0.8}, "1"),
            Score(value=INCORRECT, metadata={"risk_score": None, "option_probs": None, "target": "0"}),
        ]
        metric_fn = risk_calibration_error(n_bins=10)
        # 1 sample: risk=0.8, actual=1.0 -> |0.8-1.0|=0.2
        result = metric_fn(scores)
        assert result == pytest.approx(0.2, abs=1e-6)


# =============================================================================
# Brier Score
# =============================================================================

class TestBrierScore:

    def test_perfect_predictions(self):
        """Risk score perfectly matches outcome -> Brier = 0."""
        # positive_token = "1"; risk_score = P("1")
        scores = [
            _score(CORRECT, 1.0, {"0": 0.0, "1": 1.0}, "1"),  # y=1, p=1.0
            _score(CORRECT, 0.0, {"0": 1.0, "1": 0.0}, "0"),  # y=0, p=0.0
        ]
        metric_fn = brier_score()
        result = metric_fn(scores)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_worst_predictions(self):
        """Risk score maximally wrong -> Brier = 1.0."""
        scores = [
            _score(INCORRECT, 0.0, {"0": 1.0, "1": 0.0}, "1"),  # y=1, p=0.0
            _score(INCORRECT, 1.0, {"0": 0.0, "1": 1.0}, "0"),  # y=0, p=1.0
        ]
        metric_fn = brier_score()
        result = metric_fn(scores)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_known_intermediate_value(self):
        """Known Brier score computation."""
        # positive_token = "1"; risk_score = P("1")
        # y=[1, 0, 1], p=[0.8, 0.3, 0.6]
        # Brier = mean((1-0.8)^2, (0-0.3)^2, (1-0.6)^2) = mean(0.04, 0.09, 0.16) = 0.29/3
        scores = [
            _score(CORRECT, 0.8, {"0": 0.2, "1": 0.8}, "1"),   # y=1
            _score(INCORRECT, 0.3, {"0": 0.7, "1": 0.3}, "0"),  # y=0
            _score(CORRECT, 0.6, {"0": 0.4, "1": 0.6}, "1"),    # y=1
        ]
        metric_fn = brier_score()
        result = metric_fn(scores)
        expected = (0.04 + 0.09 + 0.16) / 3
        assert result == pytest.approx(expected, abs=1e-6)

    def test_skips_none_risk_score(self):
        """Samples with None risk_score are skipped."""
        scores = [
            _score(CORRECT, 1.0, {"0": 0.0, "1": 1.0}, "1"),
            Score(value=INCORRECT, metadata={"risk_score": None, "option_probs": None, "target": "0"}),
            _score(CORRECT, 0.0, {"0": 1.0, "1": 0.0}, "0"),
        ]
        metric_fn = brier_score()
        result = metric_fn(scores)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_fewer_than_two_samples_returns_nan(self):
        """Single sample -> return NaN."""
        scores = [_score(CORRECT, 0.8, {"0": 0.8, "1": 0.2}, "0")]
        metric_fn = brier_score()
        assert math.isnan(metric_fn(scores))

    def test_empty_scores_returns_nan(self):
        metric_fn = brier_score()
        assert math.isnan(metric_fn([]))


# =============================================================================
# AUC Score
# =============================================================================

class TestAUCScore:

    def test_perfect_separation(self):
        """When risk_score perfectly separates classes -> AUC = 1.0."""
        # positive_token = "1"; risk_score = P("1")
        # High risk for positive class, low risk for negative class
        scores = [
            _score(CORRECT, 0.9, {"0": 0.1, "1": 0.9}, "1"),   # y=1, high score
            _score(CORRECT, 0.8, {"0": 0.2, "1": 0.8}, "1"),   # y=1, high score
            _score(CORRECT, 0.1, {"0": 0.9, "1": 0.1}, "0"),   # y=0, low score
            _score(CORRECT, 0.2, {"0": 0.8, "1": 0.2}, "0"),   # y=0, low score
        ]
        metric_fn = auc_score()
        result = metric_fn(scores)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_single_class_returns_nan(self):
        """Only one class present -> AUC = NaN (undefined)."""
        scores = [
            _score(CORRECT, 0.8, {"0": 0.8, "1": 0.2}, "0"),
            _score(CORRECT, 0.6, {"0": 0.6, "1": 0.4}, "0"),
        ]
        metric_fn = auc_score()
        assert math.isnan(metric_fn(scores))

    def test_random_discrimination(self):
        """Anti-correlated risk scores -> AUC = 0.0."""
        # positive_token = "1"; risk_score = P("1")
        # y=1 gets low scores, y=0 gets high scores -> AUC near 0
        scores = [
            _score(INCORRECT, 0.1, {"0": 0.9, "1": 0.1}, "1"),  # y=1, low score
            _score(INCORRECT, 0.2, {"0": 0.8, "1": 0.2}, "1"),  # y=1, low score
            _score(CORRECT, 0.9, {"0": 0.1, "1": 0.9}, "0"),    # y=0, high score
            _score(CORRECT, 0.8, {"0": 0.2, "1": 0.8}, "0"),    # y=0, high score
        ]
        metric_fn = auc_score()
        result = metric_fn(scores)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_skips_none_risk_score(self):
        """Samples with None risk_score are skipped."""
        # positive_token = "1"; high risk for positive, low risk for negative -> AUC=1.0
        scores = [
            _score(CORRECT, 0.9, {"0": 0.1, "1": 0.9}, "1"),
            Score(value=INCORRECT, metadata={"risk_score": None, "option_probs": None, "target": "0"}),
            _score(CORRECT, 0.1, {"0": 0.9, "1": 0.1}, "0"),
        ]
        metric_fn = auc_score()
        result = metric_fn(scores)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_fewer_than_two_samples_returns_nan(self):
        scores = [_score(CORRECT, 0.8, {"0": 0.8, "1": 0.2}, "0")]
        metric_fn = auc_score()
        assert math.isnan(metric_fn(scores))

    def test_empty_scores_returns_nan(self):
        metric_fn = auc_score()
        assert math.isnan(metric_fn([]))


# =============================================================================
# Edge cases across all metrics
# =============================================================================

class TestEdgeCases:

    def test_all_none_metadata(self):
        """When all samples have None metadata, all metrics return NaN."""
        scores = [
            Score(value=INCORRECT, metadata={"risk_score": None, "option_probs": None, "target": None}),
            Score(value=INCORRECT, metadata={"risk_score": None, "option_probs": None, "target": None}),
        ]
        assert math.isnan(expected_calibration_error()(scores))
        assert math.isnan(risk_calibration_error()(scores))
        assert math.isnan(brier_score()(scores))
        assert math.isnan(auc_score()(scores))

    def test_missing_metadata_key(self):
        """Scores with no metadata at all don't crash."""
        scores = [
            Score(value=INCORRECT, metadata={}),
            Score(value=INCORRECT, metadata={}),
        ]
        assert math.isnan(expected_calibration_error()(scores))
        assert math.isnan(risk_calibration_error()(scores))
        assert math.isnan(brier_score()(scores))
        assert math.isnan(auc_score()(scores))

    def test_none_metadata(self):
        """Scores with metadata=None don't crash."""
        scores = [
            Score(value=INCORRECT, metadata=None),
            Score(value=INCORRECT, metadata=None),
        ]
        assert math.isnan(expected_calibration_error()(scores))
        assert math.isnan(risk_calibration_error()(scores))
        assert math.isnan(brier_score()(scores))
        assert math.isnan(auc_score()(scores))

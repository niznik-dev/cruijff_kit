"""Unit tests for tools/inspect/scorers/calibration_metrics.py

Tests ECE, Brier Score, and AUC metrics that aggregate over Score objects
produced by risk_scorer.
"""

import pytest
from inspect_ai.scorer import Score, CORRECT, INCORRECT

from cruijff_kit.tools.inspect.scorers.calibration_metrics import (
    expected_calibration_error,
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

    def test_empty_scores_returns_zero(self):
        """No valid scores -> 0.0."""
        metric_fn = expected_calibration_error(n_bins=10)
        assert metric_fn([]) == 0.0


# =============================================================================
# Brier Score
# =============================================================================

class TestBrierScore:

    def test_perfect_predictions(self):
        """Risk score perfectly matches outcome -> Brier = 0."""
        scores = [
            _score(CORRECT, 1.0, {"0": 1.0, "1": 0.0}, "0"),  # y=1, p=1.0
            _score(CORRECT, 0.0, {"0": 0.0, "1": 1.0}, "1"),  # y=0, p=0.0
        ]
        metric_fn = brier_score()
        result = metric_fn(scores)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_worst_predictions(self):
        """Risk score maximally wrong -> Brier = 1.0."""
        scores = [
            _score(INCORRECT, 0.0, {"0": 0.0, "1": 1.0}, "0"),  # y=1, p=0.0
            _score(INCORRECT, 1.0, {"0": 1.0, "1": 0.0}, "1"),  # y=0, p=1.0
        ]
        metric_fn = brier_score()
        result = metric_fn(scores)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_known_intermediate_value(self):
        """Known Brier score computation."""
        # y=[1, 0, 1], p=[0.8, 0.3, 0.6]
        # Brier = mean((1-0.8)^2, (0-0.3)^2, (1-0.6)^2) = mean(0.04, 0.09, 0.16) = 0.29/3
        scores = [
            _score(CORRECT, 0.8, {"0": 0.8, "1": 0.2}, "0"),   # y=1
            _score(INCORRECT, 0.3, {"0": 0.3, "1": 0.7}, "1"),  # y=0
            _score(CORRECT, 0.6, {"0": 0.6, "1": 0.4}, "0"),    # y=1
        ]
        metric_fn = brier_score()
        result = metric_fn(scores)
        expected = (0.04 + 0.09 + 0.16) / 3
        assert result == pytest.approx(expected, abs=1e-6)

    def test_skips_none_risk_score(self):
        """Samples with None risk_score are skipped."""
        scores = [
            _score(CORRECT, 1.0, {"0": 1.0, "1": 0.0}, "0"),
            Score(value=INCORRECT, metadata={"risk_score": None, "option_probs": None, "target": "0"}),
            _score(CORRECT, 0.0, {"0": 0.0, "1": 1.0}, "1"),
        ]
        metric_fn = brier_score()
        result = metric_fn(scores)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_fewer_than_two_samples_returns_zero(self):
        """Single sample -> return 0.0."""
        scores = [_score(CORRECT, 0.8, {"0": 0.8, "1": 0.2}, "0")]
        metric_fn = brier_score()
        assert metric_fn(scores) == 0.0

    def test_empty_scores_returns_zero(self):
        metric_fn = brier_score()
        assert metric_fn([]) == 0.0


# =============================================================================
# AUC Score
# =============================================================================

class TestAUCScore:

    def test_perfect_separation(self):
        """When risk_score perfectly separates classes -> AUC = 1.0."""
        scores = [
            _score(CORRECT, 0.9, {"0": 0.9, "1": 0.1}, "0"),   # y=1, high score
            _score(CORRECT, 0.8, {"0": 0.8, "1": 0.2}, "0"),   # y=1, high score
            _score(CORRECT, 0.1, {"0": 0.1, "1": 0.9}, "1"),   # y=0, low score
            _score(CORRECT, 0.2, {"0": 0.2, "1": 0.8}, "1"),   # y=0, low score
        ]
        metric_fn = auc_score()
        result = metric_fn(scores)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_single_class_returns_zero(self):
        """Only one class present -> AUC = 0.0 (undefined, fallback)."""
        scores = [
            _score(CORRECT, 0.8, {"0": 0.8, "1": 0.2}, "0"),
            _score(CORRECT, 0.6, {"0": 0.6, "1": 0.4}, "0"),
        ]
        metric_fn = auc_score()
        assert metric_fn(scores) == 0.0

    def test_random_discrimination(self):
        """Anti-correlated risk scores -> AUC = 0.0."""
        # y=1 gets low scores, y=0 gets high scores -> AUC near 0
        scores = [
            _score(INCORRECT, 0.1, {"0": 0.1, "1": 0.9}, "0"),  # y=1, low score
            _score(INCORRECT, 0.2, {"0": 0.2, "1": 0.8}, "0"),  # y=1, low score
            _score(CORRECT, 0.9, {"0": 0.9, "1": 0.1}, "1"),    # y=0, high score
            _score(CORRECT, 0.8, {"0": 0.8, "1": 0.2}, "1"),    # y=0, high score
        ]
        metric_fn = auc_score()
        result = metric_fn(scores)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_skips_none_risk_score(self):
        """Samples with None risk_score are skipped."""
        scores = [
            _score(CORRECT, 0.9, {"0": 0.9, "1": 0.1}, "0"),
            Score(value=INCORRECT, metadata={"risk_score": None, "option_probs": None, "target": "0"}),
            _score(CORRECT, 0.1, {"0": 0.1, "1": 0.9}, "1"),
        ]
        metric_fn = auc_score()
        result = metric_fn(scores)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_fewer_than_two_samples_returns_zero(self):
        scores = [_score(CORRECT, 0.8, {"0": 0.8, "1": 0.2}, "0")]
        metric_fn = auc_score()
        assert metric_fn(scores) == 0.0

    def test_empty_scores_returns_zero(self):
        metric_fn = auc_score()
        assert metric_fn([]) == 0.0


# =============================================================================
# Edge cases across all metrics
# =============================================================================

class TestEdgeCases:

    def test_all_none_metadata(self):
        """When all samples have None metadata, all metrics return 0.0."""
        scores = [
            Score(value=INCORRECT, metadata={"risk_score": None, "option_probs": None, "target": None}),
            Score(value=INCORRECT, metadata={"risk_score": None, "option_probs": None, "target": None}),
        ]
        assert expected_calibration_error()(scores) == 0.0
        assert brier_score()(scores) == 0.0
        assert auc_score()(scores) == 0.0

    def test_missing_metadata_key(self):
        """Scores with no metadata at all don't crash."""
        scores = [
            Score(value=INCORRECT, metadata={}),
            Score(value=INCORRECT, metadata={}),
        ]
        assert expected_calibration_error()(scores) == 0.0
        assert brier_score()(scores) == 0.0
        assert auc_score()(scores) == 0.0

    def test_none_metadata(self):
        """Scores with metadata=None don't crash."""
        scores = [
            Score(value=INCORRECT, metadata=None),
            Score(value=INCORRECT, metadata=None),
        ]
        assert expected_calibration_error()(scores) == 0.0
        assert brier_score()(scores) == 0.0
        assert auc_score()(scores) == 0.0

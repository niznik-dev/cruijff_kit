"""Unit tests for tools/inspect/viz_helpers.py — detect_metrics() and friends."""

import pandas as pd
import pytest

from tools.inspect.viz_helpers import (
    DetectedMetrics,
    detect_metrics,
    display_name,
    METRIC_DISPLAY_NAMES,
)


# =============================================================================
# detect_metrics()
# =============================================================================

class TestDetectMetrics:

    def test_accuracy_only(self):
        """DataFrame with only accuracy columns."""
        df = pd.DataFrame({
            "model": ["a"],
            "score_match_accuracy": [0.8],
            "score_match_stderr": [0.01],
        })
        result = detect_metrics(df)
        assert isinstance(result, DetectedMetrics)
        assert result.accuracy == ["match"]
        assert result.supplementary == []

    def test_mixed_accuracy_and_supplementary(self):
        """DataFrame with both accuracy and calibration columns."""
        df = pd.DataFrame({
            "model": ["a"],
            "score_match_accuracy": [0.8],
            "score_match_stderr": [0.01],
            "score_risk_scorer_cruijff_kit/ece": [0.15],
            "score_risk_scorer_cruijff_kit/brier_score": [0.22],
            "score_risk_scorer_cruijff_kit/auc_score": [0.65],
        })
        result = detect_metrics(df)
        assert result.accuracy == ["match"]
        assert set(result.supplementary) == {
            "risk_scorer_cruijff_kit/ece",
            "risk_scorer_cruijff_kit/brier_score",
            "risk_scorer_cruijff_kit/auc_score",
        }

    def test_headline_columns_excluded(self):
        """score_headline_* columns should not appear in either list."""
        df = pd.DataFrame({
            "model": ["a"],
            "score_match_accuracy": [0.8],
            "score_headline_match": [0.8],
        })
        result = detect_metrics(df)
        assert result.accuracy == ["match"]
        assert result.supplementary == []

    def test_stderr_columns_excluded(self):
        """score_*_stderr columns should not appear in supplementary."""
        df = pd.DataFrame({
            "model": ["a"],
            "score_match_accuracy": [0.8],
            "score_match_stderr": [0.01],
            "score_risk_scorer_cruijff_kit/ece_stderr": [0.02],
        })
        result = detect_metrics(df)
        assert result.supplementary == []

    def test_empty_dataframe(self):
        """DataFrame with no score columns returns empty lists."""
        df = pd.DataFrame({"model": ["a"], "other_col": [1]})
        result = detect_metrics(df)
        assert result.accuracy == []
        assert result.supplementary == []

    def test_multiple_accuracy_columns(self):
        """Multiple accuracy metrics detected correctly."""
        df = pd.DataFrame({
            "model": ["a"],
            "score_match_accuracy": [0.8],
            "score_includes_accuracy": [0.9],
        })
        result = detect_metrics(df)
        assert set(result.accuracy) == {"match", "includes"}

    def test_supplementary_only(self):
        """DataFrame with only supplementary columns (no accuracy)."""
        df = pd.DataFrame({
            "model": ["a"],
            "score_risk_scorer_cruijff_kit/brier_score": [0.3],
        })
        result = detect_metrics(df)
        assert result.accuracy == []
        assert result.supplementary == ["risk_scorer_cruijff_kit/brier_score"]


# =============================================================================
# display_name()
# =============================================================================

class TestDisplayName:

    def test_known_metrics(self):
        assert display_name("risk_scorer_cruijff_kit/ece") == "ECE"
        assert display_name("risk_scorer_cruijff_kit/brier_score") == "Brier Score"
        assert display_name("risk_scorer_cruijff_kit/auc_score") == "AUC"
        assert display_name("risk_scorer_cruijff_kit/mean_risk_score") == "Mean Risk Score"

    def test_unknown_metric_title_cased(self):
        assert display_name("some_scorer/custom_metric") == "Custom Metric"

    def test_plain_name(self):
        assert display_name("my_metric") == "My Metric"

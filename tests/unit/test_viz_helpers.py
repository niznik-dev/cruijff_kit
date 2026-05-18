"""Unit tests for tools/inspect/viz_helpers.py — detect_metrics() and friends."""

from unittest.mock import MagicMock, patch

import pandas as pd

from cruijff_kit.tools.inspect.viz_helpers import (
    DetectedMetrics,
    deduplicate_eval_files,
    detect_metrics,
    display_name,
    sanitize_columns_for_viz,
)


# =============================================================================
# detect_metrics()
# =============================================================================


class TestDetectMetrics:
    def test_accuracy_only(self):
        """DataFrame with only accuracy columns."""
        df = pd.DataFrame(
            {
                "model": ["a"],
                "score_match_accuracy": [0.8],
                "score_match_stderr": [0.01],
            }
        )
        result = detect_metrics(df)
        assert isinstance(result, DetectedMetrics)
        assert result.accuracy == ["match"]
        assert result.supplementary == []

    def test_mixed_accuracy_and_supplementary(self):
        """DataFrame with both accuracy and calibration columns."""
        df = pd.DataFrame(
            {
                "model": ["a"],
                "score_match_accuracy": [0.8],
                "score_match_stderr": [0.01],
                "score_risk_scorer_cruijff_kit/ece": [0.15],
                "score_risk_scorer_cruijff_kit/brier_score": [0.22],
                "score_risk_scorer_cruijff_kit/auc_score": [0.65],
            }
        )
        result = detect_metrics(df)
        assert result.accuracy == ["match"]
        assert set(result.supplementary) == {
            "risk_scorer_cruijff_kit/ece",
            "risk_scorer_cruijff_kit/brier_score",
            "risk_scorer_cruijff_kit/auc_score",
        }

    def test_headline_columns_excluded(self):
        """score_headline_* columns should not appear in either list."""
        df = pd.DataFrame(
            {
                "model": ["a"],
                "score_match_accuracy": [0.8],
                "score_headline_match": [0.8],
            }
        )
        result = detect_metrics(df)
        assert result.accuracy == ["match"]
        assert result.supplementary == []

    def test_stderr_columns_excluded(self):
        """score_*_stderr columns should not appear in supplementary."""
        df = pd.DataFrame(
            {
                "model": ["a"],
                "score_match_accuracy": [0.8],
                "score_match_stderr": [0.01],
                "score_risk_scorer_cruijff_kit/ece_stderr": [0.02],
            }
        )
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
        df = pd.DataFrame(
            {
                "model": ["a"],
                "score_match_accuracy": [0.8],
                "score_includes_accuracy": [0.9],
            }
        )
        result = detect_metrics(df)
        assert set(result.accuracy) == {"match", "includes"}

    def test_supplementary_only(self):
        """DataFrame with only supplementary columns (no accuracy)."""
        df = pd.DataFrame(
            {
                "model": ["a"],
                "score_risk_scorer_cruijff_kit/brier_score": [0.3],
            }
        )
        result = detect_metrics(df)
        assert result.accuracy == []
        assert result.supplementary == ["risk_scorer_cruijff_kit/brier_score"]


# =============================================================================
# display_name()
# =============================================================================


class TestDisplayName:
    def test_known_metrics(self):
        assert display_name("risk_scorer_cruijff_kit/ece") == "C-ECE"
        assert display_name("risk_scorer_cruijff_kit/brier_score") == "Brier Score"
        assert display_name("risk_scorer_cruijff_kit/auc_score") == "AUC"
        assert (
            display_name("risk_scorer_cruijff_kit/mean_risk_score") == "Mean Risk Score"
        )

    def test_unknown_metric_title_cased(self):
        assert display_name("some_scorer/custom_metric") == "Custom Metric"

    def test_plain_name(self):
        assert display_name("my_metric") == "My Metric"


# =============================================================================
# sanitize_columns_for_viz()
# =============================================================================


class TestSanitizeColumnsForViz:
    def test_replaces_slashes(self):
        """Slashes in column names are replaced with double underscores."""
        df = pd.DataFrame(
            {
                "model": ["a"],
                "score_risk_scorer_cruijff_kit/auc_score": [0.85],
            }
        )
        result = sanitize_columns_for_viz(df)
        assert "score_risk_scorer_cruijff_kit__auc_score" in result.columns
        assert "score_risk_scorer_cruijff_kit/auc_score" not in result.columns

    def test_no_slashes_returns_unchanged(self):
        """When no columns contain slashes, columns and values are unchanged."""
        df = pd.DataFrame({"model": ["a"], "score_match_accuracy": [0.8]})
        result = sanitize_columns_for_viz(df)
        assert list(result.columns) == list(df.columns)
        assert result["score_match_accuracy"].tolist() == [0.8]

    def test_preserves_data(self):
        """Column values are unchanged after renaming."""
        df = pd.DataFrame(
            {
                "score_risk_scorer_cruijff_kit/brier_score": [0.15, 0.22],
            }
        )
        result = sanitize_columns_for_viz(df)
        assert result["score_risk_scorer_cruijff_kit__brier_score"].tolist() == [
            0.15,
            0.22,
        ]

    def test_multiple_slashes(self):
        """Columns with multiple slashes get all replaced."""
        df = pd.DataFrame({"a/b/c": [1]})
        result = sanitize_columns_for_viz(df)
        assert "a__b__c" in result.columns


# =============================================================================
# deduplicate_eval_files()
# =============================================================================


def _mock_eval_log(model, epoch, vis_label=""):
    """Create a mock eval log with the given model, epoch, and vis_label."""
    log = MagicMock()
    log.eval.model = model
    log.eval.metadata = {"epoch": epoch}
    log.eval.task_args = {"vis_label": vis_label} if vis_label else {}
    return log


class TestDeduplicateEvalFiles:
    @patch("cruijff_kit.tools.inspect.viz_helpers.read_eval_log")
    def test_different_vis_labels_both_kept(self, mock_read):
        """Same (model, epoch) but different vis_label → both kept."""
        mock_read.side_effect = [
            _mock_eval_log("hf/model_epoch_0", 0, "original (acs_income)"),
            _mock_eval_log("hf/model_epoch_0", 0, "original (acs_income_shuffled)"),
        ]
        files = [
            "/logs/20260319T100000_abc.eval",
            "/logs/20260319T100001_def.eval",
        ]
        kept, skipped = deduplicate_eval_files(files)
        assert len(kept) == 2
        assert len(skipped) == 0

    @patch("cruijff_kit.tools.inspect.viz_helpers.read_eval_log")
    def test_same_key_keeps_newer(self, mock_read):
        """Same (model, epoch, vis_label) → only the newer file kept."""
        mock_read.side_effect = [
            _mock_eval_log("hf/model_epoch_0", 0, "rank4"),
            _mock_eval_log("hf/model_epoch_0", 0, "rank4"),
        ]
        files = [
            "/logs/20260319T090000_old.eval",
            "/logs/20260319T100000_new.eval",
        ]
        kept, skipped = deduplicate_eval_files(files)
        assert len(kept) == 1
        assert len(skipped) == 1
        assert kept[0] == "/logs/20260319T100000_new.eval"
        assert skipped[0] == "/logs/20260319T090000_old.eval"

    @patch("cruijff_kit.tools.inspect.viz_helpers.read_eval_log")
    def test_single_file_kept(self, mock_read):
        """Single file is always kept."""
        mock_read.return_value = _mock_eval_log("hf/model_epoch_0", 0, "rank4")
        files = ["/logs/20260319T100000_abc.eval"]
        kept, skipped = deduplicate_eval_files(files)
        assert len(kept) == 1
        assert len(skipped) == 0

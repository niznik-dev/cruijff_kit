"""Unit tests for tools/inspect/report_generator.py — calibration metrics support."""

import pandas as pd
import pytest

from tools.inspect.report_generator import (
    CalibrationResult,
    ModelMetrics,
    extract_calibration_metrics,
    _format_calibration_table,
    _format_model_table,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_df(**kwargs) -> pd.DataFrame:
    """Build a single-row DataFrame for testing."""
    defaults = {
        "model": "test_model",
        "results_total_samples": 500,
    }
    defaults.update(kwargs)
    return pd.DataFrame({k: [v] for k, v in defaults.items()})


# =============================================================================
# extract_calibration_metrics()
# =============================================================================

class TestExtractCalibrationMetrics:

    def test_basic_extraction(self):
        """Extract supplementary metrics from a simple dataframe."""
        df = _make_df(
            **{
                "score_risk_scorer_cruijff_kit/ece": 0.12,
                "score_risk_scorer_cruijff_kit/brier_score": 0.25,
            }
        )
        supplementary = [
            "risk_scorer_cruijff_kit/ece",
            "risk_scorer_cruijff_kit/brier_score",
        ]
        results = extract_calibration_metrics(df, supplementary)

        assert len(results) == 1
        r = results[0]
        assert r.model_name == "test_model"
        assert r.metrics["risk_scorer_cruijff_kit/ece"] == pytest.approx(0.12)
        assert r.metrics["risk_scorer_cruijff_kit/brier_score"] == pytest.approx(0.25)
        assert r.sample_size == 500

    def test_na_handling(self):
        """pd.NA values become None in CalibrationResult."""
        df = _make_df(
            **{
                "score_risk_scorer_cruijff_kit/ece": 0.12,
                "score_risk_scorer_cruijff_kit/auc_score": pd.NA,
            }
        )
        supplementary = [
            "risk_scorer_cruijff_kit/ece",
            "risk_scorer_cruijff_kit/auc_score",
        ]
        results = extract_calibration_metrics(df, supplementary)

        assert len(results) == 1
        assert results[0].metrics["risk_scorer_cruijff_kit/ece"] == pytest.approx(0.12)
        assert results[0].metrics["risk_scorer_cruijff_kit/auc_score"] is None

    def test_all_na_skipped(self):
        """Row where all supplementary metrics are NA is skipped entirely."""
        df = _make_df(
            **{
                "score_risk_scorer_cruijff_kit/ece": pd.NA,
                "score_risk_scorer_cruijff_kit/brier_score": pd.NA,
            }
        )
        supplementary = [
            "risk_scorer_cruijff_kit/ece",
            "risk_scorer_cruijff_kit/brier_score",
        ]
        results = extract_calibration_metrics(df, supplementary)
        assert len(results) == 0

    def test_empty_supplementary_list(self):
        """Empty supplementary list returns empty results."""
        df = _make_df()
        results = extract_calibration_metrics(df, [])
        assert results == []

    def test_multiple_models(self):
        """Multiple models each get their own CalibrationResult."""
        df = pd.DataFrame({
            "model": ["model_a", "model_b"],
            "results_total_samples": [500, 300],
            "score_risk_scorer_cruijff_kit/ece": [0.1, 0.2],
        })
        supplementary = ["risk_scorer_cruijff_kit/ece"]
        results = extract_calibration_metrics(df, supplementary)
        assert len(results) == 2
        names = {r.model_name for r in results}
        assert names == {"model_a", "model_b"}

    def test_baseline_identified(self):
        """Baseline flag is set when finetuned==False."""
        df = pd.DataFrame({
            "model": ["base_model", "finetuned_model"],
            "finetuned": [False, True],
            "results_total_samples": [500, 500],
            "score_risk_scorer_cruijff_kit/ece": [0.3, 0.15],
        })
        supplementary = ["risk_scorer_cruijff_kit/ece"]
        results = extract_calibration_metrics(df, supplementary)
        baseline = [r for r in results if r.is_baseline]
        assert len(baseline) == 1
        assert baseline[0].model_name == "base_model"

    def test_missing_column_yields_none(self):
        """Supplementary metric not in dataframe becomes None."""
        df = _make_df(
            **{"score_risk_scorer_cruijff_kit/ece": 0.12}
        )
        supplementary = [
            "risk_scorer_cruijff_kit/ece",
            "risk_scorer_cruijff_kit/brier_score",  # not in df
        ]
        results = extract_calibration_metrics(df, supplementary)
        assert len(results) == 1
        assert results[0].metrics["risk_scorer_cruijff_kit/ece"] == pytest.approx(0.12)
        assert results[0].metrics["risk_scorer_cruijff_kit/brier_score"] is None

    def test_with_epoch_grouping(self):
        """Results grouped by model + epoch."""
        df = pd.DataFrame({
            "model": ["model_a", "model_a"],
            "epoch": [1, 2],
            "results_total_samples": [500, 500],
            "score_risk_scorer_cruijff_kit/ece": [0.2, 0.15],
        })
        supplementary = ["risk_scorer_cruijff_kit/ece"]
        results = extract_calibration_metrics(df, supplementary)
        assert len(results) == 2
        epochs = {r.epoch for r in results}
        assert epochs == {1, 2}


# =============================================================================
# _format_calibration_table()
# =============================================================================

class TestFormatCalibrationTable:

    def test_basic_table(self):
        """Produces a valid markdown table."""
        results = [
            CalibrationResult(
                model_name="model_a",
                metrics={"risk_scorer_cruijff_kit/ece": 0.123, "risk_scorer_cruijff_kit/auc_score": 0.789},
                sample_size=500,
                epoch=3,
            ),
        ]
        table = _format_calibration_table(results)
        assert "ECE" in table
        assert "AUC" in table
        assert "0.123" in table
        assert "0.789" in table
        assert "500" in table
        assert "model_a" in table

    def test_none_renders_as_dash(self):
        """None values render as '-'."""
        results = [
            CalibrationResult(
                model_name="m",
                metrics={"risk_scorer_cruijff_kit/ece": None},
                sample_size=100,
            ),
        ]
        table = _format_calibration_table(results)
        lines = table.strip().split("\n")
        # Data row should contain "-" for the None metric
        data_row = lines[-1]
        cells = [c.strip() for c in data_row.split("|") if c.strip()]
        # cells: [model, epoch, ECE value, sample_size]
        assert cells[2] == "-"

    def test_baseline_marker(self):
        """Baseline models get an asterisk marker."""
        results = [
            CalibrationResult(
                model_name="base",
                metrics={"risk_scorer_cruijff_kit/ece": 0.3},
                sample_size=500,
                is_baseline=True,
            ),
            CalibrationResult(
                model_name="tuned",
                metrics={"risk_scorer_cruijff_kit/ece": 0.15},
                sample_size=500,
            ),
        ]
        table = _format_calibration_table(results)
        assert "base *" in table
        assert "tuned |" in table  # no marker

    def test_empty_results(self):
        """Empty results list returns informational message."""
        table = _format_calibration_table([])
        assert "No calibration metrics" in table

    def test_three_decimal_places(self):
        """Metric values formatted to 3 decimal places."""
        results = [
            CalibrationResult(
                model_name="m",
                metrics={"risk_scorer_cruijff_kit/brier_score": 0.1},
                sample_size=100,
            ),
        ]
        table = _format_calibration_table(results)
        assert "0.100" in table

    def test_epoch_none_renders_as_dash(self):
        """When epoch is None, render as '-'."""
        results = [
            CalibrationResult(
                model_name="m",
                metrics={"risk_scorer_cruijff_kit/ece": 0.2},
                sample_size=100,
                epoch=None,
            ),
        ]
        table = _format_calibration_table(results)
        lines = table.strip().split("\n")
        data_row = lines[-1]
        cells = [c.strip() for c in data_row.split("|") if c.strip()]
        assert cells[1] == "-"


# =============================================================================
# _format_model_table() with calibration
# =============================================================================

class TestFormatModelTableCombined:

    def _metric(self, name="model_a", accuracy=0.75, epoch=1, n=500, **kw):
        from tools.inspect.report_generator import compute_wilson_ci
        ci_lo, ci_hi = compute_wilson_ci(accuracy, n)
        return ModelMetrics(
            name=name, accuracy=accuracy, ci_lower=ci_lo, ci_upper=ci_hi,
            sample_size=n, epoch=epoch, **kw,
        )

    def test_without_calibration(self):
        """Table works the same when no calibration is passed."""
        table, footnotes = _format_model_table([self._metric()])
        assert "Accuracy" in table
        assert "ECE" not in table
        assert "AUC" not in table

    def test_with_calibration_adds_columns(self):
        """Supplementary columns appear in header when calibration provided."""
        cal = [CalibrationResult(
            model_name="model_a", epoch=1, sample_size=500,
            metrics={"risk_scorer_cruijff_kit/auc_score": 0.85, "risk_scorer_cruijff_kit/brier_score": 0.15},
        )]
        table, footnotes = _format_model_table([self._metric()], calibration=cal)
        assert "AUC" in table
        assert "Brier Score" in table
        assert "0.850" in table
        assert "0.150" in table

    def test_missing_calibration_shows_dash(self):
        """Models without calibration data get dashes in metric columns."""
        m_base = self._metric(name="base", accuracy=0.0, epoch=None, is_baseline=True)
        m_tuned = self._metric(name="tuned", accuracy=0.8, epoch=1)
        cal = [CalibrationResult(
            model_name="tuned", epoch=1, sample_size=500,
            metrics={"risk_scorer_cruijff_kit/auc_score": 0.9},
        )]
        table, footnotes = _format_model_table([m_base, m_tuned], calibration=cal)
        lines = table.strip().split("\n")
        # base model row should have "-" for AUC
        base_row = [l for l in lines if "base" in l][0]
        cells = [c.strip() for c in base_row.split("|") if c.strip()]
        # cells: Model, Epoch, Accuracy, AUC (no Sample Size — uniform)
        assert cells[3] == "-"

    def test_uniform_sample_size_excluded_from_table(self):
        """When all models have the same sample size, column is omitted."""
        m1 = self._metric(name="a", n=1000)
        m2 = self._metric(name="b", n=1000)
        table, footnotes = _format_model_table([m1, m2])
        assert "Sample Size" not in table
        assert any("1000" in f for f in footnotes)

    def test_varying_sample_size_included_in_table(self):
        """When sample sizes differ, column stays in the table."""
        m1 = self._metric(name="a", n=1000)
        m2 = self._metric(name="b", n=500)
        table, footnotes = _format_model_table([m1, m2])
        assert "Sample Size" in table
        assert not any("per model" in f for f in footnotes)

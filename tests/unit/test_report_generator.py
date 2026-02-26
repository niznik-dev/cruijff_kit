"""Unit tests for tools/inspect/report_generator.py — calibration metrics support."""

from pathlib import Path

import pandas as pd
import pytest

from cruijff_kit.tools.inspect.report_generator import (
    CalibrationResult,
    ModelMetrics,
    expand_details_for_pdf,
    extract_calibration_metrics,
    generate_report,
    _format_calibration_table,
    _format_inspect_view_commands,
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
        # Find the ECE column by header name, not position
        header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
        data_cells = [c.strip() for c in data_row.split("|") if c.strip()]
        ece_idx = next(i for i, h in enumerate(header_cells) if "ECE" in h)
        assert data_cells[ece_idx] == "-"

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
        # Find the Epoch column by header name, not position
        header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
        data_cells = [c.strip() for c in data_row.split("|") if c.strip()]
        epoch_idx = next(i for i, h in enumerate(header_cells) if "Epoch" in h)
        assert data_cells[epoch_idx] == "-"


# =============================================================================
# _format_model_table() with calibration
# =============================================================================

class TestFormatModelTableCombined:

    def _metric(self, name="model_a", accuracy=0.75, epoch=1, n=500, **kw):
        from cruijff_kit.tools.inspect.report_generator import compute_wilson_ci
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
        m_base = self._metric(name="base", accuracy=0.0, epoch=None)
        m_tuned = self._metric(name="tuned", accuracy=0.8, epoch=1)
        cal = [CalibrationResult(
            model_name="tuned", epoch=1, sample_size=500,
            metrics={"risk_scorer_cruijff_kit/auc_score": 0.9},
        )]
        table, footnotes = _format_model_table([m_base, m_tuned], calibration=cal)
        lines = table.strip().split("\n")
        # base model row should have "-" for AUC
        base_row = [l for l in lines if "base" in l][0]
        header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
        data_cells = [c.strip() for c in base_row.split("|") if c.strip()]
        auc_idx = next(i for i, h in enumerate(header_cells) if "AUC" in h)
        assert data_cells[auc_idx] == "-"

    def test_uniform_sample_size_excluded_from_table(self):
        """When all models have the same sample size, column is omitted."""
        m1 = self._metric(name="a", n=1000)
        m2 = self._metric(name="b", n=1000)
        table, footnotes = _format_model_table([m1, m2])
        assert "Sample Size" not in table
        assert any("1,000" in f for f in footnotes)

    def test_varying_sample_size_included_in_table(self):
        """When sample sizes differ, column stays in the table."""
        m1 = self._metric(name="a", n=1000)
        m2 = self._metric(name="b", n=500)
        table, footnotes = _format_model_table([m1, m2])
        assert "Sample Size" in table
        assert not any("per model" in f for f in footnotes)


# =============================================================================
# Provenance metadata
# =============================================================================

class TestProvenanceMetadata:

    def _make_report_df(self):
        """Minimal DataFrame for generate_report."""
        return pd.DataFrame({
            "model": ["model_a"],
            "results_total_samples": [100],
            "score_match_accuracy": [0.75],
        })

    def test_generated_by_in_header_and_footer(self, tmp_path):
        """generated_by appears in both header and footer."""
        report = generate_report(
            df=self._make_report_df(),
            experiment_name="test",
            output_path=tmp_path / "report.md",
            generated_by="Claude Opus 4.6",
        )
        assert "**Generated by:** Claude Opus 4.6" in report
        assert "Generated by Claude Opus 4.6 via analyze-experiment skill" in report

    def test_default_without_generated_by(self, tmp_path):
        """Without generated_by, no attribution in header; footer uses default."""
        report = generate_report(
            df=self._make_report_df(),
            experiment_name="test",
            output_path=tmp_path / "report.md",
        )
        assert "**Generated by:**" not in report
        assert "Generated by analyze-experiment skill" in report

    def test_eval_log_paths_in_footer(self, tmp_path):
        """eval_log_paths appear in a collapsible details block."""
        log_paths = [
            Path("/experiments/run1/eval/logs/log1.eval"),
            Path("/experiments/run2/eval/logs/log2.eval"),
        ]
        report = generate_report(
            df=self._make_report_df(),
            experiment_name="test",
            output_path=tmp_path / "report.md",
            eval_log_paths=log_paths,
        )
        assert "<details>" in report
        assert "Source eval logs" in report
        assert "log1.eval" in report
        assert "log2.eval" in report

    def test_no_eval_log_paths_no_details(self, tmp_path):
        """Without eval_log_paths, no details block appears."""
        report = generate_report(
            df=self._make_report_df(),
            experiment_name="test",
            output_path=tmp_path / "report.md",
        )
        assert "<details>" not in report

    def test_inspect_view_commands_in_report(self, tmp_path):
        """inspect view commands appear when eval_log_paths are provided."""
        log_paths = [
            Path("/experiments/run1/eval/logs/log1.eval"),
            Path("/experiments/run2/eval/logs/log2.eval"),
        ]
        report = generate_report(
            df=self._make_report_df(),
            experiment_name="test",
            output_path=tmp_path / "report.md",
            eval_log_paths=log_paths,
        )
        assert "Inspect view commands" in report
        assert "inspect view start --log-dir" in report


# =============================================================================
# compute_section parameter
# =============================================================================

class TestComputeSection:

    def _make_report_df(self):
        """Minimal DataFrame for generate_report."""
        return pd.DataFrame({
            "model": ["model_a"],
            "results_total_samples": [100],
            "score_match_accuracy": [0.75],
        })

    def test_compute_section_none_unchanged(self, tmp_path):
        """compute_section=None produces output without compute section."""
        report = generate_report(
            df=self._make_report_df(),
            experiment_name="test",
            output_path=tmp_path / "report.md",
        )
        assert "Compute Utilization" not in report

    def test_compute_section_appears_in_report(self, tmp_path):
        """compute_section content appears in the report."""
        compute = "## Compute Utilization\n\n| Run | Type | Wall Time |\n| --- | --- | --- |\n| 1B | finetune | 00:09:52 |"
        report = generate_report(
            df=self._make_report_df(),
            experiment_name="test",
            output_path=tmp_path / "report.md",
            compute_section=compute,
        )
        assert "Compute Utilization" in report
        assert "00:09:52" in report

    def test_compute_section_after_analysis(self, tmp_path):
        """compute_section appears after Analysis & Interpretation."""
        compute = "## Compute Utilization\n\nGPU data here."
        future = "### Key Findings\n\nSome analysis."
        report = generate_report(
            df=self._make_report_df(),
            experiment_name="test",
            output_path=tmp_path / "report.md",
            future_directions=future,
            compute_section=compute,
        )
        analysis_pos = report.index("Analysis & Interpretation")
        compute_pos = report.index("Compute Utilization")
        separator_pos = report.index("---", compute_pos)
        assert compute_pos > analysis_pos
        assert separator_pos > compute_pos


# =============================================================================
# _format_inspect_view_commands()
# =============================================================================

class TestFormatInspectViewCommands:

    def test_empty_paths(self):
        """Empty list returns empty string."""
        assert _format_inspect_view_commands([]) == ""

    def test_single_log_dir(self):
        """Single log path produces one command."""
        paths = [Path("/exp/run1/eval/logs/log1.eval")]
        result = _format_inspect_view_commands(paths)
        assert "inspect view start --log-dir /exp/run1/eval/logs" in result
        assert "<details>" in result
        assert "Inspect view commands" in result

    def test_deduplicates_shared_directory(self):
        """Multiple logs in the same directory produce one command."""
        paths = [
            Path("/exp/run1/eval/logs/epoch1.eval"),
            Path("/exp/run1/eval/logs/epoch2.eval"),
        ]
        result = _format_inspect_view_commands(paths)
        assert result.count("inspect view start") == 1

    def test_multiple_directories(self):
        """Logs in different directories produce one command each."""
        paths = [
            Path("/exp/run1/eval/logs/log1.eval"),
            Path("/exp/run2/eval/logs/log2.eval"),
            Path("/exp/run3/eval/logs/log3.eval"),
        ]
        result = _format_inspect_view_commands(paths)
        assert result.count("inspect view start") == 3

    def test_collapse_above_threshold(self):
        """More directories than max_commands collapses to template."""
        paths = [Path(f"/exp/run{i}/logs/log.eval") for i in range(25)]
        result = _format_inspect_view_commands(paths, max_commands=20)
        assert "&lt;LOG_DIR&gt;" in result
        assert result.count("inspect view start") == 1
        assert "25 log directories" in result

    def test_no_collapse_at_threshold(self):
        """Exactly max_commands directories are enumerated, not collapsed."""
        paths = [Path(f"/exp/run{i}/logs/log.eval") for i in range(20)]
        result = _format_inspect_view_commands(paths, max_commands=20)
        assert "&lt;LOG_DIR&gt;" not in result
        assert result.count("inspect view start") == 20

    def test_sorted_output(self):
        """Commands are sorted by directory path."""
        paths = [
            Path("/exp/zebra/logs/log.eval"),
            Path("/exp/alpha/logs/log.eval"),
        ]
        result = _format_inspect_view_commands(paths)
        lines = [l for l in result.split("\n") if "inspect view start" in l]
        assert "alpha" in lines[0]
        assert "zebra" in lines[1]


# =============================================================================
# expand_details_for_pdf()
# =============================================================================

class TestExpandDetailsForPdf:

    def test_summary_becomes_bold(self):
        """<summary> text becomes a **bold** label."""
        text = "<details>\n<summary>My Section</summary>\n\nSome content.\n\n</details>"
        result = expand_details_for_pdf(text)
        assert "**My Section**" in result
        assert "<details>" not in result
        assert "<summary>" not in result

    def test_content_preserved(self):
        """Inner content is kept after expansion."""
        text = "<details>\n<summary>Logs</summary>\n\n- `log1.eval`\n- `log2.eval`\n\n</details>"
        result = expand_details_for_pdf(text)
        assert "log1.eval" in result
        assert "log2.eval" in result

    def test_pre_code_to_fenced(self):
        """<pre><code> blocks become fenced code blocks."""
        text = (
            "<details>\n<summary>Commands</summary>\n\n"
            "<pre><code>echo hello\necho world</code></pre>\n\n</details>"
        )
        result = expand_details_for_pdf(text)
        assert "```\necho hello\necho world\n```" in result
        assert "<pre>" not in result
        assert "<code>" not in result

    def test_multiple_details_blocks(self):
        """Multiple <details> blocks are all expanded."""
        text = (
            "<details>\n<summary>First</summary>\n\nAAA\n\n</details>\n\n"
            "<details>\n<summary>Second</summary>\n\nBBB\n\n</details>"
        )
        result = expand_details_for_pdf(text)
        assert "**First**" in result
        assert "**Second**" in result
        assert "AAA" in result
        assert "BBB" in result
        assert "<details>" not in result

    def test_no_details_passthrough(self):
        """Text without <details> blocks is returned unchanged."""
        text = "# Heading\n\nSome regular markdown.\n"
        assert expand_details_for_pdf(text) == text

    def test_real_report_round_trip(self, tmp_path):
        """Expanding a generated report removes all HTML details tags."""
        df = pd.DataFrame({
            "model": ["model_a"],
            "results_total_samples": [100],
            "score_match_accuracy": [0.75],
        })
        log_paths = [
            Path("/exp/run1/eval/logs/log1.eval"),
            Path("/exp/run2/eval/logs/log2.eval"),
        ]
        report = generate_report(
            df=df,
            experiment_name="test",
            output_path=tmp_path / "report.md",
            eval_log_paths=log_paths,
        )
        expanded = expand_details_for_pdf(report)
        assert "<details>" not in expanded
        assert "</details>" not in expanded
        assert "<summary>" not in expanded
        # Content is still present
        assert "log1.eval" in expanded
        assert "inspect view start" in expanded

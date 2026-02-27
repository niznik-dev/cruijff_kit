"""Tests for cruijff_kit.tools.inspect.heterogeneity.heterogeneity_report."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cruijff_kit.tools.inspect.heterogeneity.heterogeneity_report import (
    load_data,
    calculate_group_metrics,
    find_heterogeneity,
    identify_groups,
    generate_report,
)


@pytest.fixture
def basic_df():
    """A simple DataFrame with two groups and clear performance difference."""
    return pd.DataFrame({
        "INPUT": ["a"] * 100 + ["b"] * 100,
        "TRUE_LABEL": [1, 0] * 50 + [1, 0] * 50,
        "PREDICTION": [1, 0] * 50 + [0, 1] * 50,  # group A perfect, group B inverted
        "P(TRUE LABEL)": [0.9, 0.9] * 50 + [0.1, 0.1] * 50,
        "GROUP": ["A"] * 100 + ["B"] * 100,
    })


@pytest.fixture
def uniform_df():
    """A DataFrame where all groups perform identically."""
    return pd.DataFrame({
        "INPUT": ["x"] * 60,
        "TRUE_LABEL": [1, 0] * 30,
        "PREDICTION": [1, 0] * 30,
        "P(TRUE LABEL)": [0.8] * 60,
        "GROUP": ["X"] * 20 + ["Y"] * 20 + ["Z"] * 20,
    })


@pytest.fixture
def single_class_group_df():
    """A DataFrame where one group has only one class (AUC undefined)."""
    return pd.DataFrame({
        "INPUT": ["a"] * 40,
        "TRUE_LABEL": [1, 0] * 10 + [1] * 20,  # group B is all 1s
        "PREDICTION": [1, 0] * 10 + [1] * 20,
        "P(TRUE LABEL)": [0.8] * 40,
        "GROUP": ["A"] * 20 + ["B"] * 20,
    })


@pytest.fixture
def csv_file(tmp_path, basic_df):
    """Write basic_df to a CSV file."""
    path = tmp_path / "data.csv"
    basic_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def multi_group_df():
    """A DataFrame with 5 groups where one is a clear outlier.

    Groups A-D have ~100% accuracy; group E has 0% accuracy.
    With 5 groups, z-scores have enough spread that the default
    z_threshold (-1.5) can actually trigger.
    """
    good_rows = []
    for label in ["A", "B", "C", "D"]:
        good_rows.extend([
            {"INPUT": "x", "TRUE_LABEL": 1, "PREDICTION": 1,
             "P(TRUE LABEL)": 0.9, "GROUP": label}
        ] * 25 + [
            {"INPUT": "x", "TRUE_LABEL": 0, "PREDICTION": 0,
             "P(TRUE LABEL)": 0.9, "GROUP": label}
        ] * 25)
    bad_rows = [
        {"INPUT": "x", "TRUE_LABEL": 1, "PREDICTION": 0,
         "P(TRUE LABEL)": 0.1, "GROUP": "E"}
    ] * 25 + [
        {"INPUT": "x", "TRUE_LABEL": 0, "PREDICTION": 1,
         "P(TRUE LABEL)": 0.1, "GROUP": "E"}
    ] * 25
    return pd.DataFrame(good_rows + bad_rows)


class TestLoadData:
    """Tests for load_data."""

    def test_loads_csv(self, csv_file):
        df = load_data(csv_file)
        assert len(df) == 200
        assert "GROUP" in df.columns

    def test_custom_group_column(self, tmp_path):
        df = pd.DataFrame({
            "INPUT": ["a", "b"],
            "TRUE_LABEL": [1, 0],
            "PREDICTION": [1, 0],
            "P(TRUE LABEL)": [0.9, 0.9],
            "state": ["NJ", "NY"],
        })
        path = tmp_path / "custom.csv"
        df.to_csv(path, index=False)

        result = load_data(str(path), group_column="state")
        assert "GROUP" in result.columns
        assert set(result["GROUP"]) == {"NJ", "NY"}

    def test_missing_group_column_raises(self, tmp_path):
        df = pd.DataFrame({
            "INPUT": ["a"],
            "TRUE_LABEL": [1],
            "PREDICTION": [1],
            "P(TRUE LABEL)": [0.9],
        })
        path = tmp_path / "no_group.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="not found"):
            load_data(str(path), group_column="nonexistent")

    def test_missing_default_group_raises(self, tmp_path):
        df = pd.DataFrame({
            "INPUT": ["a"],
            "TRUE_LABEL": [1],
            "PREDICTION": [1],
            "P(TRUE LABEL)": [0.9],
        })
        path = tmp_path / "no_default.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="GROUP"):
            load_data(str(path))


class TestCalculateGroupMetrics:
    """Tests for calculate_group_metrics."""

    def test_returns_dict_per_group(self, basic_df):
        metrics = calculate_group_metrics(basic_df)
        assert "A" in metrics
        assert "B" in metrics

    def test_perfect_group_accuracy(self, basic_df):
        metrics = calculate_group_metrics(basic_df)
        assert metrics["A"]["accuracy"] == 1.0

    def test_inverted_group_accuracy(self, basic_df):
        metrics = calculate_group_metrics(basic_df)
        assert metrics["B"]["accuracy"] == 0.0

    def test_sample_counts(self, basic_df):
        metrics = calculate_group_metrics(basic_df)
        assert metrics["A"]["n_samples"] == 100
        assert metrics["B"]["n_samples"] == 100

    def test_auc_computed(self, basic_df):
        metrics = calculate_group_metrics(basic_df)
        assert not np.isnan(metrics["A"]["auc"])

    def test_single_class_auc_is_nan(self, single_class_group_df):
        """AUC should be NaN when a group has only one class."""
        metrics = calculate_group_metrics(single_class_group_df)
        assert np.isnan(metrics["B"]["auc"])

    def test_uniform_groups_same_accuracy(self, uniform_df):
        metrics = calculate_group_metrics(uniform_df)
        accuracies = [m["accuracy"] for m in metrics.values()]
        assert len(set(accuracies)) == 1


class TestFindHeterogeneity:
    """Tests for find_heterogeneity."""

    def test_detects_heterogeneity(self, basic_df):
        metrics = calculate_group_metrics(basic_df)
        results = find_heterogeneity(basic_df, metrics)
        assert results["accuracy_heterogeneity"]["heterogeneity"] is True

    def test_no_heterogeneity_in_uniform(self, uniform_df):
        metrics = calculate_group_metrics(uniform_df)
        results = find_heterogeneity(uniform_df, metrics)
        assert results["accuracy_heterogeneity"]["heterogeneity"] is False

    def test_returns_p_value(self, basic_df):
        metrics = calculate_group_metrics(basic_df)
        results = find_heterogeneity(basic_df, metrics)
        assert "p_value" in results["accuracy_heterogeneity"]
        assert 0 <= results["accuracy_heterogeneity"]["p_value"] <= 1

    def test_returns_auc_stats(self, basic_df):
        metrics = calculate_group_metrics(basic_df)
        results = find_heterogeneity(basic_df, metrics)
        assert "std" in results["auc_heterogeneity"]
        assert "range" in results["auc_heterogeneity"]
        assert "mean" in results["auc_heterogeneity"]

    def test_single_group(self):
        """Single group should report no heterogeneity."""
        df = pd.DataFrame({
            "INPUT": ["a"] * 20,
            "TRUE_LABEL": [1, 0] * 10,
            "PREDICTION": [1, 0] * 10,
            "P(TRUE LABEL)": [0.8] * 20,
            "GROUP": ["only"] * 20,
        })
        metrics = calculate_group_metrics(df)
        results = find_heterogeneity(df, metrics)
        assert results["accuracy_heterogeneity"]["p_value"] == 1.0


class TestIdentifyGroups:
    """Tests for identify_groups."""

    def test_finds_accuracy_outliers(self, basic_df):
        """With only 2 groups, z-scores are exactly +/-1, so default
        threshold of -1.5 won't flag anything. Use a looser threshold."""
        metrics = calculate_group_metrics(basic_df)
        identified = identify_groups(metrics, z_threshold=-0.5)
        outlier_groups = [o["group"] for o in identified["accuracy_outliers"]]
        assert "B" in outlier_groups

    def test_no_outliers_in_uniform(self, uniform_df):
        metrics = calculate_group_metrics(uniform_df)
        identified = identify_groups(metrics)
        assert len(identified["accuracy_outliers"]) == 0

    def test_custom_thresholds(self, basic_df):
        metrics = calculate_group_metrics(basic_df)
        # Very strict threshold — should find more outliers
        identified = identify_groups(metrics, z_threshold=0, auc_threshold=0.99)
        assert len(identified["auc_outliers"]) > 0

    def test_default_threshold_with_multi_group(self, multi_group_df):
        """Default z_threshold (-1.5) detects group E as outlier with 5 groups."""
        metrics = calculate_group_metrics(multi_group_df)
        identified = identify_groups(metrics)  # uses default z_threshold=-1.5
        outlier_groups = [o["group"] for o in identified["accuracy_outliers"]]
        assert "E" in outlier_groups
        # Good groups should not be flagged
        for label in ["A", "B", "C", "D"]:
            assert label not in outlier_groups

    def test_outlying_in_both(self, basic_df):
        metrics = calculate_group_metrics(basic_df)
        identified = identify_groups(metrics)
        # B is bad in both accuracy and AUC
        for group in identified["outlying_in_both"]:
            assert group in [o["group"] for o in identified["accuracy_outliers"]]
            assert group in [o["group"] for o in identified["auc_outliers"]]


class TestGenerateReport:
    """Tests for generate_report."""

    def test_creates_json_file(self, tmp_path, basic_df):
        metrics = calculate_group_metrics(basic_df)
        heterogeneity = find_heterogeneity(basic_df, metrics)
        identified = identify_groups(metrics)

        generate_report(basic_df, metrics, heterogeneity, identified, str(tmp_path))

        report_path = tmp_path / "heterogeneity_report.json"
        assert report_path.exists()

    def test_report_structure(self, tmp_path, basic_df):
        metrics = calculate_group_metrics(basic_df)
        heterogeneity = find_heterogeneity(basic_df, metrics)
        identified = identify_groups(metrics)

        report = generate_report(
            basic_df, metrics, heterogeneity, identified, str(tmp_path)
        )

        assert "summary" in report
        assert "heterogeneity_analysis" in report
        assert "identified_groups" in report
        assert "group_metrics" in report

    def test_summary_fields(self, tmp_path, basic_df):
        metrics = calculate_group_metrics(basic_df)
        heterogeneity = find_heterogeneity(basic_df, metrics)
        identified = identify_groups(metrics)

        report = generate_report(
            basic_df, metrics, heterogeneity, identified, str(tmp_path)
        )

        assert report["summary"]["total_samples"] == 200
        assert report["summary"]["num_groups"] == 2
        assert isinstance(report["summary"]["heterogeneity_found"], bool)

    def test_report_json_is_valid(self, tmp_path, basic_df):
        metrics = calculate_group_metrics(basic_df)
        heterogeneity = find_heterogeneity(basic_df, metrics)
        identified = identify_groups(metrics)

        generate_report(basic_df, metrics, heterogeneity, identified, str(tmp_path))

        report_path = tmp_path / "heterogeneity_report.json"
        with open(report_path) as f:
            loaded = json.load(f)
        assert loaded["summary"]["total_samples"] == 200

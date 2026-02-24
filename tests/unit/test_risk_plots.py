"""Unit tests for per-sample risk data extraction and plot generation."""

import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cruijff_kit.tools.inspect.viz_helpers import (
    DetectedMetrics,
    PerSampleRiskData,
    _extract_risk_from_log,
    extract_per_sample_risk_data,
    generate_roc_overlay,
    generate_calibration_overlay,
    generate_prediction_histogram,
)


# =============================================================================
# Helpers — build fake eval log structures
# =============================================================================

def _make_sample(risk_score, target, option_probs):
    """Create a minimal fake sample with risk_scorer metadata."""
    return SimpleNamespace(
        scores={
            "risk_scorer": SimpleNamespace(
                metadata={
                    "risk_score": risk_score,
                    "target": target,
                    "option_probs": option_probs,
                }
            )
        }
    )


def _make_log(samples, model="hf/test_model", vis_label=None):
    """Create a minimal fake eval log."""
    task_args = {}
    if vis_label is not None:
        task_args["vis_label"] = vis_label
    return SimpleNamespace(
        eval=SimpleNamespace(model=model, task_args=task_args),
        samples=samples,
    )


def _make_binary_samples(n=100, seed=42):
    """Create n samples with random risk_score and balanced binary targets."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        target = "0" if i < n // 2 else "1"
        risk = float(rng.uniform(0.2 if target == "0" else 0.6, 0.5 if target == "0" else 0.95))
        samples.append(_make_sample(risk, target, {"0": risk, "1": 1 - risk}))
    return samples


# =============================================================================
# PerSampleRiskData
# =============================================================================

class TestPerSampleRiskData:

    def test_construction(self):
        d = PerSampleRiskData("m", [1.0, 0.0], [0.9, 0.1], 10, 2)
        assert d.model_name == "m"
        assert d.n_total == 10
        assert d.n_valid == 2


# =============================================================================
# _extract_risk_from_log
# =============================================================================

class TestExtractRiskFromLog:

    def test_basic_extraction(self):
        samples = _make_binary_samples(20)
        log = _make_log(samples)
        result = _extract_risk_from_log(log)
        assert result is not None
        assert result.model_name == "hf/test_model"
        assert result.n_total == 20
        assert result.n_valid == 20
        assert len(result.y_true) == 20
        assert len(result.y_score) == 20

    def test_vis_label_used(self):
        samples = _make_binary_samples(10)
        log = _make_log(samples, vis_label="My Model")
        result = _extract_risk_from_log(log)
        assert result.model_name == "My Model"

    def test_skips_single_class(self):
        """All targets the same -> only one class -> returns None."""
        samples = [_make_sample(0.8, "0", {"0": 0.8, "1": 0.2}) for _ in range(10)]
        log = _make_log(samples)
        assert _extract_risk_from_log(log) is None

    def test_skips_too_few_samples(self):
        """Fewer than 2 valid samples -> returns None."""
        samples = [_make_sample(0.8, "0", {"0": 0.8, "1": 0.2})]
        log = _make_log(samples)
        assert _extract_risk_from_log(log) is None

    def test_skips_missing_risk_scorer(self):
        """Samples without risk_scorer key are skipped."""
        samples = [
            SimpleNamespace(scores={"match": SimpleNamespace(metadata={})}),
            SimpleNamespace(scores={"match": SimpleNamespace(metadata={})}),
        ]
        log = _make_log(samples)
        assert _extract_risk_from_log(log) is None

    def test_skips_missing_metadata_fields(self):
        """Samples with partial metadata are skipped."""
        s1 = SimpleNamespace(
            scores={"risk_scorer": SimpleNamespace(metadata={"risk_score": 0.5})}
        )  # missing target + option_probs
        samples = [s1, s1]
        log = _make_log(samples)
        assert _extract_risk_from_log(log) is None

    def test_empty_samples(self):
        log = _make_log([])
        assert _extract_risk_from_log(log) is None

    def test_none_samples(self):
        log = _make_log(None)
        assert _extract_risk_from_log(log) is None

    def test_y_true_encoding(self):
        """Verify y_true = 1.0 when target matches last option_probs key."""
        samples = [
            _make_sample(0.9, "A", {"A": 0.9, "B": 0.1}),  # negative (target≠B)
            _make_sample(0.3, "B", {"A": 0.3, "B": 0.7}),  # positive (target=B)
        ]
        log = _make_log(samples)
        result = _extract_risk_from_log(log)
        assert result.y_true == [0.0, 1.0]


# =============================================================================
# extract_per_sample_risk_data (integration-level, mocks read_eval_log)
# =============================================================================

class TestExtractPerSampleRiskData:

    @patch("cruijff_kit.tools.inspect.viz_helpers.read_eval_log")
    def test_reads_multiple_files(self, mock_read):
        log1 = _make_log(_make_binary_samples(20, seed=1), model="model_a")
        log2 = _make_log(_make_binary_samples(20, seed=2), model="model_b")
        mock_read.side_effect = [log1, log2]

        results = extract_per_sample_risk_data(["f1.eval", "f2.eval"])
        assert len(results) == 2
        assert results[0].model_name == "model_a"
        assert results[1].model_name == "model_b"

    @patch("cruijff_kit.tools.inspect.viz_helpers.read_eval_log")
    def test_skips_unreadable_files(self, mock_read):
        mock_read.side_effect = Exception("corrupt")
        results = extract_per_sample_risk_data(["bad.eval"])
        assert results == []

    @patch("cruijff_kit.tools.inspect.viz_helpers.read_eval_log")
    def test_skips_unusable_models(self, mock_read):
        """Single-class model is skipped but good model is kept."""
        good_log = _make_log(_make_binary_samples(20), model="good")
        bad_log = _make_log(
            [_make_sample(0.9, "0", {"0": 0.9, "1": 0.1}) for _ in range(10)],
            model="bad",
        )
        mock_read.side_effect = [good_log, bad_log]
        results = extract_per_sample_risk_data(["good.eval", "bad.eval"])
        assert len(results) == 1
        assert results[0].model_name == "good"


# =============================================================================
# has_risk_scorer property
# =============================================================================

class TestHasRiskScorer:

    def test_true_when_auc_present(self):
        dm = DetectedMetrics(
            accuracy=["match"],
            supplementary=["risk_scorer_cruijff_kit/auc_score", "risk_scorer_cruijff_kit/ece"],
        )
        assert dm.has_risk_scorer is True

    def test_false_when_no_auc(self):
        dm = DetectedMetrics(accuracy=["match"], supplementary=["risk_scorer_cruijff_kit/ece"])
        assert dm.has_risk_scorer is False

    def test_false_when_empty(self):
        dm = DetectedMetrics()
        assert dm.has_risk_scorer is False


# =============================================================================
# generate_roc_overlay
# =============================================================================

class TestGenerateRocOverlay:

    def test_produces_png(self, tmp_path):
        rd = PerSampleRiskData("model_a", [1.0]*50 + [0.0]*50,
                               [0.9]*50 + [0.1]*50, 100, 100)
        out = generate_roc_overlay([rd], tmp_path / "roc.png")
        assert out is not None
        assert out.exists()
        assert out.stat().st_size > 0

    def test_multiple_models(self, tmp_path):
        rd1 = PerSampleRiskData("a", [1.0]*50 + [0.0]*50, [0.9]*50 + [0.1]*50, 100, 100)
        rd2 = PerSampleRiskData("b", [1.0]*50 + [0.0]*50, [0.7]*50 + [0.3]*50, 100, 100)
        out = generate_roc_overlay([rd1, rd2], tmp_path / "roc.png")
        assert out is not None
        assert out.exists()

    def test_empty_data_returns_none(self, tmp_path):
        assert generate_roc_overlay([], tmp_path / "roc.png") is None


# =============================================================================
# generate_calibration_overlay
# =============================================================================

class TestGenerateCalibrationOverlay:

    def test_produces_png(self, tmp_path):
        rd = PerSampleRiskData("model_a", [1.0]*50 + [0.0]*50,
                               [0.9]*50 + [0.1]*50, 100, 100)
        out = generate_calibration_overlay([rd], tmp_path / "cal.png")
        assert out is not None
        assert out.exists()
        assert out.stat().st_size > 0

    def test_multiple_models(self, tmp_path):
        rd1 = PerSampleRiskData("a", [1.0]*50 + [0.0]*50, [0.9]*50 + [0.1]*50, 100, 100)
        rd2 = PerSampleRiskData("b", [1.0]*50 + [0.0]*50, [0.7]*50 + [0.3]*50, 100, 100)
        out = generate_calibration_overlay([rd1, rd2], tmp_path / "cal.png")
        assert out is not None
        assert out.exists()

    def test_empty_data_returns_none(self, tmp_path):
        assert generate_calibration_overlay([], tmp_path / "cal.png") is None


# =============================================================================
# generate_prediction_histogram
# =============================================================================

class TestGeneratePredictionHistogram:

    def test_produces_png(self, tmp_path):
        rd = PerSampleRiskData("model_a", [1.0]*50 + [0.0]*50,
                               [0.9]*50 + [0.1]*50, 100, 100)
        out = generate_prediction_histogram([rd], tmp_path / "hist.png")
        assert out is not None
        assert out.exists()
        assert out.stat().st_size > 0

    def test_multiple_models(self, tmp_path):
        rd1 = PerSampleRiskData("a", [1.0]*50 + [0.0]*50, [0.9]*50 + [0.1]*50, 100, 100)
        rd2 = PerSampleRiskData("b", [1.0]*50 + [0.0]*50, [0.7]*50 + [0.3]*50, 100, 100)
        out = generate_prediction_histogram([rd1, rd2], tmp_path / "hist.png")
        assert out is not None
        assert out.exists()

    def test_empty_data_returns_none(self, tmp_path):
        assert generate_prediction_histogram([], tmp_path / "hist.png") is None

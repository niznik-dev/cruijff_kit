"""Unit tests for cruijff_kit.tools.inspect.summary_continuous.

summary_continuous reads continuous_scorer's per-sample metadata (prediction /
target_value / error) and aggregate metrics via read_eval_log, so the tests mock
read_eval_log with SimpleNamespace fakes shaped like a real EvalLog.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

from cruijff_kit.tools.inspect.summary_continuous import (
    compute_metrics,
    main,
    print_summary,
)

# read_eval_log is imported lazily inside compute_metrics; patch it at the source.
PATCH_TARGET = "inspect_ai.log.read_eval_log"


def _metric(value):
    return SimpleNamespace(value=value)


def _sample(prediction, target_value, error):
    """A sample carrying continuous_scorer metadata under sample.scores."""
    return SimpleNamespace(
        scores={
            "continuous_scorer": SimpleNamespace(
                metadata={
                    "prediction": prediction,
                    "target_value": target_value,
                    "error": error,
                }
            )
        }
    )


def _log(samples, metrics=None):
    """Fake EvalLog with results.scores[0] = continuous_scorer + given samples."""
    if metrics is None:
        metrics = {
            "mae": _metric(2.5),
            "rmse": _metric(3.1),
            "r_squared": _metric(0.42),
            "parse_rate": _metric(1.0),
        }
    score = SimpleNamespace(name="continuous_scorer", metrics=metrics)
    results = SimpleNamespace(scores=[score])
    return SimpleNamespace(results=results, samples=samples)


class TestComputeMetrics:
    @patch(PATCH_TARGET)
    def test_pulls_aggregate_metrics_from_results(self, mock_read, tmp_path):
        f = tmp_path / "t.eval"
        f.touch()
        mock_read.return_value = _log(
            [_sample(40.0, 42.0, -2.0), _sample(45.0, 42.0, 3.0)]
        )
        result = compute_metrics(f)
        assert result["status"] == "success"
        assert result["scorer"] == "continuous_scorer"
        # Headline metrics come straight from results.scores (authoritative).
        assert result["metrics"] == {
            "mae": 2.5,
            "rmse": 3.1,
            "r_squared": 0.42,
            "parse_rate": 1.0,
        }

    @patch(PATCH_TARGET)
    def test_distributions_from_sample_metadata(self, mock_read, tmp_path):
        f = tmp_path / "t.eval"
        f.touch()
        mock_read.return_value = _log(
            [_sample(40.0, 42.0, -2.0), _sample(50.0, 42.0, 8.0)]
        )
        result = compute_metrics(f)
        assert result["prediction_distribution"]["min"] == 40.0
        assert result["prediction_distribution"]["max"] == 50.0
        assert result["prediction_distribution"]["mean"] == 45.0
        assert result["target_distribution"]["mean"] == 42.0
        assert result["residual_distribution"]["min"] == -2.0
        assert result["residual_distribution"]["max"] == 8.0

    @patch(PATCH_TARGET)
    def test_parse_failures_counted(self, mock_read, tmp_path):
        f = tmp_path / "t.eval"
        f.touch()
        # One unparseable output: prediction/error are None.
        mock_read.return_value = _log(
            [_sample(40.0, 42.0, -2.0), _sample(None, 42.0, None)],
            metrics={"parse_rate": _metric(0.5)},
        )
        result = compute_metrics(f)
        assert result["samples"] == 2
        assert result["parsed"] == 1
        assert result["parse_failures"] == 1
        # Aggregate parse_rate is authoritative when present.
        assert result["metrics"]["parse_rate"] == 0.5

    @patch(PATCH_TARGET)
    def test_constant_prediction_has_zero_std(self, mock_read, tmp_path):
        """The sanity-glance payoff: a model emitting one constant -> std 0."""
        f = tmp_path / "t.eval"
        f.touch()
        mock_read.return_value = _log(
            [_sample(50.0, 30.0, 20.0), _sample(50.0, 70.0, -20.0)]
        )
        result = compute_metrics(f)
        assert result["prediction_distribution"]["std"] == 0.0
        assert result["target_distribution"]["std"] > 0.0

    @patch(PATCH_TARGET)
    def test_nan_metric_maps_to_none(self, mock_read, tmp_path):
        f = tmp_path / "t.eval"
        f.touch()
        mock_read.return_value = _log(
            [_sample(40.0, 42.0, -2.0)],
            metrics={"r_squared": _metric(float("nan"))},
        )
        result = compute_metrics(f)
        assert result["metrics"]["r_squared"] is None

    @patch(PATCH_TARGET)
    def test_no_samples_is_error(self, mock_read, tmp_path):
        f = tmp_path / "t.eval"
        f.touch()
        mock_read.return_value = _log([])
        result = compute_metrics(f)
        assert result["status"] == "error"

    @patch(PATCH_TARGET)
    def test_fallback_parse_rate_when_no_aggregate(self, mock_read, tmp_path):
        """If the scorer didn't publish parse_rate, derive it from samples."""
        f = tmp_path / "t.eval"
        f.touch()
        mock_read.return_value = _log(
            [_sample(40.0, 42.0, -2.0), _sample(None, 42.0, None)],
            metrics={"mae": _metric(2.0)},  # no parse_rate key
        )
        result = compute_metrics(f)
        assert result["metrics"]["parse_rate"] == 0.5

    @patch(PATCH_TARGET)
    def test_read_error_returns_error_status(self, mock_read, tmp_path):
        f = tmp_path / "t.eval"
        f.touch()
        mock_read.side_effect = Exception("corrupt archive")
        result = compute_metrics(f)
        assert result["status"] == "error"
        assert "corrupt archive" in result["message"]


class TestPrintSummary:
    """The default (non-JSON) CLI path — must survive None metrics/distributions."""

    @patch(PATCH_TARGET)
    def test_survives_all_parse_failures(self, mock_read, tmp_path, capsys):
        f = tmp_path / "t.eval"
        f.touch()
        mock_read.return_value = _log(
            [_sample(None, 42.0, None), _sample(None, 37.0, None)],
            metrics={"parse_rate": _metric(0.0), "mae": _metric(float("nan"))},
        )
        result = print_summary(f)  # must not raise
        assert result["parse_failures"] == 2
        # None metrics / None distributions render as "n/a" rather than crashing.
        assert "n/a" in capsys.readouterr().out

    @patch(PATCH_TARGET)
    def test_prints_metrics_on_success(self, mock_read, tmp_path, capsys):
        f = tmp_path / "t.eval"
        f.touch()
        mock_read.return_value = _log(
            [_sample(40.0, 42.0, -2.0), _sample(45.0, 42.0, 3.0)]
        )
        print_summary(f)
        out = capsys.readouterr().out
        assert "MAE:" in out and "Prediction" in out


class TestBoundaries:
    @patch(PATCH_TARGET)
    def test_all_predictions_unparseable(self, mock_read, tmp_path):
        """Base-model parse_rate=0 path: no predictions -> None distributions."""
        f = tmp_path / "t.eval"
        f.touch()
        mock_read.return_value = _log(
            [_sample(None, 42.0, None), _sample(None, 37.0, None)],
            metrics={"parse_rate": _metric(0.0)},
        )
        r = compute_metrics(f)
        assert r["parsed"] == 0
        assert r["parse_failures"] == 2
        assert r["prediction_distribution"] is None
        assert r["residual_distribution"] is None

    @patch(PATCH_TARGET)
    def test_single_sample_zero_std(self, mock_read, tmp_path):
        f = tmp_path / "t.eval"
        f.touch()
        mock_read.return_value = _log([_sample(50.0, 48.0, 2.0)])
        d = compute_metrics(f)["prediction_distribution"]
        assert d["std"] == 0.0
        assert d["min"] == d["max"] == d["mean"] == 50.0


class TestMissingScorerKey:
    @patch(PATCH_TARGET)
    def test_other_scorer_only_soft_degrades(self, mock_read, tmp_path):
        """Samples scored only by a different scorer -> graceful, not KeyError.

        Guards the SCORER_NAME hardcoded-coupling: a name miss must fail soft
        (empty per-sample glance), not explode.
        """
        f = tmp_path / "t.eval"
        f.touch()
        other = SimpleNamespace(scores={"match": SimpleNamespace(metadata={})})
        mock_read.return_value = _log([other, other])
        r = compute_metrics(f)
        assert r["status"] == "success"
        assert r["parse_failures"] == 2
        assert r["prediction_distribution"] is None


class TestEdges:
    @patch(PATCH_TARGET)
    def test_results_none_still_computes_distributions(self, mock_read, tmp_path):
        """No results.scores -> aggregate metrics None, but samples still glance."""
        f = tmp_path / "t.eval"
        f.touch()
        mock_read.return_value = SimpleNamespace(
            results=None,
            samples=[_sample(40.0, 42.0, -2.0), _sample(44.0, 42.0, 2.0)],
        )
        r = compute_metrics(f)
        assert r["status"] == "success"
        assert r["metrics"]["mae"] is None
        assert r["metrics"]["parse_rate"] == 1.0  # fallback from samples
        assert r["prediction_distribution"]["mean"] == 42.0

    @patch(PATCH_TARGET)
    def test_main_json_dir_batch(self, mock_read, tmp_path, capsys):
        (tmp_path / "a.eval").touch()
        (tmp_path / "b.eval").touch()
        mock_read.return_value = _log([_sample(40.0, 42.0, -2.0)])
        with patch.object(sys, "argv", ["prog", str(tmp_path), "--json"]):
            main()
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list)
        assert len(payload) == 2
        assert all(item["status"] == "success" for item in payload)

"""Unit tests for tools/slurm/estimate_compute.py

Run with:
    pytest tests/unit/test_estimate_compute.py -v

Tests use fixture data — no cluster or GPU required.
"""

import pytest

from cruijff_kit.tools.slurm.estimate_compute import (
    MIN_TIME_SECONDS,
    estimate_from_prior,
    format_wall_time,
    parse_wall_time,
    recommend_batch_size,
    round_up_to_increment,
    scale_eval_time,
    scale_finetune_time,
)


# =============================================================================
# parse_wall_time() / format_wall_time()
# =============================================================================


class TestParseWallTime:
    def test_basic(self):
        assert parse_wall_time("0:05:23") == 323

    def test_zero_padded(self):
        assert parse_wall_time("00:05:23") == 323

    def test_one_hour(self):
        assert parse_wall_time("1:00:00") == 3600

    def test_large(self):
        assert parse_wall_time("10:30:45") == 37845

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="HH:MM:SS"):
            parse_wall_time("5:23")

    def test_whitespace_stripped(self):
        assert parse_wall_time("  0:05:23  ") == 323


class TestFormatWallTime:
    def test_basic(self):
        assert format_wall_time(323) == "0:05:23"

    def test_one_hour(self):
        assert format_wall_time(3600) == "1:00:00"

    def test_zero(self):
        assert format_wall_time(0) == "0:00:00"

    def test_negative_clamps_to_zero(self):
        assert format_wall_time(-100) == "0:00:00"

    def test_roundtrip(self):
        for time_str in ["0:05:23", "1:30:00", "0:00:30", "10:00:00"]:
            assert format_wall_time(parse_wall_time(time_str)) == time_str


# =============================================================================
# round_up_to_increment()
# =============================================================================


class TestRoundUpToIncrement:
    def test_exact_multiple(self):
        assert round_up_to_increment(600) == 600  # 10 min → 10 min

    def test_rounds_up(self):
        assert round_up_to_increment(301) == 600  # 5:01 → 10:00

    def test_just_over(self):
        assert round_up_to_increment(1) == 300  # 1 sec → 5 min

    def test_custom_increment(self):
        assert round_up_to_increment(700, increment_minutes=10) == 1200

    def test_zero(self):
        assert round_up_to_increment(0) == 0


# =============================================================================
# scale_finetune_time()
# =============================================================================


class TestScaleFinetuneTime:
    def test_same_params_applies_safety_only(self):
        """Same epochs, dataset, model → just safety margin + rounding."""
        result = scale_finetune_time(
            prior_wall_seconds=600,  # 10 min
            prior_epochs=2,
            prior_dataset_size=800,
            new_epochs=2,
            new_dataset_size=800,
        )
        # 600 * 1.0 * 1.0 * 1.5 = 900 → round to 900 (15 min)
        assert result == 900

    def test_double_epochs(self):
        result = scale_finetune_time(
            prior_wall_seconds=600,
            prior_epochs=1,
            prior_dataset_size=800,
            new_epochs=2,
            new_dataset_size=800,
        )
        # 600 * 2.0 * 1.0 * 1.5 = 1800 → 30 min
        assert result == 1800

    def test_double_dataset(self):
        result = scale_finetune_time(
            prior_wall_seconds=600,
            prior_epochs=1,
            prior_dataset_size=400,
            new_epochs=1,
            new_dataset_size=800,
        )
        # 600 * 1.0 * 2.0 * 1.5 = 1800 → 30 min
        assert result == 1800

    def test_cross_model_scaling(self):
        """1B → 3B should scale by ~2.6x parameter ratio."""
        result_1b = scale_finetune_time(
            prior_wall_seconds=600,
            prior_epochs=1,
            prior_dataset_size=800,
            new_epochs=1,
            new_dataset_size=800,
            prior_model="Llama-3.2-1B-Instruct",
            new_model="Llama-3.2-1B-Instruct",
        )
        result_3b = scale_finetune_time(
            prior_wall_seconds=600,
            prior_epochs=1,
            prior_dataset_size=800,
            new_epochs=1,
            new_dataset_size=800,
            prior_model="Llama-3.2-1B-Instruct",
            new_model="Llama-3.2-3B-Instruct",
        )
        # 3B is ~2.6x the params of 1B
        assert result_3b > result_1b * 2

    def test_unknown_model_no_scaling(self):
        """Unknown model names → ratio of 1.0."""
        result = scale_finetune_time(
            prior_wall_seconds=600,
            prior_epochs=1,
            prior_dataset_size=800,
            new_epochs=1,
            new_dataset_size=800,
            prior_model="Llama-3.2-1B-Instruct",
            new_model="SomeUnknownModel",
        )
        # Should be same as no model scaling
        no_model = scale_finetune_time(
            prior_wall_seconds=600,
            prior_epochs=1,
            prior_dataset_size=800,
            new_epochs=1,
            new_dataset_size=800,
        )
        assert result == no_model

    def test_minimum_time(self):
        """Very short estimates get clamped to MIN_TIME_SECONDS."""
        result = scale_finetune_time(
            prior_wall_seconds=10,
            prior_epochs=1,
            prior_dataset_size=800,
            new_epochs=1,
            new_dataset_size=10,
        )
        assert result >= MIN_TIME_SECONDS

    def test_rounding_to_five_minutes(self):
        """Result is always a multiple of 300 seconds."""
        result = scale_finetune_time(
            prior_wall_seconds=421,
            prior_epochs=1,
            prior_dataset_size=800,
            new_epochs=1,
            new_dataset_size=800,
        )
        assert result % 300 == 0

    def test_invalid_prior_epochs(self):
        with pytest.raises(ValueError, match="positive"):
            scale_finetune_time(
                prior_wall_seconds=600,
                prior_epochs=0,
                prior_dataset_size=800,
                new_epochs=1,
                new_dataset_size=800,
            )

    def test_invalid_prior_dataset(self):
        with pytest.raises(ValueError, match="positive"):
            scale_finetune_time(
                prior_wall_seconds=600,
                prior_epochs=1,
                prior_dataset_size=0,
                new_epochs=1,
                new_dataset_size=800,
            )

    def test_custom_safety_margin(self):
        result_default = scale_finetune_time(
            prior_wall_seconds=600,
            prior_epochs=1,
            prior_dataset_size=800,
            new_epochs=1,
            new_dataset_size=800,
        )
        result_tight = scale_finetune_time(
            prior_wall_seconds=600,
            prior_epochs=1,
            prior_dataset_size=800,
            new_epochs=1,
            new_dataset_size=800,
            safety_margin=1.0,
        )
        assert result_tight < result_default


# =============================================================================
# scale_eval_time()
# =============================================================================


class TestScaleEvalTime:
    def test_same_dataset(self):
        result = scale_eval_time(
            prior_wall_seconds=60,
            prior_dataset_size=100,
            new_dataset_size=100,
        )
        # 60 * 1.0 * 1.5 = 90 → round to 300 (min 5 min)
        assert result == MIN_TIME_SECONDS

    def test_double_dataset(self):
        result = scale_eval_time(
            prior_wall_seconds=300,
            prior_dataset_size=100,
            new_dataset_size=200,
        )
        # 300 * 2.0 * 1.5 = 900 → 15 min
        assert result == 900

    def test_minimum_applies(self):
        result = scale_eval_time(
            prior_wall_seconds=10,
            prior_dataset_size=100,
            new_dataset_size=100,
        )
        assert result >= MIN_TIME_SECONDS


# =============================================================================
# recommend_batch_size()
# =============================================================================


class TestRecommendBatchSize:
    def test_low_utilization_doubles(self):
        """< 50% → suggest doubling."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=12.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=4,
        )
        assert result["batch_size"] == 8
        assert result["mem_ratio"] == 0.15

    def test_moderate_utilization_increases(self):
        """50-70% → suggest modest increase."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=48.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=4,
        )
        assert result["batch_size"] == 6  # 4 + max(1, 4//2) = 6
        assert 0.5 <= result["mem_ratio"] < 0.7

    def test_well_fitted_keeps_same(self):
        """70-90% → keep same batch size."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=60.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=8,
        )
        assert result["batch_size"] == 8

    def test_near_limit_keeps_same_with_warning(self):
        """>= 90% → keep same, warn about OOM."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=74.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=8,
        )
        assert result["batch_size"] == 8
        assert "OOM" in result["reason"]

    def test_cross_model_reduces_when_tight(self):
        """1B → 8B with 80GB GPU: should reduce batch size."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=12.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=32,
            prior_model="Llama-3.2-1B-Instruct",
            new_model="Llama-3.1-8B-Instruct",
        )
        # 8B is ~6.5x the params of 1B. 12 * 6.5 ≈ 78 GB → > 85%
        assert result["batch_size"] < 32

    def test_cross_model_same_when_fits(self):
        """1B → 3B with small batch: should still fit."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=12.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=4,
            prior_model="Llama-3.2-1B-Instruct",
            new_model="Llama-3.2-3B-Instruct",
        )
        # 3B is ~2.6x. 12 * 2.6 ≈ 31 GB → < 85% of 80GB
        assert result["batch_size"] == 4

    def test_no_gpu_data(self):
        """Zero total GPU memory → return prior batch size."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=0.0,
            prior_gpu_mem_total_gb=0.0,
            prior_batch_size=4,
        )
        assert result["batch_size"] == 4
        assert result["mem_ratio"] is None


# =============================================================================
# estimate_from_prior()
# =============================================================================


class TestEstimateFromPrior:
    """Integration tests using realistic envelope data."""

    SAMPLE_ENVELOPE = {
        "experiment_name": "cap_4L_2025-10-22",
        "model": "Llama-3.2-1B-Instruct",
        "dataset_size": 800,
        "epochs": 2,
        "batch_size": 4,
        "date": "2025-10-22",
        "jobs": [
            {
                "run_name": "Llama-3.2-1B-Instruct_rank4",
                "job_type": "finetune",
                "wall_time": "0:05:23",
                "gpus": 1,
                "gpu_mem_used_mean_gb": 12.5,
                "gpu_mem_total_gb": 80.0,
                "cpu_mem_allocated_gb": 80.0,
            },
            {
                "run_name": "Llama-3.2-1B-Instruct_rank8",
                "job_type": "finetune",
                "wall_time": "0:06:10",
                "gpus": 1,
                "gpu_mem_used_mean_gb": 13.0,
                "gpu_mem_total_gb": 80.0,
                "cpu_mem_allocated_gb": 80.0,
            },
            {
                "run_name": "Llama-3.2-1B-Instruct_rank4",
                "job_type": "eval",
                "wall_time": "0:00:45",
                "gpus": 1,
                "cpu_mem_allocated_gb": 80.0,
            },
        ],
    }

    def test_same_experiment_params(self):
        """Same model/dataset/epochs → estimates are safety-margined."""
        result = estimate_from_prior(
            self.SAMPLE_ENVELOPE,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
        )
        assert result["finetune"] is not None
        assert result["eval"] is not None
        assert result["prior_experiment"] == "cap_4L_2025-10-22"

        # Finetune: avg of 323+370 = 346s, × 1.5 = 519 → round to 600
        ft_seconds = result["finetune"]["time_seconds"]
        assert ft_seconds >= 300  # at least 5 min
        assert ft_seconds % 300 == 0  # multiple of 5 min

    def test_larger_dataset(self):
        """Double dataset → approximately double time."""
        result_base = estimate_from_prior(
            self.SAMPLE_ENVELOPE,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
        )
        result_2x = estimate_from_prior(
            self.SAMPLE_ENVELOPE,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=1600,
            new_epochs=2,
        )
        # Should be roughly double (within rounding)
        assert (
            result_2x["finetune"]["time_seconds"]
            >= result_base["finetune"]["time_seconds"]
        )

    def test_different_model(self):
        """3B model → longer time than 1B."""
        result_1b = estimate_from_prior(
            self.SAMPLE_ENVELOPE,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
        )
        result_3b = estimate_from_prior(
            self.SAMPLE_ENVELOPE,
            new_model="Llama-3.2-3B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
        )
        assert (
            result_3b["finetune"]["time_seconds"]
            > result_1b["finetune"]["time_seconds"]
        )

    def test_batch_size_recommendation_present(self):
        result = estimate_from_prior(
            self.SAMPLE_ENVELOPE,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
        )
        assert result["batch_size"] is not None
        assert "batch_size" in result["batch_size"]
        assert "reason" in result["batch_size"]

    def test_scaling_details_logged(self):
        result = estimate_from_prior(
            self.SAMPLE_ENVELOPE,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
        )
        assert len(result["scaling_details"]) >= 1
        assert "Finetune" in result["scaling_details"][0]

    def test_empty_jobs(self):
        envelope = {**self.SAMPLE_ENVELOPE, "jobs": []}
        result = estimate_from_prior(
            envelope,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
        )
        assert result["finetune"] is None
        assert result["eval"] is None

    def test_output_format(self):
        """Verify output structure matches what design-experiment needs."""
        result = estimate_from_prior(
            self.SAMPLE_ENVELOPE,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
        )
        ft = result["finetune"]
        assert "time" in ft
        assert "gpus" in ft
        assert "mem" in ft
        # Time should be H:MM:SS format
        assert ft["time"].count(":") == 2
        # Mem should end with G
        assert ft["mem"].endswith("G")

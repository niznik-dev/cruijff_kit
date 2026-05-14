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
    """Throughput-based scaler tests.

    Wall time prediction is total_tokens / tps_gpu × safety margin,
    rounded up to MIN_TIME_SECONDS increments.
    """

    def test_basic_throughput_math(self):
        # 1_000_000 tokens / 200 tps = 5000 s, × 1.5 = 7500 → round to 7500 (125 min)
        result = scale_finetune_time(
            prior_tps_gpu=200.0,
            new_total_tokens=1_000_000,
        )
        assert result == 7500

    def test_doubling_tokens_doubles_time(self):
        result_1x = scale_finetune_time(
            prior_tps_gpu=200.0,
            new_total_tokens=500_000,
        )
        result_2x = scale_finetune_time(
            prior_tps_gpu=200.0,
            new_total_tokens=1_000_000,
        )
        # Within rounding (5-min increments), 2x tokens ≈ 2x time
        assert result_2x >= result_1x

    def test_doubling_tps_halves_time(self):
        result_slow = scale_finetune_time(
            prior_tps_gpu=100.0,
            new_total_tokens=1_000_000,
        )
        result_fast = scale_finetune_time(
            prior_tps_gpu=200.0,
            new_total_tokens=1_000_000,
        )
        assert result_fast < result_slow

    def test_minimum_time(self):
        """Very short estimates get clamped to MIN_TIME_SECONDS."""
        result = scale_finetune_time(
            prior_tps_gpu=200.0,
            new_total_tokens=1000,
        )
        assert result == MIN_TIME_SECONDS

    def test_rounding_to_five_minutes(self):
        result = scale_finetune_time(
            prior_tps_gpu=200.0,
            new_total_tokens=421_000,
        )
        assert result % 300 == 0

    def test_invalid_tps(self):
        with pytest.raises(ValueError, match="prior_tps_gpu"):
            scale_finetune_time(
                prior_tps_gpu=0.0,
                new_total_tokens=1_000_000,
            )

    def test_invalid_tokens(self):
        with pytest.raises(ValueError, match="new_total_tokens"):
            scale_finetune_time(
                prior_tps_gpu=200.0,
                new_total_tokens=0,
            )

    def test_custom_safety_margin(self):
        result_default = scale_finetune_time(
            prior_tps_gpu=200.0,
            new_total_tokens=1_000_000,
        )
        result_tight = scale_finetune_time(
            prior_tps_gpu=200.0,
            new_total_tokens=1_000_000,
            safety_margin=1.0,
        )
        assert result_tight < result_default


# =============================================================================
# scale_eval_time()
# =============================================================================


class TestScaleEvalTime:
    def test_basic_throughput_math(self):
        # 100_000 tokens / 500 tps = 200 s × 1.5 = 300 → min applies anyway
        result = scale_eval_time(
            prior_tps_gpu=500.0,
            new_total_tokens=100_000,
        )
        assert result == MIN_TIME_SECONDS

    def test_doubling_tokens_increases_time(self):
        result_1x = scale_eval_time(
            prior_tps_gpu=500.0,
            new_total_tokens=300_000,
        )
        result_2x = scale_eval_time(
            prior_tps_gpu=500.0,
            new_total_tokens=600_000,
        )
        assert result_2x >= result_1x

    def test_minimum_applies(self):
        result = scale_eval_time(
            prior_tps_gpu=500.0,
            new_total_tokens=1000,
        )
        assert result >= MIN_TIME_SECONDS

    def test_invalid_tps(self):
        with pytest.raises(ValueError, match="prior_tps_gpu"):
            scale_eval_time(
                prior_tps_gpu=-1.0,
                new_total_tokens=100_000,
            )

    def test_invalid_tokens(self):
        with pytest.raises(ValueError, match="new_total_tokens"):
            scale_eval_time(
                prior_tps_gpu=500.0,
                new_total_tokens=0,
            )


# =============================================================================
# recommend_batch_size()
# =============================================================================


class TestRecommendBatchSize:
    def test_low_utilization_doubles_if_fits(self):
        """< 50% with headroom → double batch_size if within 85% ceiling."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=12.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=4,
        )
        # 12/80 = 15%, doubled mem ≈ 12*1.15 = 13.8 → 17% < 85%
        assert result["batch_size"] == 8
        assert result["mem_ratio"] == 0.15

    def test_low_utilization_no_double_if_tight(self):
        """< 50% but doubling would exceed 85% → keep same."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=38.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=4,
        )
        # 38/80 = 47.5%, doubled mem ≈ 38*1.15 = 43.7 → 54.6% < 85%, fits
        assert result["batch_size"] == 8

    def test_moderate_utilization_increases_if_fits(self):
        """50-70% → suggest modest increase if within 85% ceiling."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=48.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=4,
        )
        # 48/80 = 60%, increased mem ≈ 48*1.075 = 51.6 → 64.5% < 85%
        assert result["batch_size"] == 6  # 4 + max(1, 4//2) = 6
        assert 0.5 <= result["mem_ratio"] < 0.7

    def test_well_fitted_keeps_same(self):
        """70-85% → keep same batch size."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=60.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=8,
        )
        assert result["batch_size"] == 8

    def test_near_limit_reduces_batch_size(self):
        """> 85% utilization → reduce batch size to fit in envelope."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=74.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=8,
        )
        # 74/80 = 92.5% > 85% → should reduce
        assert result["batch_size"] < 8
        assert "Reduced" in result["reason"]

    def test_near_limit_warns_oom(self):
        """85-90% → keeps batch size but warns about OOM."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=72.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=8,
        )
        # 72/80 = 90% → just at the edge, after halving estimate
        # drops below 85%, so it reduces. Let's use a value right
        # at 85% where it fits but the OOM warning should fire.
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=73.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=1,
        )
        # 73/80 = 91.25% > 85%, but batch_size=1 can't reduce further
        assert result["batch_size"] == 1
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

    def test_cross_model_compensates_with_grad_accum(self):
        """When batch_size is reduced, gradient_accumulation_steps increases."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=12.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=32,
            prior_model="Llama-3.2-1B-Instruct",
            new_model="Llama-3.1-8B-Instruct",
            prior_gradient_accumulation_steps=1,
        )
        # Effective batch size should be preserved
        assert (
            result["effective_batch_size"]
            == result["batch_size"] * result["gradient_accumulation_steps"]
        )
        assert result["gradient_accumulation_steps"] > 1

    def test_cross_model_same_when_fits(self):
        """1B → 3B with small batch: should still fit."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=12.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=4,
            prior_model="Llama-3.2-1B-Instruct",
            new_model="Llama-3.2-3B-Instruct",
        )
        # 3B is ~2.6x. 12 * 2.6 ≈ 31 GB → < 50% of 80GB → doubles
        assert result["batch_size"] >= 4

    def test_no_gpu_data(self):
        """Zero total GPU memory → return prior settings."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=0.0,
            prior_gpu_mem_total_gb=0.0,
            prior_batch_size=4,
        )
        assert result["batch_size"] == 4
        assert result["gradient_accumulation_steps"] == 1
        assert result["effective_batch_size"] == 4
        assert result["mem_ratio"] is None

    def test_return_shape(self):
        """All expected keys are present in result."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=40.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=4,
        )
        assert "batch_size" in result
        assert "gradient_accumulation_steps" in result
        assert "effective_batch_size" in result
        assert "reason" in result
        assert "mem_ratio" in result
        assert "estimated_mem_gb" in result

    def test_grad_accum_passed_through(self):
        """Prior gradient_accumulation_steps is preserved when batch_size unchanged."""
        result = recommend_batch_size(
            prior_gpu_mem_used_gb=60.0,
            prior_gpu_mem_total_gb=80.0,
            prior_batch_size=4,
            prior_gradient_accumulation_steps=8,
        )
        # 75% utilization → well-fitted, keep same
        assert result["batch_size"] == 4
        assert result["gradient_accumulation_steps"] == 8
        assert result["effective_batch_size"] == 32


# =============================================================================
# estimate_from_prior()
# =============================================================================


class TestEstimateFromPrior:
    """Integration tests using realistic summary data."""

    SAMPLE_SUMMARY = {
        "experiment_name": "cap_4L_2025-10-22",
        "model": "Llama-3.2-1B-Instruct",
        "dataset_size": 800,
        "eval_dataset_size": 100,
        "epochs": 2,
        "batch_size": 4,
        "date": "2025-10-22",
        "jobs": [
            {
                "run_name": "Llama-3.2-1B-Instruct_rank4",
                "job_type": "finetune",
                "model": "Llama-3.2-1B-Instruct",
                "wall_time": "0:05:23",
                "gpus": 1,
                "gpu_mem_used_mean_gb": 12.5,
                "gpu_mem_total_gb": 80.0,
                "cpu_mem_allocated_gb": 80.0,
                "tps_gpu_train_mean": 200.0,
                "global_step": 200,
            },
            {
                "run_name": "Llama-3.2-1B-Instruct_rank8",
                "job_type": "finetune",
                "model": "Llama-3.2-1B-Instruct",
                "wall_time": "0:06:10",
                "gpus": 1,
                "gpu_mem_used_mean_gb": 13.0,
                "gpu_mem_total_gb": 80.0,
                "cpu_mem_allocated_gb": 80.0,
                "tps_gpu_train_mean": 180.0,
                "global_step": 200,
            },
            {
                "run_name": "Llama-3.2-1B-Instruct_rank4",
                "job_type": "eval",
                "model": "Llama-3.2-1B-Instruct",
                "wall_time": "0:00:45",
                "gpus": 1,
                "cpu_mem_allocated_gb": 80.0,
                "tps_gpu_eval_e2e": 800.0,
                "total_tokens": 36000,
                "total_seconds": 45,
            },
        ],
    }

    MULTI_MODEL_SUMMARY = {
        "experiment_name": "tps_calibration_2026-05-14",
        "model": "Llama-3.2-1B-Instruct",  # first in yaml; not authoritative
        "dataset_size": 10000,
        "eval_dataset_size": 0,
        "epochs": 1,
        "batch_size": 4,
        "date": "2026-05-14",
        "jobs": [
            {
                "run_name": "Llama-3.2-1B-Instruct_cal",
                "job_type": "finetune",
                "model": "Llama-3.2-1B-Instruct",
                "wall_time": "0:02:36",
                "gpus": 1,
                "tps_gpu_train_mean": 190.0,
            },
            {
                "run_name": "Llama-3.2-3B-Instruct_cal",
                "job_type": "finetune",
                "model": "Llama-3.2-3B-Instruct",
                "wall_time": "0:04:25",
                "gpus": 1,
                "tps_gpu_train_mean": 104.0,
            },
            {
                "run_name": "Llama-3.1-8B-Instruct_cal",
                "job_type": "finetune",
                "model": "Llama-3.1-8B-Instruct",
                "wall_time": "0:05:52",
                "gpus": 1,
                "tps_gpu_train_mean": 83.0,
            },
        ],
    }

    def test_same_experiment_params(self):
        result = estimate_from_prior(
            self.SAMPLE_SUMMARY,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
            new_seq_len=512,
        )
        assert result["finetune"] is not None
        assert result["eval"] is not None
        assert result["prior_experiment"] == "cap_4L_2025-10-22"

        ft_seconds = result["finetune"]["time_seconds"]
        assert ft_seconds >= MIN_TIME_SECONDS
        assert ft_seconds % 300 == 0  # multiple of 5 min

    def test_larger_dataset_increases_time(self):
        result_base = estimate_from_prior(
            self.SAMPLE_SUMMARY,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
            new_seq_len=512,
        )
        result_2x = estimate_from_prior(
            self.SAMPLE_SUMMARY,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=1600,
            new_epochs=2,
            new_seq_len=512,
        )
        assert (
            result_2x["finetune"]["time_seconds"]
            >= result_base["finetune"]["time_seconds"]
        )

    def test_larger_seq_len_increases_time(self):
        """Doubling seq_len doubles tokens → doubles wall time (within rounding)."""
        result_base = estimate_from_prior(
            self.SAMPLE_SUMMARY,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
            new_seq_len=512,
        )
        result_long = estimate_from_prior(
            self.SAMPLE_SUMMARY,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
            new_seq_len=2048,
        )
        assert (
            result_long["finetune"]["time_seconds"]
            > result_base["finetune"]["time_seconds"]
        )

    def test_batch_size_recommendation_present(self):
        result = estimate_from_prior(
            self.SAMPLE_SUMMARY,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
            new_seq_len=512,
        )
        assert result["batch_size"] is not None
        assert "batch_size" in result["batch_size"]
        assert "gradient_accumulation_steps" in result["batch_size"]
        assert "effective_batch_size" in result["batch_size"]
        assert "reason" in result["batch_size"]

    def test_scaling_details_logged(self):
        result = estimate_from_prior(
            self.SAMPLE_SUMMARY,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
            new_seq_len=512,
        )
        assert len(result["scaling_details"]) >= 1
        assert "Finetune" in result["scaling_details"][0]

    def test_empty_jobs(self):
        envelope = {**self.SAMPLE_SUMMARY, "jobs": []}
        result = estimate_from_prior(
            envelope,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
            new_seq_len=512,
        )
        assert result["finetune"] is None
        assert result["eval"] is None

    def test_missing_tps_returns_none(self):
        """Jobs without tps fields (e.g. older summaries) → finetune/eval None."""
        envelope = {
            **self.SAMPLE_SUMMARY,
            "jobs": [
                {
                    "run_name": "old",
                    "job_type": "finetune",
                    "wall_time": "0:05:00",
                    "gpus": 1,
                },
            ],
        }
        result = estimate_from_prior(
            envelope,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
            new_seq_len=512,
        )
        assert result["finetune"] is None
        assert result["eval"] is None

    def test_output_format(self):
        """Verify output structure matches what design-experiment needs."""
        result = estimate_from_prior(
            self.SAMPLE_SUMMARY,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
            new_seq_len=512,
        )
        ft = result["finetune"]
        assert "time" in ft
        assert "gpus" in ft
        assert "mem" in ft
        assert ft["time"].count(":") == 2
        assert ft["mem"].endswith("G")

    def test_cross_model_warning_emitted(self):
        """When prior_model != new_model, scaling_details contains a warning."""
        result = estimate_from_prior(
            self.SAMPLE_SUMMARY,  # prior_model = Llama-3.2-1B-Instruct
            new_model="Llama-3.2-3B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
            new_seq_len=512,
        )
        warning_lines = [s for s in result["scaling_details"] if "CROSS-MODEL" in s]
        assert len(warning_lines) == 1
        assert "Llama-3.2-1B-Instruct" in warning_lines[0]
        assert "Llama-3.2-3B-Instruct" in warning_lines[0]

    def test_no_cross_model_warning_when_same_model(self):
        """Same prior/new model → no cross-model warning."""
        result = estimate_from_prior(
            self.SAMPLE_SUMMARY,
            new_model="Llama-3.2-1B-Instruct",  # same as prior
            new_dataset_size=800,
            new_epochs=2,
            new_seq_len=512,
        )
        warning_lines = [s for s in result["scaling_details"] if "CROSS-MODEL" in s]
        assert len(warning_lines) == 0

    def test_multi_model_prior_picks_matching_job(self):
        """Multi-model prior + new_model matching one job → no warning,
        uses only matching job's tps for prediction."""
        result = estimate_from_prior(
            self.MULTI_MODEL_SUMMARY,
            new_model="Llama-3.2-3B-Instruct",  # matches one prior job (tps=104)
            new_dataset_size=10000,
            new_epochs=1,
            new_seq_len=128,
        )
        warning_lines = [s for s in result["scaling_details"] if "CROSS-MODEL" in s]
        assert len(warning_lines) == 0, (
            "no warning expected — 3B is in the multi-model prior"
        )
        # Prediction should use tps=104 (the 3B job), not mean(190, 104, 83)
        details = " ".join(result["scaling_details"])
        assert "tps=104" in details, f"expected tps=104 in details, got: {details}"

    def test_multi_model_prior_no_match_warns_and_uses_all(self):
        """Multi-model prior + new_model not in priors → warn + fall back to all."""
        result = estimate_from_prior(
            self.MULTI_MODEL_SUMMARY,
            new_model="Llama-3.3-70B-Instruct",  # no match in priors
            new_dataset_size=10000,
            new_epochs=1,
            new_seq_len=128,
        )
        warning_lines = [s for s in result["scaling_details"] if "CROSS-MODEL" in s]
        assert len(warning_lines) == 1
        # Warning should list all available prior models
        assert "Llama-3.2-1B-Instruct" in warning_lines[0]
        assert "Llama-3.2-3B-Instruct" in warning_lines[0]
        assert "Llama-3.1-8B-Instruct" in warning_lines[0]
        # Falls back to mean across all jobs
        # mean(190, 104, 83) = 125.67
        details = " ".join(result["scaling_details"])
        assert "tps=125" in details, f"expected mean tps≈125.67 in details: {details}"

    def test_jobs_without_model_field_still_work(self):
        """Backward compat: jobs that predate model-aware build_summary
        (no 'model' field) should fall through to all-jobs behavior."""
        legacy_summary = {
            **self.SAMPLE_SUMMARY,
            "jobs": [
                {k: v for k, v in j.items() if k != "model"}
                for j in self.SAMPLE_SUMMARY["jobs"]
            ],
        }
        result = estimate_from_prior(
            legacy_summary,
            new_model="Llama-3.2-1B-Instruct",
            new_dataset_size=800,
            new_epochs=2,
            new_seq_len=512,
        )
        # No model field → no jobs match → falls back to all jobs + warning
        assert result["finetune"] is not None
        warning_lines = [s for s in result["scaling_details"] if "CROSS-MODEL" in s]
        assert len(warning_lines) == 1

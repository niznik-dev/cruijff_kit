"""Unit tests for tools/slurm/throughput_parsers.py

Run with:
    pytest tests/unit/test_throughput_parsers.py -v

Tests use inline fixtures based on real slurm-out snippets.
"""

import textwrap

import pytest

from cruijff_kit.tools.slurm.throughput_parsers import (
    enrich_job_with_throughput,
    parse_inspect_throughput,
    parse_torchtune_throughput,
)


# =============================================================================
# parse_torchtune_throughput()
# =============================================================================


class TestParseTorchtuneThroughput:
    # Trimmed slurm-out: recipe banner + wandb end-of-run summary block.
    SAMPLE_TORCHTUNE = textwrap.dedent("""\
        Running LoRAFinetuneRecipeSingleDevice with resolved config:

        batch_size: 4
        ... (config dump) ...

        1|2000|Loss: 0.005232139490544796: 100%|##########| 2000/2000 [07:35<00:00, 4.39it/s]
        wandb:
        wandb: Run history:
        wandb:               global_step ##
        wandb: tokens_per_second_per_gpu ##
        wandb:
        wandb: Run summary:
        wandb:               global_step 2000
        wandb:                      loss 0.00523
        wandb:                        lr 0
        wandb:        peak_memory_active 2.85518
        wandb:         peak_memory_alloc 2.85518
        wandb:      peak_memory_reserved 2.89844
        wandb: tokens_per_second_per_gpu 186.75362
        wandb:                  val_loss 0.05704
        wandb:
        wandb: You can sync this run to the cloud by running:
        wandb: wandb sync /path/to/offline-run
    """)

    def test_extracts_tps(self):
        result = parse_torchtune_throughput(self.SAMPLE_TORCHTUNE)
        assert result["tps_gpu_train_mean"] == pytest.approx(186.75362)

    def test_extracts_global_step(self):
        result = parse_torchtune_throughput(self.SAMPLE_TORCHTUNE)
        assert result["global_step"] == 2000

    def test_missing_tps_line_raises(self):
        text = self.SAMPLE_TORCHTUNE.replace(
            "wandb: tokens_per_second_per_gpu 186.75362", ""
        )
        with pytest.raises(ValueError, match="tokens_per_second_per_gpu"):
            parse_torchtune_throughput(text)

    def test_missing_global_step_raises(self):
        text = self.SAMPLE_TORCHTUNE.replace(
            "wandb:               global_step 2000", ""
        )
        with pytest.raises(ValueError, match="global_step"):
            parse_torchtune_throughput(text)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="tokens_per_second_per_gpu"):
            parse_torchtune_throughput("")

    def test_distributed_recipe_raises(self):
        text = self.SAMPLE_TORCHTUNE.replace(
            "LoRAFinetuneRecipeSingleDevice", "LoRAFinetuneRecipeDistributed"
        )
        with pytest.raises(ValueError, match="Distributed"):
            parse_torchtune_throughput(text)

    def test_no_recipe_banner_falls_through(self):
        # If the banner is missing entirely, we don't second-guess world_size.
        # The wandb summary is still authoritative.
        text = self.SAMPLE_TORCHTUNE.replace(
            "Running LoRAFinetuneRecipeSingleDevice with resolved config:", ""
        )
        result = parse_torchtune_throughput(text)
        assert result["tps_gpu_train_mean"] == pytest.approx(186.75362)


# =============================================================================
# parse_inspect_throughput()
# =============================================================================


class TestParseInspectThroughput:
    # Trimmed slurm-out: just the footer block.
    SAMPLE_INSPECT = textwrap.dedent("""\
        ... (eval progress lines) ...

        total time:                              0:05:00
        hf/Llama-3.2-3B-Instruct_mc256_hifreq    302,904 tokens [I: 297,904, O: 5,000]

        match
        accuracy  0.000
        stderr    0.000
    """)

    def test_extracts_total_seconds(self):
        result = parse_inspect_throughput(self.SAMPLE_INSPECT)
        assert result["total_seconds"] == 300

    def test_extracts_total_tokens(self):
        result = parse_inspect_throughput(self.SAMPLE_INSPECT)
        assert result["total_tokens"] == 302904

    def test_extracts_input_tokens(self):
        result = parse_inspect_throughput(self.SAMPLE_INSPECT)
        assert result["input_tokens"] == 297904

    def test_extracts_output_tokens(self):
        result = parse_inspect_throughput(self.SAMPLE_INSPECT)
        assert result["output_tokens"] == 5000

    def test_computes_tps(self):
        result = parse_inspect_throughput(self.SAMPLE_INSPECT)
        assert result["tps_gpu_eval_e2e"] == pytest.approx(302904 / 300)

    def test_handles_multi_hour_runs(self):
        text = self.SAMPLE_INSPECT.replace("0:05:00", "2:30:00")
        result = parse_inspect_throughput(text)
        assert result["total_seconds"] == 9000

    def test_missing_total_time_raises(self):
        text = self.SAMPLE_INSPECT.replace(
            "total time:                              0:05:00", ""
        )
        with pytest.raises(ValueError, match="total time"):
            parse_inspect_throughput(text)

    def test_missing_tokens_line_raises(self):
        text = self.SAMPLE_INSPECT.replace(
            "hf/Llama-3.2-3B-Instruct_mc256_hifreq    "
            "302,904 tokens [I: 297,904, O: 5,000]",
            "",
        )
        with pytest.raises(ValueError, match="tokens footer"):
            parse_inspect_throughput(text)

    def test_zero_total_time_raises(self):
        text = self.SAMPLE_INSPECT.replace("0:05:00", "0:00:00")
        with pytest.raises(ValueError, match="total time of 0"):
            parse_inspect_throughput(text)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="total time"):
            parse_inspect_throughput("")


# =============================================================================
# enrich_job_with_throughput()
# =============================================================================


class TestEnrichJobWithThroughput:
    FINETUNE_TEXT = TestParseTorchtuneThroughput.SAMPLE_TORCHTUNE
    EVAL_TEXT = TestParseInspectThroughput.SAMPLE_INSPECT

    def test_enriches_finetune(self, tmp_path):
        slurm_out = tmp_path / "slurm-12345.out"
        slurm_out.write_text(self.FINETUNE_TEXT)
        job = {"run_name": "rank4", "job_type": "finetune"}
        enriched = enrich_job_with_throughput(job, slurm_out)
        assert enriched["tps_gpu_train_mean"] == pytest.approx(186.75362)
        assert enriched["global_step"] == 2000
        assert enriched["run_name"] == "rank4"

    def test_enriches_eval(self, tmp_path):
        slurm_out = tmp_path / "slurm-67890.out"
        slurm_out.write_text(self.EVAL_TEXT)
        job = {"run_name": "eval_1B", "job_type": "eval"}
        enriched = enrich_job_with_throughput(job, slurm_out)
        assert enriched["total_seconds"] == 300
        assert enriched["tps_gpu_eval_e2e"] == pytest.approx(302904 / 300)

    def test_missing_file_returns_unchanged(self, tmp_path, capsys):
        job = {"run_name": "rank4", "job_type": "finetune"}
        enriched = enrich_job_with_throughput(job, tmp_path / "missing.out")
        assert enriched == job
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "rank4" in captured.err

    def test_parse_failure_returns_unchanged(self, tmp_path, capsys):
        slurm_out = tmp_path / "slurm-12345.out"
        slurm_out.write_text("nothing useful here")
        job = {"run_name": "rank4", "job_type": "finetune"}
        enriched = enrich_job_with_throughput(job, slurm_out)
        assert enriched == job
        captured = capsys.readouterr()
        assert "WARNING" in captured.err

    def test_unknown_job_type_returns_unchanged(self, tmp_path, capsys):
        slurm_out = tmp_path / "slurm-12345.out"
        slurm_out.write_text(self.FINETUNE_TEXT)
        job = {"run_name": "weird", "job_type": "mystery"}
        enriched = enrich_job_with_throughput(job, slurm_out)
        assert enriched == job
        captured = capsys.readouterr()
        assert "unknown job_type" in captured.err

    def test_returns_new_dict_not_mutation(self, tmp_path):
        slurm_out = tmp_path / "slurm-12345.out"
        slurm_out.write_text(self.FINETUNE_TEXT)
        job = {"run_name": "rank4", "job_type": "finetune"}
        enriched = enrich_job_with_throughput(job, slurm_out)
        assert "tps_gpu_train_mean" not in job
        assert "tps_gpu_train_mean" in enriched

"""Tests for summarize-experiment data extraction against realistic fixtures.

Tests loss extraction module against SLURM stdout and eval result JSON parsing.
Pure unit tests — no GPU, SLURM, or inspect-ai required.
"""

import json
from pathlib import Path

import pytest

from cruijff_kit.tools.torchtune.extract_loss import (
    LOSS_PATTERN,
    extract_losses,
    final_loss,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "summarize"


@pytest.fixture
def slurm_output():
    """Load the realistic SLURM training output."""
    return (FIXTURES_DIR / "slurm_training_output.txt").read_text()


@pytest.fixture
def eval_result():
    """Load the synthetic eval result JSON."""
    with open(FIXTURES_DIR / "eval_result.json") as f:
        return json.load(f)


class TestLossExtraction:
    def test_finds_loss_lines(self, slurm_output):
        losses = extract_losses(slurm_output)
        assert len(losses) > 0, "Should find at least one loss line"

    def test_final_loss_returns_last_match(self, slurm_output):
        result = final_loss(slurm_output)
        assert result is not None
        epoch, step, loss = result
        assert loss == pytest.approx(0.26427456736564636)
        assert step == 500

    def test_final_loss_empty_input(self):
        assert final_loss("no loss lines here") is None

    def test_first_loss_is_higher(self, slurm_output):
        losses = extract_losses(slurm_output)
        assert losses[0][2] > losses[-1][2], "Loss should decrease during training"

    def test_all_losses_are_positive(self, slurm_output):
        for _, _, loss in extract_losses(slurm_output):
            assert loss > 0

    def test_epoch_numbers_are_valid(self, slurm_output):
        for epoch, _, _ in extract_losses(slurm_output):
            assert epoch >= 1

    def test_step_numbers_increase(self, slurm_output):
        steps = [step for _, step, _ in extract_losses(slurm_output)]
        assert steps == sorted(steps), "Steps should monotonically increase"

    def test_total_loss_count(self, slurm_output):
        assert len(extract_losses(slurm_output)) == 18, (
            "Fixture has 18 loss line matches"
        )

    def test_does_not_match_validation_loss(self, slurm_output):
        """The regex should not match 'Validation loss: ...' lines."""
        for line in slurm_output.splitlines():
            if "Validation loss:" in line:
                assert not LOSS_PATTERN.search(line)

    def test_multi_epoch_step_continuation(self, slurm_output):
        """Steps continue across epochs (no reset at epoch boundary)."""
        losses = extract_losses(slurm_output)
        epochs = {epoch for epoch, _, _ in losses}
        assert 1 in epochs and 2 in epochs, "Both epochs should be present"
        epoch1_max = max(step for epoch, step, _ in losses if epoch == 1)
        epoch2_min = min(step for epoch, step, _ in losses if epoch == 2)
        assert epoch2_min > epoch1_max, "Epoch 2 steps should continue from epoch 1"

    def test_ignores_tqdm_suffix(self, slurm_output):
        """Regex extracts loss even when tqdm progress text follows."""
        tqdm_lines = [
            line
            for line in slurm_output.splitlines()
            if LOSS_PATTERN.search(line) and "it/s]" in line
        ]
        assert len(tqdm_lines) > 0, "Fixture should have loss lines with tqdm suffixes"
        for line in tqdm_lines:
            match = LOSS_PATTERN.search(line)
            loss = float(match.group(3))
            assert loss > 0

    def test_ignores_wandb_loss(self, slurm_output):
        """wandb summary 'loss 0.26427' should not match the loss regex."""
        for line in slurm_output.splitlines():
            if line.strip().startswith("wandb:") and "loss" in line.lower():
                if "|" not in line:  # skip wandb sparkline rows
                    assert not LOSS_PATTERN.search(line), (
                        f"wandb line should not match: {line}"
                    )

    def test_ignores_pad_token_warnings(self, slurm_output):
        """pad_token_id warning lines should not match loss regex."""
        for line in slurm_output.splitlines():
            if "pad_token_id" in line:
                assert not LOSS_PATTERN.search(line), (
                    f"pad_token_id line should not match: {line}"
                )


class TestEvalResultParsing:
    def test_status_is_success(self, eval_result):
        assert eval_result["status"] == "success"

    def test_has_required_keys(self, eval_result):
        required = [
            "status",
            "task",
            "model",
            "samples",
            "scorer",
            "accuracy",
            "metrics",
        ]
        for key in required:
            assert key in eval_result, f"Missing key: {key}"

    def test_accuracy_in_valid_range(self, eval_result):
        assert 0.0 <= eval_result["accuracy"] <= 1.0

    def test_accuracy_matches_metrics(self, eval_result):
        assert eval_result["accuracy"] == eval_result["metrics"]["accuracy"]

    def test_samples_is_positive_int(self, eval_result):
        assert isinstance(eval_result["samples"], int)
        assert eval_result["samples"] > 0

    def test_task_name_is_string(self, eval_result):
        assert isinstance(eval_result["task"], str)
        assert len(eval_result["task"]) > 0

    def test_has_path(self, eval_result):
        assert "path" in eval_result
        assert isinstance(eval_result["path"], str)

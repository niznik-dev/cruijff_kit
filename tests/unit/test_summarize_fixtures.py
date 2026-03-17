"""Tests for summarize-experiment data extraction against realistic fixtures.

Tests loss regex extraction from SLURM stdout and eval result JSON parsing.
Pure unit tests — no GPU, SLURM, or inspect-ai required.
"""

import json
import re
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "summarize"

# Loss regex from summarize-experiment skill (SKILL.md line 63)
LOSS_PATTERN = re.compile(r"(\d+)\|(\d+)\|Loss: ([0-9.]+)")


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
        matches = LOSS_PATTERN.findall(slurm_output)
        assert len(matches) > 0, "Should find at least one loss line"

    def test_final_loss_is_last_match(self, slurm_output):
        matches = LOSS_PATTERN.findall(slurm_output)
        epoch, step, loss = matches[-1]
        assert float(loss) == pytest.approx(0.26427456736564636)
        assert int(step) == 500

    def test_first_loss_is_higher(self, slurm_output):
        matches = LOSS_PATTERN.findall(slurm_output)
        first_loss = float(matches[0][2])
        last_loss = float(matches[-1][2])
        assert first_loss > last_loss, "Loss should decrease during training"

    def test_all_losses_are_positive(self, slurm_output):
        matches = LOSS_PATTERN.findall(slurm_output)
        for _, _, loss_str in matches:
            assert float(loss_str) > 0

    def test_epoch_numbers_are_valid(self, slurm_output):
        matches = LOSS_PATTERN.findall(slurm_output)
        for epoch_str, _, _ in matches:
            assert int(epoch_str) >= 1

    def test_step_numbers_increase(self, slurm_output):
        matches = LOSS_PATTERN.findall(slurm_output)
        steps = [int(m[1]) for m in matches]
        assert steps == sorted(steps), "Steps should monotonically increase"

    def test_total_loss_count(self, slurm_output):
        matches = LOSS_PATTERN.findall(slurm_output)
        assert len(matches) == 18, "Fixture has 18 loss line matches"

    def test_does_not_match_validation_loss(self, slurm_output):
        """The regex should not match 'Validation loss: ...' lines."""
        for line in slurm_output.splitlines():
            if "Validation loss:" in line:
                assert not LOSS_PATTERN.search(line)

    def test_multi_epoch_step_continuation(self, slurm_output):
        """Steps continue across epochs (no reset at epoch boundary)."""
        matches = LOSS_PATTERN.findall(slurm_output)
        epochs = {int(m[0]) for m in matches}
        assert 1 in epochs and 2 in epochs, "Both epochs should be present"
        # Epoch 2 steps should be > epoch 1's max step
        epoch1_max = max(int(m[1]) for m in matches if m[0] == "1")
        epoch2_min = min(int(m[1]) for m in matches if m[0] == "2")
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

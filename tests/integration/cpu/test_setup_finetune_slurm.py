"""Integration tests for setup_finetune.py SLURM generation.

Tests the model-aware SLURM resource allocation and MIG support.
"""

import subprocess
import pytest
from pathlib import Path

from cruijff_kit.tools.torchtune.model_configs import MODEL_CONFIGS


# Get repo root relative to this test file
REPO_ROOT = Path(__file__).parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "tools/torchtune/setup_finetune.py"
TEMPLATES_PATH = REPO_ROOT / "tools/torchtune/templates"


@pytest.fixture
def temp_run_dir(tmp_path):
    """Create a temp directory with required templates copied."""
    # Copy templates to the location the script expects (relative to script)
    # The script uses script_dir / "templates" so we need to run from a dir
    # where that path resolves correctly, OR we can symlink

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    yield run_dir


def run_setup_finetune(run_dir, model_name="Llama-3.2-1B-Instruct", extra_args=None):
    """Helper to run setup_finetune.py and return generated slurm content."""
    cmd = [
        "python", str(SCRIPT_PATH),
        "--torchtune_model_name", model_name,
        "--dataset_label", "test_dataset",
        "--dataset_ext", ".json",
        "--input_dir_base", "/fake/path/",
        "--output_dir_base", "/fake/output/",
        "--models_dir", "/fake/models/",
        "--my_wandb_run_name", "test_run",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(
        cmd,
        cwd=run_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"setup_finetune.py failed:\nstderr: {result.stderr}\nstdout: {result.stdout}")

    slurm_file = run_dir / "finetune.slurm"
    if not slurm_file.exists():
        pytest.fail(f"finetune.slurm not created. stdout: {result.stdout}")

    return slurm_file.read_text()


class TestModelAwareSlurmResources:
    """Test that SLURM resources are set correctly based on model size."""

    def test_1b_default_memory(self, temp_run_dir):
        """1B model defaults to MODEL_CONFIGS memory."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-1B-Instruct")
        expected_mem = MODEL_CONFIGS["Llama-3.2-1B-Instruct"]["slurm"]["mem"]
        assert f"#SBATCH --mem={expected_mem}" in slurm

    def test_1b_no_partition(self, temp_run_dir):
        """1B model without CLI partition leaves line commented."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-1B-Instruct")
        # Partition line should remain commented out
        assert "##SBATCH --partition=<PART>" in slurm

    def test_1b_no_constraint_by_default(self, temp_run_dir):
        """1B model without CLI constraint leaves line commented."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-1B-Instruct")
        assert "##SBATCH --constraint=<CONST>" in slurm

    def test_1b_constraint_via_cli(self, temp_run_dir):
        """1B model with --constraint sets constraint in SLURM script."""
        slurm = run_setup_finetune(
            temp_run_dir, "Llama-3.2-1B-Instruct",
            extra_args=["--constraint", "gpu80"]
        )
        assert "#SBATCH --constraint=gpu80" in slurm

    def test_3b_default_memory(self, temp_run_dir):
        """3B model defaults to MODEL_CONFIGS memory."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-3B-Instruct")
        expected_mem = MODEL_CONFIGS["Llama-3.2-3B-Instruct"]["slurm"]["mem"]
        assert f"#SBATCH --mem={expected_mem}" in slurm

    def test_3b_no_constraint_by_default(self, temp_run_dir):
        """3B model without CLI constraint leaves line commented."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-3B-Instruct")
        assert "##SBATCH --constraint=<CONST>" in slurm

    def test_3b_no_partition(self, temp_run_dir):
        """3B model without CLI partition leaves line commented."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-3B-Instruct")
        assert "##SBATCH --partition=<PART>" in slurm

    def test_8b_default_memory(self, temp_run_dir):
        """8B model defaults to MODEL_CONFIGS memory."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.1-8B-Instruct")
        expected_mem = MODEL_CONFIGS["Llama-3.1-8B-Instruct"]["slurm"]["mem"]
        assert f"#SBATCH --mem={expected_mem}" in slurm

    def test_8b_no_constraint_by_default(self, temp_run_dir):
        """8B model without CLI constraint leaves line commented."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.1-8B-Instruct")
        assert "##SBATCH --constraint=<CONST>" in slurm

    def test_8b_single_gpu(self, temp_run_dir):
        """8B model uses single GPU."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.1-8B-Instruct")
        expected_gpus = MODEL_CONFIGS["Llama-3.1-8B-Instruct"]["slurm"]["gpus"]
        assert f"#SBATCH --gres=gpu:{expected_gpus}" in slurm
        assert "lora_finetune_single_device" in slurm

    def test_70b_default_memory(self, temp_run_dir):
        """70B model defaults to MODEL_CONFIGS memory."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.3-70B-Instruct")
        expected_mem = MODEL_CONFIGS["Llama-3.3-70B-Instruct"]["slurm"]["mem"]
        assert f"#SBATCH --mem={expected_mem}" in slurm

    def test_70b_no_constraint_by_default(self, temp_run_dir):
        """70B model without CLI constraint leaves line commented."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.3-70B-Instruct")
        assert "##SBATCH --constraint=<CONST>" in slurm

    def test_70b_no_partition(self, temp_run_dir):
        """70B model without CLI partition leaves line commented."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.3-70B-Instruct")
        assert "##SBATCH --partition=<PART>" in slurm

    def test_70b_multi_gpu(self, temp_run_dir):
        """70B model defaults to MODEL_CONFIGS GPU count."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.3-70B-Instruct")
        expected_gpus = MODEL_CONFIGS["Llama-3.3-70B-Instruct"]["slurm"]["gpus"]
        assert f"#SBATCH --gres=gpu:{expected_gpus}" in slurm

    def test_70b_distributed_training(self, temp_run_dir):
        """70B model uses distributed training recipe."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.3-70B-Instruct")
        expected_gpus = MODEL_CONFIGS["Llama-3.3-70B-Instruct"]["slurm"]["gpus"]
        assert "lora_finetune_distributed" in slurm
        assert f"--nproc_per_node={expected_gpus}" in slurm


class TestMigSupport:
    """Test MIG (Multi-Instance GPU) support for 1B models."""

    def test_mig_partition_stays_commented(self, temp_run_dir):
        """When MIG enabled (partition=''), partition line stays commented."""
        slurm = run_setup_finetune(
            temp_run_dir,
            "Llama-3.2-1B-Instruct",
            extra_args=["--partition", "", "--mem", "16G"]
        )
        assert "##SBATCH --partition=<PART>" in slurm

    def test_mig_enabled_reduced_memory(self, temp_run_dir):
        """When MIG enabled, memory is set to user-specified value."""
        slurm = run_setup_finetune(
            temp_run_dir,
            "Llama-3.2-1B-Instruct",
            extra_args=["--partition", "", "--mem", "16G"]
        )
        assert "#SBATCH --mem=16G" in slurm


class TestUserOverrides:
    """Test that user-specified values override model defaults."""

    def test_mem_override(self, temp_run_dir):
        """Custom --mem overrides model default."""
        slurm = run_setup_finetune(
            temp_run_dir,
            "Llama-3.2-1B-Instruct",
            extra_args=["--mem", "64G"]
        )
        assert "#SBATCH --mem=64G" in slurm
        assert "#SBATCH --mem=40G" not in slurm

    def test_partition_override(self, temp_run_dir):
        """Custom --partition sets partition in SLURM script."""
        slurm = run_setup_finetune(
            temp_run_dir,
            "Llama-3.2-1B-Instruct",
            extra_args=["--partition", "gpu-debug"]
        )
        assert "#SBATCH --partition=gpu-debug" in slurm

    def test_constraint_override(self, temp_run_dir):
        """Custom --constraint sets constraint in SLURM script."""
        slurm = run_setup_finetune(
            temp_run_dir,
            "Llama-3.2-1B-Instruct",
            extra_args=["--constraint", "gpu40"]
        )
        assert "#SBATCH --constraint=gpu40" in slurm


class TestCpuAllocation:
    """Test CPU allocation based on model and GPU count."""

    def test_1b_default_cpus(self, temp_run_dir):
        """1B model defaults to MODEL_CONFIGS CPU count."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-1B-Instruct")
        expected_cpus = MODEL_CONFIGS["Llama-3.2-1B-Instruct"]["slurm"]["cpus"]
        assert f"#SBATCH --cpus-per-task={expected_cpus}" in slurm

    def test_3b_default_cpus(self, temp_run_dir):
        """3B model defaults to MODEL_CONFIGS CPU count."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-3B-Instruct")
        expected_cpus = MODEL_CONFIGS["Llama-3.2-3B-Instruct"]["slurm"]["cpus"]
        assert f"#SBATCH --cpus-per-task={expected_cpus}" in slurm

    def test_8b_default_cpus(self, temp_run_dir):
        """8B model defaults to MODEL_CONFIGS CPU count."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.1-8B-Instruct")
        expected_cpus = MODEL_CONFIGS["Llama-3.1-8B-Instruct"]["slurm"]["cpus"]
        assert f"#SBATCH --cpus-per-task={expected_cpus}" in slurm

    def test_70b_default_cpus(self, temp_run_dir):
        """70B model defaults to MODEL_CONFIGS CPU count."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.3-70B-Instruct")
        expected_cpus = MODEL_CONFIGS["Llama-3.3-70B-Instruct"]["slurm"]["cpus"]
        assert f"#SBATCH --cpus-per-task={expected_cpus}" in slurm

"""Integration tests for setup_finetune.py SLURM generation.

Tests the model-aware SLURM resource allocation and MIG support.
"""

import subprocess
import pytest
from pathlib import Path


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
        """1B model defaults to 40G memory."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-1B-Instruct")
        assert "#SBATCH --mem=40G" in slurm

    def test_1b_default_partition_nomig(self, temp_run_dir):
        """1B model defaults to nomig partition."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-1B-Instruct")
        assert "#SBATCH --partition=nomig" in slurm

    def test_1b_no_gpu_constraint(self, temp_run_dir):
        """1B model should not have gpu80 constraint."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-1B-Instruct")
        # Constraint line should remain commented out
        assert "#SBATCH --constraint=gpu80" not in slurm

    def test_3b_default_memory(self, temp_run_dir):
        """3B model defaults to 80G memory."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-3B-Instruct")
        assert "#SBATCH --mem=80G" in slurm

    def test_3b_gpu80_constraint(self, temp_run_dir):
        """3B model requires gpu80 constraint."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-3B-Instruct")
        assert "#SBATCH --constraint=gpu80" in slurm

    def test_3b_no_partition(self, temp_run_dir):
        """3B model should not set a partition (uses constraint instead)."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-3B-Instruct")
        # Partition line should remain commented out for 3B
        # (it uses constraint=gpu80 instead)
        assert "#SBATCH --partition=nomig" not in slurm

    def test_8b_default_memory(self, temp_run_dir):
        """8B model defaults to 80G memory."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.1-8B-Instruct")
        assert "#SBATCH --mem=80G" in slurm

    def test_8b_gpu80_constraint(self, temp_run_dir):
        """8B model requires gpu80 constraint."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.1-8B-Instruct")
        assert "#SBATCH --constraint=gpu80" in slurm

    def test_8b_single_gpu(self, temp_run_dir):
        """8B model uses single GPU."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.1-8B-Instruct")
        assert "#SBATCH --gres=gpu:1" in slurm
        assert "lora_finetune_single_device" in slurm

    def test_70b_default_memory(self, temp_run_dir):
        """70B model defaults to 320G memory (4 GPUs × 80GB)."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.3-70B-Instruct")
        assert "#SBATCH --mem=320G" in slurm

    def test_70b_gpu80_constraint(self, temp_run_dir):
        """70B model requires gpu80 constraint."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.3-70B-Instruct")
        assert "#SBATCH --constraint=gpu80" in slurm

    def test_70b_no_partition(self, temp_run_dir):
        """70B model should not set a partition (uses constraint instead)."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.3-70B-Instruct")
        assert "#SBATCH --partition=nomig" not in slurm

    def test_70b_multi_gpu(self, temp_run_dir):
        """70B model defaults to 4 GPUs."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.3-70B-Instruct")
        assert "#SBATCH --gres=gpu:4" in slurm

    def test_70b_distributed_training(self, temp_run_dir):
        """70B model uses distributed training recipe."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.3-70B-Instruct")
        assert "lora_finetune_distributed" in slurm
        assert "--nproc_per_node=4" in slurm


class TestMigSupport:
    """Test MIG (Multi-Instance GPU) support for 1B models."""

    def test_mig_enabled_no_partition_line(self, temp_run_dir):
        """When MIG enabled (partition=''), partition line stays commented."""
        slurm = run_setup_finetune(
            temp_run_dir,
            "Llama-3.2-1B-Instruct",
            extra_args=["--partition", "", "--mem", "16G"]
        )
        # Should NOT have an active partition line
        assert "#SBATCH --partition=nomig" not in slurm
        assert "#SBATCH --partition=" not in slurm or "##SBATCH --partition=" in slurm

    def test_mig_enabled_reduced_memory(self, temp_run_dir):
        """When MIG enabled, memory is set to user-specified value."""
        slurm = run_setup_finetune(
            temp_run_dir,
            "Llama-3.2-1B-Instruct",
            extra_args=["--partition", "", "--mem", "16G"]
        )
        assert "#SBATCH --mem=16G" in slurm

    def test_mig_partition_commented_out(self, temp_run_dir):
        """Verify the partition line remains in commented form for MIG."""
        slurm = run_setup_finetune(
            temp_run_dir,
            "Llama-3.2-1B-Instruct",
            extra_args=["--partition", "", "--mem", "16G"]
        )
        # The template has ##SBATCH --partition=<PART>
        # With empty partition, it should stay commented
        assert "##SBATCH --partition=<PART>" in slurm


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
        """Custom --partition overrides model default."""
        slurm = run_setup_finetune(
            temp_run_dir,
            "Llama-3.2-1B-Instruct",
            extra_args=["--partition", "gpu-debug"]
        )
        assert "#SBATCH --partition=gpu-debug" in slurm
        assert "#SBATCH --partition=nomig" not in slurm

    def test_constraint_override(self, temp_run_dir):
        """Custom --constraint overrides model default."""
        slurm = run_setup_finetune(
            temp_run_dir,
            "Llama-3.2-1B-Instruct",
            extra_args=["--constraint", "gpu40"]
        )
        assert "#SBATCH --constraint=gpu40" in slurm


class TestCpuAllocation:
    """Test CPU allocation based on model and GPU count."""

    def test_1b_default_cpus(self, temp_run_dir):
        """1B model defaults to 4 CPUs."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-1B-Instruct")
        assert "#SBATCH --cpus-per-task=4" in slurm

    def test_3b_default_cpus(self, temp_run_dir):
        """3B model defaults to 4 CPUs."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.2-3B-Instruct")
        assert "#SBATCH --cpus-per-task=4" in slurm

    def test_8b_default_cpus(self, temp_run_dir):
        """8B model defaults to 4 CPUs."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.1-8B-Instruct")
        assert "#SBATCH --cpus-per-task=4" in slurm

    def test_70b_default_cpus(self, temp_run_dir):
        """70B model defaults to 16 CPUs (4 per GPU × 4 GPUs)."""
        slurm = run_setup_finetune(temp_run_dir, "Llama-3.3-70B-Instruct")
        assert "#SBATCH --cpus-per-task=16" in slurm

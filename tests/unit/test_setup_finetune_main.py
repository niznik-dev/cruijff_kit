"""Tests for setup_finetune.py main() — config merging, YAML/SLURM generation.

These tests exercise the template rendering pipeline that the existing
test_setup_finetune.py doesn't cover (helper functions only).
"""

import sys
import yaml
from unittest.mock import patch

import pytest

from cruijff_kit.tools.torchtune.setup_finetune import main


@pytest.fixture
def setup_yaml(tmp_path):
    """Write a minimal setup_finetune.yaml config."""
    config = {
        "torchtune_model_name": "Llama-3.2-1B-Instruct",
        "output_dir_base": str(tmp_path / "outputs"),
        "input_dir_base": str(tmp_path / "inputs") + "/",
        "models_dir": str(tmp_path / "models"),
        "experiment_name": "test_exp",
        "dataset_label": "test_data",
        "dataset_ext": ".json",
        "my_wandb_project": "test_project",
        "batch_size": 2,
        "epochs": 1,
        "lora_rank": 8,
        "lr": 2e-4,
        "time": "01:00:00",
        "gpus": 1,
        "conda_env": "test_env",
    }
    config_path = tmp_path / "setup_finetune.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def run_main(tmp_path, setup_yaml, monkeypatch):
    """Run main() in tmp_path with the setup YAML and return output paths."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER", "testuser")

    def _run(extra_args=None):
        argv = ["setup_finetune.py", "--config_file", str(setup_yaml)]
        if extra_args:
            argv.extend(extra_args)
        with patch.object(sys, "argv", argv):
            main()
        finetune_yaml = tmp_path / "finetune.yaml"
        finetune_slurm = tmp_path / "finetune.slurm"
        with open(finetune_yaml) as f:
            config = yaml.safe_load(f)
        slurm = finetune_slurm.read_text()
        return config, slurm

    return _run


class TestMainYamlGeneration:
    """Tests for YAML config generation."""

    def test_generates_yaml_file(self, run_main, tmp_path):
        run_main()
        assert (tmp_path / "finetune.yaml").exists()

    def test_generates_slurm_file(self, run_main, tmp_path):
        run_main()
        assert (tmp_path / "finetune.slurm").exists()

    def test_model_component_set(self, run_main):
        config, _ = run_main()
        assert "lora_llama3_2_1b" in config["model"]["_component_"]

    def test_lora_rank_and_alpha(self, run_main):
        config, _ = run_main()
        assert config["model"]["lora_rank"] == 8
        assert config["model"]["lora_alpha"] == 16  # 2 * rank

    def test_learning_rate(self, run_main):
        config, _ = run_main()
        assert config["optimizer"]["lr"] == 2e-4

    def test_batch_size(self, run_main):
        config, _ = run_main()
        assert config["batch_size"] == 2

    def test_output_dir_constructed(self, run_main, tmp_path):
        config, _ = run_main()
        output_dir = config["output_dir"]
        assert "test_exp" in output_dir
        assert "ck-out-" in output_dir
        assert str(tmp_path) in output_dir

    def test_input_dir_constructed(self, run_main, tmp_path):
        config, _ = run_main()
        assert str(tmp_path) in config["input_dir"]

    def test_models_dir_substituted(self, run_main):
        config, _ = run_main()
        assert "$USER" not in config["models_dir"]

    def test_checkpoint_dir_set(self, run_main):
        config, _ = run_main()
        assert "Llama-3.2-1B-Instruct" in config["checkpointer"]["checkpoint_dir"]

    def test_tokenizer_path_set(self, run_main):
        config, _ = run_main()
        assert "tokenizer" in config["tokenizer"]["path"]

    def test_wandb_project(self, run_main):
        config, _ = run_main()
        assert config["my_wandb_project"] == "test_project"

    def test_dataset_chat_completion_default(self, run_main):
        config, _ = run_main()
        assert "chat_completion" in config["dataset"]["_component_"]

    def test_validation_removed_when_zero(self, run_main, tmp_path, setup_yaml):
        """run_val_every_n_steps=0 should remove validation config."""
        # Rewrite config with run_val_every_n_steps=0
        with open(setup_yaml) as f:
            cfg = yaml.safe_load(f)
        cfg["run_val_every_n_steps"] = 0
        with open(setup_yaml, "w") as f:
            yaml.dump(cfg, f)

        config, _ = run_main()
        assert "dataset_val" not in config
        assert "run_val_every_n_steps" not in config


class TestMainSlurmGeneration:
    """Tests for SLURM script generation."""

    def test_job_name_in_slurm(self, run_main):
        _, slurm = run_main()
        assert "#SBATCH --job-name=" in slurm
        # Should not contain the placeholder
        assert "<JOBNAME>" not in slurm

    def test_time_in_slurm(self, run_main):
        _, slurm = run_main()
        assert "01:00:00" in slurm

    def test_conda_env_in_slurm(self, run_main):
        _, slurm = run_main()
        assert "conda activate test_env" in slurm

    def test_netid_replaced(self, run_main):
        _, slurm = run_main()
        assert "<NETID>" not in slurm
        assert "testuser@princeton.edu" in slurm

    def test_output_dir_in_slurm(self, run_main):
        _, slurm = run_main()
        assert "<OUTPUT_DIR>" not in slurm
        assert "test_exp" in slurm

    def test_mem_replaced(self, run_main):
        _, slurm = run_main()
        assert "<MEM>" not in slurm

    def test_account_when_provided(self, run_main):
        _, slurm = run_main(extra_args=["--account", "myaccount"])
        assert "#SBATCH --account=myaccount" in slurm

    def test_partition_when_provided(self, run_main):
        _, slurm = run_main(extra_args=["--partition", "gpu"])
        assert "#SBATCH --partition=gpu" in slurm

    def test_constraint_when_provided(self, run_main):
        _, slurm = run_main(extra_args=["--constraint", "gpu80"])
        assert "#SBATCH --constraint=gpu80" in slurm

    def test_single_gpu_uses_single_device(self, run_main):
        _, slurm = run_main()
        assert "lora_finetune_single_device" in slurm

    def test_multi_gpu_uses_distributed(self, run_main):
        _, slurm = run_main(extra_args=["--gpus", "4"])
        assert "lora_finetune_distributed" in slurm
        assert "--gres=gpu:4" in slurm


class TestMainConfigPrecedence:
    """Tests for CLI > config_file > argparse_default precedence."""

    def test_cli_overrides_config_file(self, run_main):
        """CLI args should override config file values."""
        config, _ = run_main(extra_args=["--lora_rank", "16"])
        assert config["model"]["lora_rank"] == 16
        assert config["model"]["lora_alpha"] == 32

    def test_cli_overrides_batch_size(self, run_main):
        config, _ = run_main(extra_args=["--batch_size", "8"])
        assert config["batch_size"] == 8

    def test_custom_run_name(self, run_main):
        config, _ = run_main(extra_args=["--my_wandb_run_name", "my_run"])
        assert config["my_wandb_run_name"] == "my_run"


class TestMainDatasetTypes:
    """Tests for different dataset type configurations."""

    def test_text_completion_type(self, run_main, setup_yaml):
        with open(setup_yaml) as f:
            cfg = yaml.safe_load(f)
        cfg["dataset_type"] = "text_completion"
        with open(setup_yaml, "w") as f:
            yaml.dump(cfg, f)

        config, _ = run_main()
        assert "text_completion" in config["dataset"]["_component_"]
        assert "system_prompt" not in config["dataset"]

    def test_custom_prompt(self, run_main):
        config, _ = run_main(extra_args=["--prompt", "Answer: {input}"])
        assert config["dataset"]["prompt"] == "Answer: {input}"

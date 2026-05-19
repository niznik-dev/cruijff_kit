"""Tests for setup_finetune.py main() — config merging, YAML/SLURM generation.

These tests exercise the template rendering pipeline that the existing
test_setup_finetune.py doesn't cover (helper functions only).
"""

import sys
import warnings
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

    def test_batch_size_val_absent_by_default(self, run_main):
        # No flag → no batch_size_val key. Recipe falls back to cfg.batch_size.
        config, _ = run_main()
        assert "batch_size_val" not in config

    def test_batch_size_val_via_cli(self, run_main):
        config, _ = run_main(extra_args=["--batch_size_val", "128"])
        assert config["batch_size_val"] == 128

    def test_batch_size_val_via_yaml(self, run_main, setup_yaml):
        with open(setup_yaml) as f:
            data = yaml.safe_load(f)
        data["batch_size_val"] = 64
        with open(setup_yaml, "w") as f:
            yaml.dump(data, f)
        config, _ = run_main()
        assert config["batch_size_val"] == 64

    def test_seed_default(self, run_main):
        """When seed is not specified, the default (14 — Cruijff's number) is used."""
        config, _ = run_main()
        assert config["seed"] == 14

    def test_seed_via_cli(self, run_main):
        """--seed on the command line flows through to finetune.yaml."""
        config, _ = run_main(extra_args=["--seed", "7"])
        assert config["seed"] == 7

    def test_seed_via_config_file(self, run_main, setup_yaml):
        """seed in setup_finetune.yaml flows through to finetune.yaml."""
        with open(setup_yaml) as f:
            data = yaml.safe_load(f)
        data["seed"] = 23
        with open(setup_yaml, "w") as f:
            yaml.dump(data, f)
        config, _ = run_main()
        assert config["seed"] == 23

    def test_output_dir_constructed(self, run_main, tmp_path):
        config, _ = run_main()
        output_dir = config["output_dir"]
        assert "test_exp" in output_dir
        assert "/artifacts/" in output_dir
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

    def test_validation_preserved_when_nonzero(self, run_main, tmp_path, setup_yaml):
        """run_val_every_n_steps>0 should preserve validation config end-to-end.

        Guards against regression of the scaffold-torchtune agent silently emitting
        run_val_every_n_steps=0 despite validation_during_training=true in the
        experiment design.
        """
        with open(setup_yaml) as f:
            cfg = yaml.safe_load(f)
        cfg["run_val_every_n_steps"] = 50
        with open(setup_yaml, "w") as f:
            yaml.dump(cfg, f)

        config, _ = run_main()
        assert config["run_val_every_n_steps"] == 50
        assert "dataset_val" in config

    def test_unknown_key_warns(self, run_main, setup_yaml):
        """A setup_finetune.yaml key with no matching argparse arg should warn."""
        with open(setup_yaml) as f:
            cfg = yaml.safe_load(f)
        cfg["lora_ranl"] = 42  # typo of lora_rank
        cfg["evaluation_temperature"] = 0.5  # wrong namespace
        with open(setup_yaml, "w") as f:
            yaml.dump(cfg, f)

        with pytest.warns(UserWarning, match="setup_finetune.yaml has keys"):
            run_main()

    def test_known_keys_do_not_warn(self, run_main, setup_yaml):
        """Default fixture has only argparse-known keys; no unknown-key warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_main()
        unknown_warnings = [
            w for w in caught if "setup_finetune.yaml has keys" in str(w.message)
        ]
        assert not unknown_warnings, [str(w.message) for w in unknown_warnings]


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


class TestCpusPerTask:
    """Tests for the --cpus_per_task override (issue #449).

    Resolution order: CLI > setup_finetune.yaml > MODEL_CONFIGS["slurm"]["cpus"] > 1.
    The fixture's model is Llama-3.2-1B-Instruct, which has cpus=1 in MODEL_CONFIGS.
    """

    def test_cli_override_writes_to_slurm(self, run_main):
        _, slurm = run_main(extra_args=["--cpus_per_task", "8"])
        assert "#SBATCH --cpus-per-task=8" in slurm

    def test_default_falls_back_to_model_config(self, run_main):
        _, slurm = run_main()
        assert "#SBATCH --cpus-per-task=1" in slurm

    def test_cli_overrides_yaml(self, run_main, setup_yaml):
        with open(setup_yaml) as f:
            data = yaml.safe_load(f)
        data["cpus_per_task"] = 4
        with open(setup_yaml, "w") as f:
            yaml.dump(data, f)
        _, slurm = run_main(extra_args=["--cpus_per_task", "12"])
        assert "#SBATCH --cpus-per-task=12" in slurm

    def test_cpus_per_task_not_in_finetune_yaml(self, run_main):
        # SLURM-only knob; must not leak into the rendered torchtune config.
        config, _ = run_main(extra_args=["--cpus_per_task", "8"])
        assert "cpus_per_task" not in config


class TestCustomRecipeGuard:
    """Tests for the auto-switch guard added in #471.

    Pre-#471, asking for a single-device recipe with gpus>1 silently rewrote it
    to the distributed equivalent — including for `_nightly`, where no
    distributed variant exists. Failures showed up at SLURM runtime instead of
    scaffold time. The guard now hard-errors when the rewritten recipe doesn't
    resolve to a real module on disk.
    """

    NIGHTLY = (
        "cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_nightly"
    )
    STABLE_SINGLE = (
        "cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_stable"
    )
    STABLE_DIST = (
        "cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_distributed_stable"
    )

    def test_single_device_nightly_with_one_gpu_resolves(self, run_main):
        _, slurm = run_main(extra_args=["--custom_recipe", self.NIGHTLY])
        assert "lora_finetune_single_device_nightly" in slurm

    def test_single_device_nightly_with_multi_gpu_raises(self, run_main):
        with pytest.raises(FileNotFoundError, match="474"):
            run_main(extra_args=["--custom_recipe", self.NIGHTLY, "--gpus", "4"])

    def test_single_device_stable_with_multi_gpu_auto_switches(self, run_main):
        _, slurm = run_main(
            extra_args=["--custom_recipe", self.STABLE_SINGLE, "--gpus", "4"]
        )
        assert "lora_finetune_distributed_stable" in slurm

    def test_distributed_stable_with_multi_gpu_resolves(self, run_main):
        _, slurm = run_main(
            extra_args=["--custom_recipe", self.STABLE_DIST, "--gpus", "4"]
        )
        assert "lora_finetune_distributed_stable" in slurm


class TestCustomRecipeDefault:
    """Tests for the GPU-aware --custom_recipe default added in #471.

    Anchors the default recipe choice in code rather than the scaffold-torchtune
    agent doc, so omitting --custom_recipe still produces a working recipe path.
    """

    def test_single_gpu_default_is_nightly(self, run_main):
        """No --custom_recipe + 1 GPU → single_device_nightly (val-capable)."""
        _, slurm = run_main()
        assert "lora_finetune_single_device_nightly" in slurm

    def test_multi_gpu_default_is_distributed_stable(self, run_main):
        """No --custom_recipe + multi-GPU → distributed_stable (no val yet, see #474)."""
        _, slurm = run_main(extra_args=["--gpus", "4"])
        assert "lora_finetune_distributed_stable" in slurm

    def test_explicit_custom_recipe_overrides_default(self, run_main):
        """An explicit --custom_recipe still wins over the default."""
        _, slurm = run_main(
            extra_args=[
                "--custom_recipe",
                "cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_stable",
            ]
        )
        assert "lora_finetune_single_device_stable" in slurm
        assert "lora_finetune_single_device_nightly" not in slurm


class TestMultiGpuValWarning:
    """Tests for the Python-side warning when val is requested for multi-GPU.

    The scaffold-torchtune agent is supposed to flag this combination, but
    setup_finetune.py duplicates the warning so it fires regardless of how
    the script was invoked. Tracked in #474.
    """

    def _set_val(self, setup_yaml, n_steps):
        with open(setup_yaml) as f:
            cfg = yaml.safe_load(f)
        cfg["run_val_every_n_steps"] = n_steps
        with open(setup_yaml, "w") as f:
            yaml.dump(cfg, f)

    def test_warning_emitted_for_val_plus_multi_gpu(self, run_main, setup_yaml):
        self._set_val(setup_yaml, 50)
        with pytest.warns(UserWarning, match="474"):
            run_main(extra_args=["--gpus", "4"])

    def test_no_warning_for_val_plus_single_gpu(self, run_main, setup_yaml):
        self._set_val(setup_yaml, 50)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn any warning into a test failure
            run_main()

    def test_no_warning_for_no_val_plus_multi_gpu(self, run_main, setup_yaml):
        self._set_val(setup_yaml, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run_main(extra_args=["--gpus", "4"])


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

    def test_epochs_to_save_string_from_config_parsed_to_list(
        self, run_main, setup_yaml
    ):
        """A comma-separated string in the config file must go through
        parse_epochs and come out as a list of ints (regression for #431)."""
        with open(setup_yaml) as f:
            data = yaml.safe_load(f)
        data["epochs_to_save"] = "3,5"
        with open(setup_yaml, "w") as f:
            yaml.dump(data, f)
        config, _ = run_main()
        assert config["epochs_to_save"] == [3, 5]

    def test_quoted_bool_string_from_config_parsed_to_bool(self, run_main, setup_yaml):
        """A quoted string like "true" in the config file must go through
        parse_bool and land as a Python bool, not a truthy string
        (regression for #431)."""
        with open(setup_yaml) as f:
            data = yaml.safe_load(f)
        data["packed"] = "true"
        with open(setup_yaml, "w") as f:
            yaml.dump(data, f)
        config, _ = run_main()
        assert config["dataset"]["packed"] is True


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

    def test_text_completion_omits_new_system_prompt(self, run_main, setup_yaml):
        """Test that text_completion must not emit
        `new_system_prompt`, which `text_completion_dataset()` rejects."""
        with open(setup_yaml) as f:
            cfg = yaml.safe_load(f)
        cfg["dataset_type"] = "text_completion"
        cfg["system_prompt"] = "You are a helpful assistant."
        with open(setup_yaml, "w") as f:
            yaml.dump(cfg, f)

        config, _ = run_main()
        assert "new_system_prompt" not in config["dataset"]
        if "dataset_val" in config:
            assert "new_system_prompt" not in config["dataset_val"]

    def test_custom_prompt(self, run_main):
        config, _ = run_main(extra_args=["--prompt", "Answer: {input}"])
        assert config["dataset"]["prompt"] == "Answer: {input}"


class TestTrainingSamples:
    """Regression: --training_samples must slice train data, not val (#447)."""

    def test_training_samples_writes_max_samples_to_dataset(self, run_main):
        config, _ = run_main(extra_args=["--training_samples", "50"])
        assert config["dataset"]["max_samples"] == 50

    def test_training_samples_does_not_affect_dataset_val(self, run_main):
        config, _ = run_main(extra_args=["--training_samples", "50"])
        assert "max_samples" not in config.get("dataset_val", {}), (
            "training_samples must NOT slice the validation set — "
            "val must stay full across consistency-curve runs"
        )

    def test_training_samples_via_config_file(self, run_main, setup_yaml):
        """Mirrors the consistency-curve workflow: knob set in setup_finetune.yaml, not CLI."""
        with open(setup_yaml) as f:
            config_data = yaml.safe_load(f)
        config_data["training_samples"] = 50
        with open(setup_yaml, "w") as f:
            yaml.dump(config_data, f)
        config, _ = run_main()
        assert config["dataset"]["max_samples"] == 50
        assert "max_samples" not in config.get("dataset_val", {})

    def test_no_training_samples_means_no_max_samples(self, run_main):
        config, _ = run_main()
        assert "max_samples" not in config["dataset"]

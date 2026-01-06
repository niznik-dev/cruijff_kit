"""Unit tests for torchtune config generation modules.

Tests for:
- tools/torchtune/config_utils.py
- tools/torchtune/dataset_config.py
- tools/torchtune/merge_recipe_params.py
"""

import argparse
import pytest
import sys
import yaml
from copy import deepcopy
from pathlib import Path

# Add tools/torchtune to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools" / "torchtune"))

from config_utils import (
    parse_epochs,
    parse_bool,
    calculate_lora_alpha,
    construct_output_dir,
    strip_slurm_params,
    expand_user_in_paths,
    validate_lr_scheduler,
    validate_dataset_type,
)
from dataset_config import (
    build_dataset_config,
    build_dataset_pair,
)
from merge_recipe_params import (
    apply_recipe_overrides,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_recipe_path():
    """Path to sample.yaml torchtune recipe."""
    return Path(__file__).parent.parent / "data/sample_torchtune_recipe.yaml"


@pytest.fixture
def sample_recipe(sample_recipe_path):
    """Load sample.yaml as a dictionary."""
    with open(sample_recipe_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def minimal_recipe():
    """Minimal recipe config for testing overrides."""
    return {
        "output_dir": "/tmp/output",
        "model": {
            "_component_": "torchtune.models.llama3_2.lora_llama3_2_1b",
            "lora_rank": 64,
            "lora_alpha": 128,
        },
        "tokenizer": {
            "_component_": "torchtune.models.llama3.llama3_tokenizer",
            "path": "${models_dir}/Llama-3.2-1B-Instruct/tokenizer.model",
        },
        "checkpointer": {
            "_component_": "torchtune.training.FullModelHFCheckpointer",
            "checkpoint_dir": "${models_dir}/Llama-3.2-1B-Instruct/",
            "output_dir": "${output_dir}",
        },
        "dataset": {
            "_component_": "torchtune.datasets.alpaca_cleaned_dataset",
            "packed": False,
        },
        "optimizer": {
            "_component_": "torch.optim.AdamW",
            "lr": 3e-4,
        },
        "lr_scheduler": {
            "_component_": "torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup",
            "num_warmup_steps": 100,
        },
        "metric_logger": {
            "_component_": "torchtune.training.metric_logging.DiskLogger",
            "log_dir": "${output_dir}/logs",
        },
        "epochs": 1,
        "batch_size": 4,
    }


# =============================================================================
# Tests for config_utils.py
# =============================================================================

class TestParseEpochs:
    """Tests for parse_epochs function."""

    def test_all(self):
        assert parse_epochs("all") == "all"
        assert parse_epochs("ALL") == "all"
        assert parse_epochs("All") == "all"

    def test_none(self):
        assert parse_epochs("none") == []
        assert parse_epochs("NONE") == []
        assert parse_epochs("None") == []

    def test_comma_delimited(self):
        assert parse_epochs("0,1,2") == [0, 1, 2]
        assert parse_epochs("0, 1, 2") == [0, 1, 2]
        assert parse_epochs("5") == [5]

    def test_already_list(self):
        assert parse_epochs([0, 1, 2]) == [0, 1, 2]

    def test_invalid(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_epochs("invalid")
        with pytest.raises(argparse.ArgumentTypeError):
            parse_epochs("1,2,abc")


class TestParseBool:
    """Tests for parse_bool function."""

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "Yes"])
    def test_true_values(self, value):
        assert parse_bool(value) is True

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "No"])
    def test_false_values(self, value):
        assert parse_bool(value) is False

    def test_already_bool(self):
        assert parse_bool(True) is True
        assert parse_bool(False) is False

    def test_invalid(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_bool("invalid")
        with pytest.raises(argparse.ArgumentTypeError):
            parse_bool("2")


class TestCalculateLoraAlpha:
    """Tests for calculate_lora_alpha function."""

    @pytest.mark.parametrize("rank,expected_alpha", [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
    ])
    def test_alpha_is_double_rank(self, rank, expected_alpha):
        assert calculate_lora_alpha(rank) == expected_alpha


class TestConstructOutputDir:
    """Tests for construct_output_dir function."""

    def test_with_experiment_name(self):
        result = construct_output_dir(
            "/scratch/output/",
            "my_experiment",
            "run_123"
        )
        assert result == "/scratch/output/my_experiment/ck-out-run_123/"

    def test_without_experiment_name(self):
        result = construct_output_dir(
            "/scratch/output/",
            "",
            "run_123"
        )
        assert result == "/scratch/output/ck-out-run_123/"

    def test_adds_trailing_slash(self):
        result = construct_output_dir(
            "/scratch/output",  # no trailing slash
            "exp",
            "run"
        )
        assert result == "/scratch/output/exp/ck-out-run/"


class TestStripSlurmParams:
    """Tests for strip_slurm_params function."""

    def test_removes_slurm_params(self):
        config = {
            "epochs": 1,
            "batch_size": 4,
            "time": "00:30:00",
            "gpus": 1,
            "conda_env": "myenv",
            "account": "myaccount",
        }
        result = strip_slurm_params(config)

        assert "epochs" in result
        assert "batch_size" in result
        assert "time" not in result
        assert "gpus" not in result
        assert "conda_env" not in result
        assert "account" not in result

    def test_preserves_non_slurm_params(self):
        config = {"epochs": 1, "lr": 1e-4, "model": {"lora_rank": 8}}
        result = strip_slurm_params(config)
        assert result == config


class TestExpandUserInPaths:
    """Tests for expand_user_in_paths function."""

    def test_expands_user_variable(self):
        config = {
            "input_dir": "/scratch/$USER/input",
            "output_dir": "/scratch/$USER/output",
        }
        result = expand_user_in_paths(config, username="testuser")

        assert result["input_dir"] == "/scratch/testuser/input"
        assert result["output_dir"] == "/scratch/testuser/output"

    def test_ignores_non_path_keys(self):
        config = {
            "input_dir": "/scratch/$USER/input",
            "epochs": 1,
            "model": {"lora_rank": 8},
        }
        result = expand_user_in_paths(config, username="testuser")

        assert result["input_dir"] == "/scratch/testuser/input"
        assert result["epochs"] == 1
        assert result["model"] == {"lora_rank": 8}


class TestValidateLrScheduler:
    """Tests for validate_lr_scheduler function."""

    @pytest.mark.parametrize("scheduler", [
        "get_cosine_schedule_with_warmup",
        "get_linear_schedule_with_warmup",
        "get_constant_schedule_with_warmup",
        "get_exponential_schedule_with_warmup",
    ])
    def test_valid_schedulers(self, scheduler):
        validate_lr_scheduler(scheduler)  # Should not raise

    def test_invalid_scheduler(self):
        with pytest.raises(ValueError, match="Invalid lr_scheduler"):
            validate_lr_scheduler("invalid_scheduler")


class TestValidateDatasetType:
    """Tests for validate_dataset_type function."""

    @pytest.mark.parametrize("dtype", [
        "instruct_dataset",
        "chat_dataset",
        "text_completion_dataset",
    ])
    def test_valid_types(self, dtype):
        validate_dataset_type(dtype)  # Should not raise

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid dataset_type"):
            validate_dataset_type("invalid_type")


# =============================================================================
# Tests for dataset_config.py
# =============================================================================

class TestBuildDatasetConfig:
    """Tests for build_dataset_config function."""

    def test_json_instruct_training(self):
        result = build_dataset_config(
            data_path="/data/words_5L",
            data_format="json",
            dataset_type="instruct_dataset",
            split="train",
        )

        assert result["_component_"] == "torchtune.datasets.instruct_dataset"
        assert result["source"] == "json"
        assert result["data_files"] == "/data/words_5L.json"
        assert result["field"] == "train"
        assert result["packed"] is True
        assert result["train_on_input"] is False

    def test_json_instruct_validation(self):
        result = build_dataset_config(
            data_path="/data/words_5L",
            data_format="json",
            dataset_type="instruct_dataset",
            split="validation",
        )

        assert result["field"] == "validation"
        assert "packed" not in result  # Not packed for validation

    def test_json_chat_training(self):
        result = build_dataset_config(
            data_path="/data/chat_folder",
            data_format="json",
            dataset_type="chat_dataset",
            split="train",
        )

        assert result["_component_"] == "torchtune.datasets.chat_dataset"
        assert result["source"] == "json"
        assert result["data_files"] == "/data/chat_folder/train.json"
        assert result["conversation_column"] == "messages"
        assert result["conversation_style"] == "openai"
        assert "field" not in result

    def test_parquet_training(self):
        result = build_dataset_config(
            data_path="/data/my_dataset",
            data_format="parquet",
            dataset_type="instruct_dataset",
            split="train",
        )

        assert result["source"] == "parquet"
        assert result["data_dir"] == "/data/my_dataset/train.parquet"
        assert result["split"] == "train"

    def test_parquet_validation(self):
        result = build_dataset_config(
            data_path="/data/my_dataset",
            data_format="parquet",
            dataset_type="instruct_dataset",
            split="validation",
        )

        assert result["data_dir"] == "/data/my_dataset/validation.parquet"
        assert result["split"] == "validation"

    def test_with_system_prompt(self):
        result = build_dataset_config(
            data_path="/data/words",
            data_format="json",
            dataset_type="instruct_dataset",
            split="train",
            system_prompt="You are a helpful assistant.",
        )

        assert result["new_system_prompt"] == "You are a helpful assistant."

    def test_with_train_on_input(self):
        result = build_dataset_config(
            data_path="/data/words",
            data_format="json",
            dataset_type="instruct_dataset",
            split="train",
            train_on_input=True,
        )

        assert result["train_on_input"] is True

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Unsupported data_format"):
            build_dataset_config(
                data_path="/data/words",
                data_format="csv",  # Invalid
                dataset_type="instruct_dataset",
                split="train",
            )


class TestBuildDatasetPair:
    """Tests for build_dataset_pair function."""

    def test_with_validation(self):
        result = build_dataset_pair(
            data_path="/data/words",
            data_format="json",
            dataset_type="instruct_dataset",
            include_validation=True,
        )

        assert "dataset" in result
        assert "dataset_val" in result
        assert result["dataset"]["field"] == "train"
        assert result["dataset_val"]["field"] == "validation"

    def test_without_validation(self):
        result = build_dataset_pair(
            data_path="/data/words",
            data_format="json",
            dataset_type="instruct_dataset",
            include_validation=False,
        )

        assert "dataset" in result
        assert "dataset_val" not in result

    def test_training_is_packed_validation_is_not(self):
        result = build_dataset_pair(
            data_path="/data/words",
            data_format="json",
            dataset_type="instruct_dataset",
            include_validation=True,
            packed=True,
        )

        assert result["dataset"]["packed"] is True
        assert "packed" not in result["dataset_val"]


# =============================================================================
# Tests for merge_recipe_params.py with sample.yaml
# =============================================================================

class TestApplyRecipeOverrides:
    """Tests for apply_recipe_overrides function."""

    def test_replaces_dataset_section(self, minimal_recipe):
        """Dataset section should be completely replaced."""
        config = deepcopy(minimal_recipe)
        additional_params = {
            "data_path": "/data/my_dataset",
            "data_format": "json",
            "dataset_type": "instruct_dataset",
            "validation_during_training": False,
        }

        result = apply_recipe_overrides(config, additional_params)

        # Original dataset was alpaca_cleaned_dataset, should be replaced
        assert result["dataset"]["_component_"] == "torchtune.datasets.instruct_dataset"
        assert result["dataset"]["source"] == "json"
        assert result["dataset"]["data_files"] == "/data/my_dataset.json"

    def test_adds_validation_dataset_when_requested(self, minimal_recipe):
        """Should add dataset_val when validation_during_training is True."""
        config = deepcopy(minimal_recipe)
        additional_params = {
            "data_path": "/data/my_dataset",
            "data_format": "json",
            "dataset_type": "instruct_dataset",
            "validation_during_training": True,
        }

        result = apply_recipe_overrides(config, additional_params)

        assert "dataset_val" in result
        assert result["dataset_val"]["field"] == "validation"
        assert "run_val_every_n_steps" in result

    def test_removes_validation_dataset_when_not_requested(self, minimal_recipe):
        """Should remove dataset_val when validation_during_training is False."""
        config = deepcopy(minimal_recipe)
        config["dataset_val"] = {"_component_": "some_val_dataset"}
        config["run_val_every_n_steps"] = 50

        additional_params = {
            "data_path": "/data/my_dataset",
            "data_format": "json",
            "validation_during_training": False,
        }

        result = apply_recipe_overrides(config, additional_params)

        assert "dataset_val" not in result
        assert "run_val_every_n_steps" not in result

    def test_recalculates_lora_alpha(self, minimal_recipe):
        """LoRA alpha should be recalculated as 2 * rank."""
        config = deepcopy(minimal_recipe)
        # Change lora_rank in the config (simulating merge_parameters)
        config["model"]["lora_rank"] = 8

        result = apply_recipe_overrides(config, {})

        assert result["model"]["lora_alpha"] == 16  # 2 * 8

    def test_constructs_output_dir(self, minimal_recipe):
        """Should construct proper output directory path."""
        config = deepcopy(minimal_recipe)
        additional_params = {
            "output_dir_base": "/scratch/outputs/",
            "experiment_name": "test_exp",
            "my_wandb_run_name": "run_001",
        }

        result = apply_recipe_overrides(config, additional_params)

        assert result["output_dir"] == "/scratch/outputs/test_exp/ck-out-run_001/"

    def test_updates_wandb_config(self, minimal_recipe):
        """Should update WandB configuration."""
        config = deepcopy(minimal_recipe)
        additional_params = {
            "my_wandb_run_name": "my_run",
            "my_wandb_project": "my_project",
        }

        result = apply_recipe_overrides(config, additional_params)

        assert result["my_wandb_run_name"] == "my_run"
        assert result["my_wandb_project"] == "my_project"

    def test_system_prompt_added_to_dataset(self, minimal_recipe):
        """System prompt should be added to dataset config."""
        config = deepcopy(minimal_recipe)
        additional_params = {
            "data_path": "/data/words",
            "data_format": "json",
            "system_prompt": "You are a helpful assistant.",
        }

        result = apply_recipe_overrides(config, additional_params)

        assert result["dataset"]["new_system_prompt"] == "You are a helpful assistant."


class TestMergeWithSampleRecipe:
    """Integration tests using the actual sample.yaml recipe."""

    def test_sample_recipe_loads(self, sample_recipe):
        """Verify sample.yaml can be loaded."""
        assert "model" in sample_recipe
        assert "dataset" in sample_recipe
        assert sample_recipe["model"]["lora_rank"] == 64
        assert sample_recipe["model"]["lora_alpha"] == 128

    def test_override_lora_rank(self, sample_recipe):
        """Override lora_rank and verify alpha is recalculated."""
        config = deepcopy(sample_recipe)
        config["model"]["lora_rank"] = 8  # Override

        result = apply_recipe_overrides(config, {})

        assert result["model"]["lora_rank"] == 8
        assert result["model"]["lora_alpha"] == 16  # 2 * 8

    def test_replace_hf_dataset_with_local(self, sample_recipe):
        """Replace HuggingFace dataset with local JSON file."""
        config = deepcopy(sample_recipe)
        additional_params = {
            "data_path": "/data/green/capitalization/words_5L_80P_1000",
            "data_format": "json",
            "dataset_type": "instruct_dataset",
            "validation_during_training": True,
        }

        result = apply_recipe_overrides(config, additional_params)

        # Should replace alpaca_cleaned_dataset
        assert "alpaca" not in result["dataset"]["_component_"]
        assert result["dataset"]["_component_"] == "torchtune.datasets.instruct_dataset"
        assert result["dataset"]["source"] == "json"
        assert result["dataset"]["data_files"].endswith(".json")

        # Should add validation
        assert "dataset_val" in result

    def test_full_override_scenario(self, sample_recipe):
        """Test a realistic full override scenario."""
        config = deepcopy(sample_recipe)

        # Simulate what merge_parameters would do
        config["model"]["lora_rank"] = 4
        config["batch_size"] = 8
        config["epochs"] = 2

        additional_params = {
            "data_path": "/scratch/data/words_5L",
            "data_format": "json",
            "dataset_type": "instruct_dataset",
            "validation_during_training": True,
            "system_prompt": "You are a helpful assistant.",
            "output_dir_base": "/scratch/outputs/",
            "experiment_name": "lora_comparison",
            "my_wandb_run_name": "rank4_run",
            "my_wandb_project": "cap_experiments",
            "models_dir": "/scratch/models",
        }

        result = apply_recipe_overrides(config, additional_params)

        # Check lora_alpha was recalculated
        assert result["model"]["lora_rank"] == 4
        assert result["model"]["lora_alpha"] == 8

        # Check dataset was replaced
        assert result["dataset"]["source"] == "json"
        assert result["dataset"]["new_system_prompt"] == "You are a helpful assistant."

        # Check validation dataset was added
        assert "dataset_val" in result
        assert result["dataset_val"]["new_system_prompt"] == "You are a helpful assistant."

        # Check output directory
        assert result["output_dir"] == "/scratch/outputs/lora_comparison/ck-out-rank4_run/"

        # Check wandb config
        assert result["my_wandb_run_name"] == "rank4_run"
        assert result["my_wandb_project"] == "cap_experiments"


class TestParquetDatasetOverride:
    """Tests specifically for parquet dataset configuration."""

    def test_parquet_dataset_replacement(self, sample_recipe):
        """Test replacing HF dataset with parquet format."""
        config = deepcopy(sample_recipe)
        additional_params = {
            "data_path": "/data/my_parquet_dataset",
            "data_format": "parquet",
            "dataset_type": "instruct_dataset",
            "validation_during_training": True,
        }

        result = apply_recipe_overrides(config, additional_params)

        assert result["dataset"]["source"] == "parquet"
        assert result["dataset"]["data_dir"] == "/data/my_parquet_dataset/train.parquet"
        assert result["dataset"]["split"] == "train"

        assert result["dataset_val"]["source"] == "parquet"
        assert result["dataset_val"]["data_dir"] == "/data/my_parquet_dataset/validation.parquet"


class TestChatDatasetOverride:
    """Tests specifically for chat dataset configuration."""

    def test_chat_dataset_replacement(self, sample_recipe):
        """Test replacing HF dataset with chat format."""
        config = deepcopy(sample_recipe)
        additional_params = {
            "data_path": "/data/chat_folder",
            "data_format": "json",
            "dataset_type": "chat_dataset",
            "validation_during_training": False,
        }

        result = apply_recipe_overrides(config, additional_params)

        assert result["dataset"]["_component_"] == "torchtune.datasets.chat_dataset"
        assert result["dataset"]["conversation_column"] == "messages"
        assert result["dataset"]["conversation_style"] == "openai"
        assert result["dataset"]["data_files"] == "/data/chat_folder/train.json"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_additional_params(self, minimal_recipe):
        """Should handle empty additional_params gracefully."""
        config = deepcopy(minimal_recipe)
        result = apply_recipe_overrides(config, {})

        # Should still recalculate lora_alpha
        assert result["model"]["lora_alpha"] == 128  # 2 * 64

    def test_none_additional_params(self, minimal_recipe):
        """Should handle None additional_params."""
        config = deepcopy(minimal_recipe)
        result = apply_recipe_overrides(config, None)

        assert "model" in result

    def test_partial_additional_params(self, minimal_recipe):
        """Should handle partial additional_params."""
        config = deepcopy(minimal_recipe)
        additional_params = {
            "my_wandb_run_name": "test_run",
            # No data_path, so dataset shouldn't be replaced
        }

        result = apply_recipe_overrides(config, additional_params)

        # Dataset should remain unchanged
        assert result["dataset"]["_component_"] == "torchtune.datasets.alpaca_cleaned_dataset"
        # But wandb name should be set
        assert result["my_wandb_run_name"] == "test_run"


# =============================================================================
# Tests for recipe_config_loader.py
# =============================================================================

from recipe_config_loader import (
    merge_parameters,
    load_recipe_defaults,
    format_merge_tracking,
)


class TestMergeParameters:
    """Tests for merge_parameters function - the core merging logic."""

    def test_recipe_defaults_preserved_when_no_overrides(self):
        """Recipe defaults should be preserved when no overrides provided."""
        recipe_defaults = {
            "epochs": 1,
            "batch_size": 4,
            "model": {
                "lora_rank": 64,
                "lora_alpha": 128,
            }
        }

        merged, tracking = merge_parameters(recipe_defaults, {}, {})

        assert merged["epochs"] == 1
        assert merged["batch_size"] == 4
        assert merged["model"]["lora_rank"] == 64
        assert merged["model"]["lora_alpha"] == 128

    def test_controls_override_recipe(self):
        """Control parameters should override recipe defaults."""
        recipe_defaults = {
            "epochs": 1,
            "batch_size": 4,
        }
        controls = {
            "epochs": 2,
        }

        merged, tracking = merge_parameters(recipe_defaults, controls, {})

        assert merged["epochs"] == 2  # Overridden by control
        assert merged["batch_size"] == 4  # Unchanged
        assert tracking["epochs"]["source"] == "control"
        assert tracking["batch_size"]["source"] == "recipe"

    def test_run_params_override_controls(self):
        """Run parameters should override control parameters."""
        recipe_defaults = {
            "epochs": 1,
            "batch_size": 4,
        }
        controls = {
            "epochs": 2,
            "batch_size": 8,
        }
        run_params = {
            "epochs": 3,
        }

        merged, tracking = merge_parameters(recipe_defaults, controls, run_params)

        assert merged["epochs"] == 3  # Overridden by run
        assert merged["batch_size"] == 8  # From control
        assert tracking["epochs"]["source"] == "run"
        assert tracking["batch_size"]["source"] == "control"

    def test_flat_key_overrides_nested_value(self):
        """Flat key in controls should override nested value with same key name."""
        recipe_defaults = {
            "model": {
                "lora_rank": 64,
                "lora_alpha": 128,
            }
        }
        controls = {
            "lora_rank": 8,  # Flat key should override model.lora_rank
        }

        merged, tracking = merge_parameters(recipe_defaults, controls, {})

        assert merged["model"]["lora_rank"] == 8
        assert tracking["model.lora_rank"]["source"] == "control"

    def test_nested_override_works(self):
        """Nested structure in controls should override nested structure in recipe."""
        recipe_defaults = {
            "optimizer": {
                "lr": 1e-4,
                "weight_decay": 0.01,
            }
        }
        controls = {
            "optimizer": {
                "lr": 2e-4,
            }
        }

        merged, tracking = merge_parameters(recipe_defaults, controls, {})

        assert merged["optimizer"]["lr"] == 2e-4
        assert merged["optimizer"]["weight_decay"] == 0.01  # Preserved
        assert tracking["optimizer.lr"]["source"] == "control"

    def test_run_flat_key_overrides_control_nested(self):
        """Run flat key should override control's nested value."""
        recipe_defaults = {
            "model": {
                "lora_rank": 64,
            }
        }
        controls = {
            "model": {
                "lora_rank": 16,
            }
        }
        run_params = {
            "lora_rank": 8,  # Flat key should win
        }

        merged, tracking = merge_parameters(recipe_defaults, controls, run_params)

        assert merged["model"]["lora_rank"] == 8
        assert tracking["model.lora_rank"]["source"] == "run"

    def test_unknown_keys_not_added(self):
        """Keys not in recipe should not be added to merged config."""
        recipe_defaults = {
            "epochs": 1,
        }
        controls = {
            "epochs": 2,
            "unknown_param": "should_not_appear",  # Not in recipe
        }

        merged, tracking = merge_parameters(recipe_defaults, controls, {})

        assert merged["epochs"] == 2
        assert "unknown_param" not in merged

    def test_tracking_captures_all_sources(self):
        """Tracking should correctly identify source of each parameter."""
        recipe_defaults = {
            "epochs": 1,
            "batch_size": 4,
            "lr": 1e-4,
        }
        controls = {
            "batch_size": 8,
        }
        run_params = {
            "lr": 2e-4,
        }

        merged, tracking = merge_parameters(recipe_defaults, controls, run_params)

        assert tracking["epochs"]["source"] == "recipe"
        assert tracking["batch_size"]["source"] == "control"
        assert tracking["lr"]["source"] == "run"

    def test_deep_nesting_preserved(self):
        """Deeply nested structures should be preserved and overridable."""
        recipe_defaults = {
            "checkpointer": {
                "_component_": "torchtune.training.FullModelHFCheckpointer",
                "checkpoint_dir": "/path/to/model",
                "output_dir": "/path/to/output",
            },
            "metric_logger": {
                "_component_": "torchtune.training.metric_logging.DiskLogger",
                "log_dir": "/path/to/logs",
            }
        }
        controls = {}
        run_params = {}

        merged, tracking = merge_parameters(recipe_defaults, controls, run_params)

        assert merged["checkpointer"]["_component_"] == "torchtune.training.FullModelHFCheckpointer"
        assert merged["metric_logger"]["log_dir"] == "/path/to/logs"

    def test_multiple_flat_keys_override_multiple_nested(self):
        """Multiple flat keys should each find and override their nested targets."""
        recipe_defaults = {
            "model": {
                "lora_rank": 64,
                "lora_alpha": 128,
            },
            "optimizer": {
                "lr": 1e-4,
            },
            "lr_scheduler": {
                "num_warmup_steps": 100,
            }
        }
        controls = {
            "lora_rank": 8,
            "lr": 2e-4,
            "num_warmup_steps": 50,
        }

        merged, tracking = merge_parameters(recipe_defaults, controls, {})

        assert merged["model"]["lora_rank"] == 8
        assert merged["optimizer"]["lr"] == 2e-4
        assert merged["lr_scheduler"]["num_warmup_steps"] == 50

    def test_empty_dicts_preserved(self):
        """Empty nested dicts should not cause issues."""
        recipe_defaults = {
            "epochs": 1,
            "model": {},
        }
        controls = {
            "epochs": 2,
        }

        merged, tracking = merge_parameters(recipe_defaults, controls, {})

        assert merged["epochs"] == 2
        assert merged["model"] == {}

    def test_list_values_replaced_not_merged(self):
        """List values should be replaced, not merged."""
        recipe_defaults = {
            "model": {
                "lora_attn_modules": ["q_proj", "v_proj"],
            }
        }
        controls = {
            "model": {
                "lora_attn_modules": ["q_proj", "k_proj", "v_proj"],
            }
        }

        merged, tracking = merge_parameters(recipe_defaults, controls, {})

        assert merged["model"]["lora_attn_modules"] == ["q_proj", "k_proj", "v_proj"]

    def test_none_values_preserved(self):
        """None values should be handled correctly."""
        recipe_defaults = {
            "max_steps_per_epoch": None,
            "seed": None,
        }
        controls = {
            "max_steps_per_epoch": 100,
        }

        merged, tracking = merge_parameters(recipe_defaults, controls, {})

        assert merged["max_steps_per_epoch"] == 100
        assert merged["seed"] is None


class TestMergeParametersWithSampleRecipe:
    """Integration tests for merge_parameters using sample_torchtune_recipe.yaml."""

    def test_override_lora_rank_in_sample(self, sample_recipe):
        """Override lora_rank in sample recipe."""
        controls = {"lora_rank": 8}

        merged, tracking = merge_parameters(sample_recipe, controls, {})

        assert merged["model"]["lora_rank"] == 8
        assert tracking["model.lora_rank"]["source"] == "control"
        # lora_alpha should still be recipe default (128), not recalculated
        # (recalculation happens in apply_recipe_overrides, not merge_parameters)
        assert merged["model"]["lora_alpha"] == 128

    def test_override_learning_rate_in_sample(self, sample_recipe):
        """Override learning rate in sample recipe."""
        controls = {"lr": 1e-5}

        merged, tracking = merge_parameters(sample_recipe, controls, {})

        assert merged["optimizer"]["lr"] == 1e-5
        assert tracking["optimizer.lr"]["source"] == "control"

    def test_run_overrides_control_in_sample(self, sample_recipe):
        """Run parameter should override control in sample recipe."""
        controls = {"lora_rank": 16, "epochs": 2}
        run_params = {"lora_rank": 4}

        merged, tracking = merge_parameters(sample_recipe, controls, run_params)

        assert merged["model"]["lora_rank"] == 4  # Run wins
        assert merged["epochs"] == 2  # Control wins (no run override)
        assert tracking["model.lora_rank"]["source"] == "run"
        assert tracking["epochs"]["source"] == "control"

    def test_all_sample_recipe_keys_tracked(self, sample_recipe):
        """All keys from sample recipe should be tracked."""
        merged, tracking = merge_parameters(sample_recipe, {}, {})

        # Check that major keys are tracked
        assert "epochs" in tracking
        assert "batch_size" in tracking
        assert "model.lora_rank" in tracking
        assert "optimizer.lr" in tracking

    def test_preserve_complex_structures(self, sample_recipe):
        """Complex structures like checkpointer should be preserved."""
        merged, tracking = merge_parameters(sample_recipe, {}, {})

        assert "checkpointer" in merged
        assert merged["checkpointer"]["_component_"] == "torchtune.training.FullModelHFCheckpointer"
        assert "checkpoint_files" in merged["checkpointer"]


class TestLoadRecipeDefaults:
    """Tests for load_recipe_defaults function."""

    def test_loads_yaml_file(self, sample_recipe_path):
        """Should load YAML file and return dict."""
        from recipe_config_loader import load_recipe_defaults

        result = load_recipe_defaults(str(sample_recipe_path))

        assert isinstance(result, dict)
        assert "model" in result
        assert "dataset" in result

    def test_file_not_found(self, tmp_path):
        """Should raise error for non-existent file."""
        from recipe_config_loader import load_recipe_defaults, RecipeExtractionError

        with pytest.raises(RecipeExtractionError, match="not found"):
            load_recipe_defaults(str(tmp_path / "nonexistent.yaml"))

    def test_invalid_yaml(self, tmp_path):
        """Should raise error for invalid YAML."""
        from recipe_config_loader import load_recipe_defaults, RecipeExtractionError

        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [")

        with pytest.raises(RecipeExtractionError, match="Failed to parse"):
            load_recipe_defaults(str(invalid_yaml))

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        """Empty YAML file should return empty dict."""
        from recipe_config_loader import load_recipe_defaults

        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("")

        result = load_recipe_defaults(str(empty_yaml))
        assert result == {}


class TestFormatMergeTracking:
    """Tests for format_merge_tracking function."""

    def test_formats_tracking_by_source(self):
        """Should format tracking grouped by source."""
        tracking = {
            "epochs": {"source": "recipe", "value": 1},
            "batch_size": {"source": "control", "value": 8},
            "lr": {"source": "run", "value": 1e-4},
        }

        result = format_merge_tracking(tracking)

        assert "RECIPE parameters" in result
        assert "CONTROL parameters" in result
        assert "RUN parameters" in result
        assert "epochs = 1" in result
        assert "batch_size = 8" in result

    def test_handles_complex_values(self):
        """Should handle dict and list values in tracking."""
        tracking = {
            "model": {"source": "recipe", "value": {"lora_rank": 64}},
            "list_param": {"source": "control", "value": [1, 2, 3]},
        }

        result = format_merge_tracking(tracking)

        assert "model = <dict>" in result
        assert "list_param = <list>" in result

    def test_empty_tracking(self):
        """Should handle empty tracking dict."""
        result = format_merge_tracking({})
        assert "Parameter merge tracking" in result


# =============================================================================
# Tests for infer_data_format from dataset_config.py
# =============================================================================

from dataset_config import infer_data_format


class TestInferDataFormat:
    """Tests for infer_data_format function."""

    def test_json_extension(self):
        """Should infer json from .json extension."""
        assert infer_data_format("/data/words.json") == "json"

    def test_parquet_extension(self):
        """Should infer parquet from .parquet extension."""
        assert infer_data_format("/data/words.parquet") == "parquet"

    def test_directory_with_json(self, tmp_path):
        """Should infer json from directory with train.json."""
        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        (data_dir / "train.json").touch()

        assert infer_data_format(str(data_dir)) == "json"

    def test_directory_with_parquet(self, tmp_path):
        """Should infer parquet from directory with train.parquet."""
        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        (data_dir / "train.parquet").touch()

        assert infer_data_format(str(data_dir)) == "parquet"

    def test_unknown_format_raises(self, tmp_path):
        """Should raise for unknown format."""
        with pytest.raises(ValueError, match="Cannot infer data format"):
            infer_data_format(str(tmp_path / "unknown.csv"))


# =============================================================================
# Integration Tests: Full Pipeline from experiment_summary to finetune.yaml
# =============================================================================

from unittest.mock import patch, MagicMock
from merge_recipe_params import merge_for_run, apply_recipe_overrides


class TestFullPipelineIntegration:
    """Integration tests for the complete experiment_summary → finetune.yaml pipeline.

    These tests validate that the full workflow produces correct finetune.yaml
    configurations from experiment parameters.
    """

    @pytest.fixture
    def mock_recipe_config(self, sample_recipe):
        """Mock get_recipe_config to return sample recipe without needing tune CLI."""
        with patch('merge_recipe_params.get_recipe_config') as mock:
            mock.return_value = sample_recipe
            yield mock

    def test_merge_for_run_basic(self, mock_recipe_config, sample_recipe):
        """Test merge_for_run produces valid configuration."""
        controls = {"epochs": 2, "batch_size": 8}
        run_params = {"lora_rank": 4}
        additional_params = {
            "data_path": "/data/words_5L",
            "data_format": "json",
            "dataset_type": "instruct_dataset",
            "validation_during_training": False,
            "output_dir_base": "/scratch/outputs/",
            "experiment_name": "test_exp",
            "my_wandb_run_name": "run_001",
            "my_wandb_project": "test_project",
        }

        result = merge_for_run(
            recipe_name="llama3_2/1B_lora_single_device",
            controls=controls,
            run_parameters=run_params,
            additional_params=additional_params,
        )

        # Verify recipe was loaded
        mock_recipe_config.assert_called_once_with("llama3_2/1B_lora_single_device")

        # Verify controls applied
        assert result["epochs"] == 2
        assert result["batch_size"] == 8

        # Verify run parameters override (lora_rank and calculated lora_alpha)
        assert result["model"]["lora_rank"] == 4
        assert result["model"]["lora_alpha"] == 8  # 2 * 4

        # Verify dataset replaced
        assert result["dataset"]["_component_"] == "torchtune.datasets.instruct_dataset"
        assert result["dataset"]["source"] == "json"

        # Verify output directory constructed
        assert result["output_dir"] == "/scratch/outputs/test_exp/ck-out-run_001/"

        # Verify wandb config
        assert result["my_wandb_run_name"] == "run_001"
        assert result["my_wandb_project"] == "test_project"

    def test_merge_for_run_with_validation(self, mock_recipe_config):
        """Test merge_for_run with validation dataset enabled."""
        controls = {}
        run_params = {}
        additional_params = {
            "data_path": "/data/words_5L",
            "data_format": "json",
            "dataset_type": "instruct_dataset",
            "validation_during_training": True,
        }

        result = merge_for_run(
            recipe_name="llama3_2/1B_lora_single_device",
            controls=controls,
            run_parameters=run_params,
            additional_params=additional_params,
        )

        # Verify validation dataset added
        assert "dataset_val" in result
        assert result["dataset_val"]["field"] == "validation"
        assert "run_val_every_n_steps" in result

    def test_merge_for_run_strips_slurm_params(self, mock_recipe_config):
        """SLURM-only parameters should be stripped from final config."""
        controls = {"time": "00:30:00", "gpus": 1, "epochs": 2}
        run_params = {}
        additional_params = {}

        result = merge_for_run(
            recipe_name="llama3_2/1B_lora_single_device",
            controls=controls,
            run_parameters=run_params,
            additional_params=additional_params,
        )

        # SLURM params should be stripped
        assert "time" not in result
        assert "gpus" not in result
        # But epochs should remain
        assert result["epochs"] == 2

    def test_merge_for_run_run_overrides_control(self, mock_recipe_config):
        """Run parameters should override control parameters."""
        controls = {"lora_rank": 16, "epochs": 2}
        run_params = {"lora_rank": 4}  # Override
        additional_params = {}

        result = merge_for_run(
            recipe_name="llama3_2/1B_lora_single_device",
            controls=controls,
            run_parameters=run_params,
            additional_params=additional_params,
        )

        assert result["model"]["lora_rank"] == 4  # Run wins
        assert result["epochs"] == 2  # Control value (no run override)

    def test_merge_for_run_preserves_recipe_structure(self, mock_recipe_config, sample_recipe):
        """Recipe structure (checkpointer, optimizer, etc.) should be preserved."""
        result = merge_for_run(
            recipe_name="llama3_2/1B_lora_single_device",
            controls={},
            run_parameters={},
            additional_params={},
        )

        # Check key structures preserved
        assert "checkpointer" in result
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert "model" in result

        # Verify components preserved
        assert result["checkpointer"]["_component_"] == sample_recipe["checkpointer"]["_component_"]
        assert result["optimizer"]["_component_"] == sample_recipe["optimizer"]["_component_"]


class TestExperimentSummaryScenarios:
    """Test scenarios that mirror real experiment_summary.yaml configurations."""

    @pytest.fixture
    def mock_recipe_config(self, sample_recipe):
        """Mock get_recipe_config to return sample recipe."""
        with patch('merge_recipe_params.get_recipe_config') as mock:
            mock.return_value = sample_recipe
            yield mock

    def test_lora_rank_comparison_experiment(self, mock_recipe_config):
        """Simulate a LoRA rank comparison experiment with multiple runs."""
        # Experiment-level controls (shared across runs)
        # Note: system_prompt goes in additional_params, not controls
        controls = {
            "epochs": 2,
            "batch_size": 4,
        }

        # Additional params (shared) - includes dataset configuration
        additional_base = {
            "data_path": "/data/green/capitalization/words_5L_80P_1000",
            "data_format": "json",
            "dataset_type": "instruct_dataset",
            "validation_during_training": True,
            "output_dir_base": "/scratch/outputs/",
            "experiment_name": "lora_comparison",
            "my_wandb_project": "cap_experiments",
            "system_prompt": "You are a helpful assistant.",
        }

        # Run 1: lora_rank=4
        additional_run1 = {**additional_base, "my_wandb_run_name": "rank4_run"}
        result1 = merge_for_run(
            recipe_name="llama3_2/1B_lora_single_device",
            controls=controls,
            run_parameters={"lora_rank": 4},
            additional_params=additional_run1,
        )

        # Run 2: lora_rank=8
        additional_run2 = {**additional_base, "my_wandb_run_name": "rank8_run"}
        result2 = merge_for_run(
            recipe_name="llama3_2/1B_lora_single_device",
            controls=controls,
            run_parameters={"lora_rank": 8},
            additional_params=additional_run2,
        )

        # Verify run-specific parameters
        assert result1["model"]["lora_rank"] == 4
        assert result1["model"]["lora_alpha"] == 8
        assert result1["output_dir"] == "/scratch/outputs/lora_comparison/ck-out-rank4_run/"

        assert result2["model"]["lora_rank"] == 8
        assert result2["model"]["lora_alpha"] == 16
        assert result2["output_dir"] == "/scratch/outputs/lora_comparison/ck-out-rank8_run/"

        # Verify shared parameters
        for result in [result1, result2]:
            assert result["epochs"] == 2
            assert result["batch_size"] == 4
            assert result["dataset"]["new_system_prompt"] == "You are a helpful assistant."
            assert "dataset_val" in result

    def test_learning_rate_sweep(self, mock_recipe_config):
        """Simulate a learning rate sweep experiment."""
        controls = {"epochs": 1, "lora_rank": 8}
        additional_base = {
            "data_path": "/data/test",
            "data_format": "json",
            "output_dir_base": "/scratch/outputs/",
            "experiment_name": "lr_sweep",
            "my_wandb_project": "lr_experiments",
        }

        learning_rates = [1e-5, 5e-5, 1e-4]
        results = []

        for lr in learning_rates:
            run_name = f"lr_{lr:.0e}".replace("-", "m")
            additional = {**additional_base, "my_wandb_run_name": run_name}
            result = merge_for_run(
                recipe_name="llama3_2/1B_lora_single_device",
                controls=controls,
                run_parameters={"lr": lr},
                additional_params=additional,
            )
            results.append(result)

        # Verify each run has correct learning rate
        for lr, result in zip(learning_rates, results):
            assert result["optimizer"]["lr"] == lr

        # All should have same lora_rank from controls
        for result in results:
            assert result["model"]["lora_rank"] == 8

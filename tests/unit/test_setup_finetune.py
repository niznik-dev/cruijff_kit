"""Unit tests for tools/torchtune/setup_finetune.py"""

import argparse
import warnings
import pytest
from cruijff_kit.tools.torchtune.setup_finetune import (
    parse_epochs,
    parse_bool,
    calculate_lora_alpha,
    validate_lr_scheduler,
    validate_dataset_type,
    construct_output_dir,
    configure_dataset_for_format,
    extract_flat_params,
    compute_training_steps,
    warn_on_low_steps,
    RECIPE_PARAM_MAPPING,
)


def test_parse_epochs_all():
    """Test that 'all' is parsed correctly."""
    result = parse_epochs("all")
    assert result == "all"


def test_parse_epochs_none():
    """Test that 'none' is parsed as empty list."""
    result = parse_epochs("none")
    assert result == []


def test_parse_epochs_comma_delimited():
    """Test comma-delimited epochs."""
    result = parse_epochs("0,1,2")
    assert result == [0, 1, 2]


def test_parse_epochs_single_value():
    """Test single epoch value."""
    result = parse_epochs("5")
    assert result == [5]


def test_parse_epochs_with_whitespace():
    """Test that whitespace is handled correctly in comma-delimited list."""
    result = parse_epochs("0, 1, 2")
    assert result == [0, 1, 2]


def test_parse_epochs_case_insensitive_all():
    """Test that 'ALL' is case-insensitive."""
    result = parse_epochs("ALL")
    assert result == "all"


def test_parse_epochs_case_insensitive_none():
    """Test that 'NONE' is case-insensitive."""
    result = parse_epochs("NONE")
    assert result == []


def test_parse_epochs_invalid_format():
    """Test that invalid epoch format raises ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        parse_epochs("1,2,abc")
    assert "Invalid epochs format" in str(exc_info.value)


def test_parse_epochs_invalid_text():
    """Test that random text raises ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        parse_epochs("invalid")
    assert "Invalid epochs format" in str(exc_info.value)


# Tests for parse_bool()

@pytest.mark.parametrize("value,expected", [
    ("true", True),
    ("True", True),
    ("TRUE", True),
    ("1", True),
    ("yes", True),
    ("Yes", True),
    ("YES", True),
])
def test_parse_bool_true_values(value, expected):
    """Test that various true representations are parsed correctly."""
    assert parse_bool(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("false", False),
    ("False", False),
    ("FALSE", False),
    ("0", False),
    ("no", False),
    ("No", False),
    ("NO", False),
])
def test_parse_bool_false_values(value, expected):
    """Test that various false representations are parsed correctly."""
    assert parse_bool(value) == expected


def test_parse_bool_already_boolean():
    """Test that actual boolean values pass through unchanged."""
    assert parse_bool(True) is True
    assert parse_bool(False) is False


def test_parse_bool_invalid_value():
    """Test that invalid values raise ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        parse_bool("invalid")
    assert "Boolean value expected" in str(exc_info.value)
    assert "invalid" in str(exc_info.value)


def test_parse_bool_invalid_numeric():
    """Test that invalid numeric strings raise ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        parse_bool("2")
    assert "Boolean value expected" in str(exc_info.value)


# Tests for calculate_lora_alpha()

def test_calculate_lora_alpha():
    """Test LoRA alpha calculation (alpha = 2 * rank) for standard values."""
    assert calculate_lora_alpha(8) == 16
    assert calculate_lora_alpha(16) == 32
    assert calculate_lora_alpha(32) == 64
    assert calculate_lora_alpha(64) == 128


# Tests for validate_lr_scheduler()

@pytest.mark.parametrize("scheduler_name", [
    "get_cosine_schedule_with_warmup",
    "get_linear_schedule_with_warmup",
    "get_constant_schedule_with_warmup",
    "get_exponential_schedule_with_warmup",
])
def test_validate_lr_scheduler_valid(scheduler_name):
    """Test that valid lr_scheduler names pass validation."""
    # Should not raise any exception
    validate_lr_scheduler(scheduler_name)


def test_validate_lr_scheduler_invalid():
    """Test that invalid lr_scheduler raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        validate_lr_scheduler("invalid_scheduler")
    assert "Invalid lr_scheduler" in str(exc_info.value)
    assert "invalid_scheduler" in str(exc_info.value)


# Tests for validate_dataset_type()

@pytest.mark.parametrize("dataset_type", [
    "chat_completion",
    "instruct_dataset",
    "chat_dataset",
    "text_completion_dataset",
])
def test_validate_dataset_type_valid(dataset_type):
    """Test that valid dataset_type names pass validation."""
    validate_dataset_type(dataset_type)


def test_validate_dataset_type_invalid():
    """Test that invalid dataset_type raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        validate_dataset_type("invalid_dataset")
    assert "Invalid dataset_type" in str(exc_info.value)
    assert "invalid_dataset" in str(exc_info.value)


# Tests for construct_output_dir()

def test_construct_output_dir_with_experiment_and_trailing_slash():
    """Test output dir construction with experiment name and trailing slash."""
    result = construct_output_dir(
        output_dir_base="/scratch/output/",
        experiment_name="my_experiment",
        model_run_name="run_123"
    )
    assert result == "/scratch/output/my_experiment/ck-out-run_123/"


def test_construct_output_dir_with_experiment_no_trailing_slash():
    """Test output dir construction with experiment name, no trailing slash."""
    result = construct_output_dir(
        output_dir_base="/scratch/output",
        experiment_name="my_experiment",
        model_run_name="run_123"
    )
    assert result == "/scratch/output/my_experiment/ck-out-run_123/"


def test_construct_output_dir_without_experiment_trailing_slash():
    """Test output dir construction without experiment name, with trailing slash."""
    result = construct_output_dir(
        output_dir_base="/scratch/output/",
        experiment_name="",
        model_run_name="run_123"
    )
    assert result == "/scratch/output/ck-out-run_123/"


def test_construct_output_dir_without_experiment_no_trailing_slash():
    """Test output dir construction without experiment name, no trailing slash."""
    result = construct_output_dir(
        output_dir_base="/scratch/output",
        experiment_name="",
        model_run_name="run_123"
    )
    assert result == "/scratch/output/ck-out-run_123/"


def test_construct_output_dir_experiment_name_none():
    """Test output dir construction when experiment_name is None."""
    result = construct_output_dir(
        output_dir_base="/scratch/output/",
        experiment_name=None,
        model_run_name="run_123"
    )
    assert result == "/scratch/output/ck-out-run_123/"


# Tests for configure_dataset_for_format()

def test_configure_dataset_parquet_with_validation():
    """Test parquet format configuration with validation dataset."""
    config = {
        "dataset": {"data_dir": "/data/my_dataset"},
        "dataset_val": {"data_dir": "/data/my_dataset"}
    }

    result = configure_dataset_for_format(
        config,
        dataset_label="my_dataset",
        dataset_ext=".parquet",
        dataset_type="instruct_dataset"  # type doesn't matter for parquet
    )

    assert result["dataset_label"] == "my_dataset"
    assert result["dataset"]["data_dir"] == "/data/my_dataset/train.parquet"
    assert result["dataset_val"]["data_dir"] == "/data/my_dataset/validation.parquet"


def test_configure_dataset_parquet_without_validation():
    """Test parquet format configuration without validation dataset."""
    config = {
        "dataset": {"data_dir": "/data/my_dataset"}
    }

    result = configure_dataset_for_format(
        config,
        dataset_label="my_dataset",
        dataset_ext=".parquet",
        dataset_type="instruct_dataset"
    )

    assert result["dataset_label"] == "my_dataset"
    assert result["dataset"]["data_dir"] == "/data/my_dataset/train.parquet"
    assert "dataset_val" not in result


def test_configure_dataset_json_instruct_with_validation():
    """Test JSON instruct_dataset format with validation dataset."""
    config = {
        "dataset": {"data_dir": "/data/my_dataset", "split": "train"},
        "dataset_val": {"data_dir": "/data/my_dataset", "split": "validation"}
    }

    result = configure_dataset_for_format(
        config,
        dataset_label="my_dataset",
        dataset_ext=".json",
        dataset_type="instruct_dataset"
    )

    assert result["dataset_label"] == "my_dataset"
    assert result["dataset"]["source"] == "json"
    assert result["dataset"]["data_files"] == "/data/my_dataset.json"
    assert result["dataset"]["field"] == "train"
    assert "split" not in result["dataset"]
    assert "data_dir" not in result["dataset"]

    assert result["dataset_val"]["source"] == "json"
    assert result["dataset_val"]["data_files"] == "/data/my_dataset.json"
    assert result["dataset_val"]["field"] == "validation"
    assert "split" not in result["dataset_val"]
    assert "data_dir" not in result["dataset_val"]


def test_configure_dataset_json_instruct_without_validation():
    """Test JSON instruct_dataset format without validation dataset."""
    config = {
        "dataset": {"data_dir": "/data/my_dataset", "split": "train"}
    }

    result = configure_dataset_for_format(
        config,
        dataset_label="my_dataset",
        dataset_ext=".json",
        dataset_type="instruct_dataset"
    )

    assert result["dataset_label"] == "my_dataset"
    assert result["dataset"]["source"] == "json"
    assert result["dataset"]["data_files"] == "/data/my_dataset.json"
    assert result["dataset"]["field"] == "train"
    assert "split" not in result["dataset"]
    assert "data_dir" not in result["dataset"]
    assert "dataset_val" not in result


def test_configure_dataset_json_chat_with_validation():
    """Test JSON chat_dataset format with validation dataset."""
    config = {
        "dataset": {"data_dir": "/data/my_dataset", "split": "train"},
        "dataset_val": {"data_dir": "/data/my_dataset", "split": "validation"}
    }

    result = configure_dataset_for_format(
        config,
        dataset_label="my_dataset",
        dataset_ext=".json",
        dataset_type="chat_dataset"
    )

    assert result["dataset_label"] == "my_dataset"
    assert result["dataset"]["source"] == "json"
    assert result["dataset"]["data_files"] == "/data/my_dataset/train.json"
    assert "split" not in result["dataset"]
    assert "data_dir" not in result["dataset"]

    assert result["dataset_val"]["source"] == "json"
    assert result["dataset_val"]["data_files"] == "/data/my_dataset/validation.json"
    assert "split" not in result["dataset_val"]
    assert "data_dir" not in result["dataset_val"]


def test_configure_dataset_json_chat_without_validation():
    """Test JSON chat_dataset format without validation dataset."""
    config = {
        "dataset": {"data_dir": "/data/my_dataset", "split": "train"}
    }

    result = configure_dataset_for_format(
        config,
        dataset_label="my_dataset",
        dataset_ext=".json",
        dataset_type="chat_dataset"
    )

    assert result["dataset_label"] == "my_dataset"
    assert result["dataset"]["source"] == "json"
    assert result["dataset"]["data_files"] == "/data/my_dataset/train.json"
    assert "split" not in result["dataset"]
    assert "data_dir" not in result["dataset"]
    assert "dataset_val" not in result


def test_configure_dataset_chat_completion():
    """Test chat_completion format - should not modify dataset config."""
    config = {
        "dataset": {
            "data_files": "/data/my_dataset.json",
            "split": "train",
            "model_path": "/models/Llama-3.2-1B-Instruct",
            "prompt": "{input}\n",
            "system_prompt": ""
        },
        "dataset_val": {
            "data_files": "/data/my_dataset.json",
            "split": "validation",
            "model_path": "/models/Llama-3.2-1B-Instruct",
            "prompt": "{input}\n",
            "system_prompt": ""
        }
    }

    result = configure_dataset_for_format(
        config,
        dataset_label="my_dataset",
        dataset_ext=".json",
        dataset_type="chat_completion"
    )

    # chat_completion should pass through config unchanged (except dataset_label)
    assert result["dataset_label"] == "my_dataset"
    assert result["dataset"]["data_files"] == "/data/my_dataset.json"
    assert result["dataset"]["split"] == "train"
    assert result["dataset"]["model_path"] == "/models/Llama-3.2-1B-Instruct"
    assert result["dataset"]["prompt"] == "{input}\n"
    assert result["dataset_val"]["data_files"] == "/data/my_dataset.json"
    assert result["dataset_val"]["split"] == "validation"


# Tests for extract_flat_params()

def test_extract_flat_params_basic():
    """Test extracting flat parameters from nested recipe config."""
    recipe_config = {
        "model": {
            "lora_rank": 64,
            "lora_dropout": 0.1,
        },
        "optimizer": {
            "lr": 3e-4,
            "weight_decay": 0.01,
        },
        "batch_size": 4,
        "epochs": 2,
    }

    result = extract_flat_params(recipe_config, RECIPE_PARAM_MAPPING)

    assert result["lora_rank"] == 64
    assert result["lora_dropout"] == 0.1
    assert result["lr"] == 3e-4
    assert result["weight_decay"] == 0.01
    assert result["batch_size"] == 4
    assert result["epochs"] == 2


def test_extract_flat_params_missing_keys():
    """Test that missing keys are silently skipped."""
    recipe_config = {
        "model": {
            "lora_rank": 64,
            # lora_dropout is missing
        },
        # optimizer section is missing entirely
        "batch_size": 4,
    }

    result = extract_flat_params(recipe_config, RECIPE_PARAM_MAPPING)

    assert result["lora_rank"] == 64
    assert result["batch_size"] == 4
    assert "lora_dropout" not in result
    assert "lr" not in result
    assert "weight_decay" not in result


def test_extract_flat_params_empty_config():
    """Test with empty recipe config."""
    recipe_config = {}

    result = extract_flat_params(recipe_config, RECIPE_PARAM_MAPPING)

    assert result == {}


def test_extract_flat_params_deeply_nested():
    """Test extracting from deeply nested config."""
    recipe_config = {
        "lr_scheduler": {
            "num_warmup_steps": 100,
        },
        "tokenizer": {
            "max_seq_len": 2048,
        },
    }

    result = extract_flat_params(recipe_config, RECIPE_PARAM_MAPPING)

    assert result["num_warmup_steps"] == 100
    assert result["max_seq_len"] == 2048


def test_extract_flat_params_with_none_values():
    """Test that None values are extracted correctly."""
    recipe_config = {
        "model": {
            "lora_rank": None,
        },
        "batch_size": None,
    }

    result = extract_flat_params(recipe_config, RECIPE_PARAM_MAPPING)

    assert result["lora_rank"] is None
    assert result["batch_size"] is None


def test_extract_flat_params_custom_mapping():
    """Test with a custom mapping dictionary."""
    recipe_config = {
        "custom": {
            "nested": {
                "value": 42,
            }
        },
        "top_level": "hello",
    }

    custom_mapping = {
        "custom.nested.value": "my_value",
        "top_level": "my_top",
    }

    result = extract_flat_params(recipe_config, custom_mapping)

    assert result["my_value"] == 42
    assert result["my_top"] == "hello"


def test_extract_flat_params_all_mapped_params():
    """Test extraction of all parameters in RECIPE_PARAM_MAPPING."""
    # Create a recipe config that has all the mapped parameters
    recipe_config = {
        "model": {
            "lora_rank": 8,
            "lora_dropout": 0.05,
        },
        "optimizer": {
            "lr": 1e-5,
            "weight_decay": 0.02,
        },
        "batch_size": 2,
        "epochs": 3,
        "gradient_accumulation_steps": 16,
        "lr_scheduler": {
            "num_warmup_steps": 50,
        },
        "tokenizer": {
            "max_seq_len": 4096,
        },
    }

    result = extract_flat_params(recipe_config, RECIPE_PARAM_MAPPING)

    # Verify all mapped parameters are extracted
    assert result["lora_rank"] == 8
    assert result["lora_dropout"] == 0.05
    assert result["lr"] == 1e-5
    assert result["weight_decay"] == 0.02
    assert result["batch_size"] == 2
    assert result["epochs"] == 3
    assert result["gradient_accumulation_steps"] == 16
    assert result["num_warmup_steps"] == 50
    assert result["max_seq_len"] == 4096


def test_recipe_param_mapping_has_expected_keys():
    """Test that RECIPE_PARAM_MAPPING contains all expected parameter mappings."""
    expected_mappings = {
        "model.lora_rank": "lora_rank",
        "model.lora_dropout": "lora_dropout",
        "optimizer.lr": "lr",
        "optimizer.weight_decay": "weight_decay",
        "batch_size": "batch_size",
        "epochs": "epochs",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "lr_scheduler.num_warmup_steps": "num_warmup_steps",
        "tokenizer.max_seq_len": "max_seq_len",
    }

    for recipe_path, arg_name in expected_mappings.items():
        assert recipe_path in RECIPE_PARAM_MAPPING, f"Missing mapping for {recipe_path}"
        assert RECIPE_PARAM_MAPPING[recipe_path] == arg_name


# Tests for compute_training_steps()

def test_compute_training_steps_basic():
    """Test basic step computation: 1000 samples, batch 4, no accumulation, 1 epoch."""
    result = compute_training_steps(
        training_samples=1000, batch_size=4,
        gradient_accumulation_steps=1, epochs=1
    )
    assert result['steps_per_epoch'] == 250
    assert result['total_steps'] == 250
    assert result['effective_batch_size'] == 4


def test_compute_training_steps_with_accumulation():
    """Test step computation with gradient accumulation."""
    result = compute_training_steps(
        training_samples=1000, batch_size=4,
        gradient_accumulation_steps=8, epochs=1
    )
    # effective batch = 32, steps = ceil(1000/32) = 32
    assert result['steps_per_epoch'] == 32
    assert result['total_steps'] == 32
    assert result['effective_batch_size'] == 32


def test_compute_training_steps_multiple_epochs():
    """Test step computation with multiple epochs."""
    result = compute_training_steps(
        training_samples=100, batch_size=10,
        gradient_accumulation_steps=1, epochs=3
    )
    assert result['steps_per_epoch'] == 10
    assert result['total_steps'] == 30


def test_compute_training_steps_ceiling_division():
    """Test that partial batches are counted (ceil division)."""
    result = compute_training_steps(
        training_samples=101, batch_size=10,
        gradient_accumulation_steps=1, epochs=1
    )
    # ceil(101/10) = 11
    assert result['steps_per_epoch'] == 11
    assert result['total_steps'] == 11


def test_compute_training_steps_large_batch_collapse():
    """Test the scenario that motivated this feature: large effective batch collapses steps."""
    result = compute_training_steps(
        training_samples=100, batch_size=32,
        gradient_accumulation_steps=8, epochs=1
    )
    # effective batch = 256, steps = ceil(100/256) = 1
    assert result['steps_per_epoch'] == 1
    assert result['total_steps'] == 1
    assert result['effective_batch_size'] == 256


def test_compute_training_steps_exact_division():
    """Test when samples divide evenly into batches."""
    result = compute_training_steps(
        training_samples=100, batch_size=10,
        gradient_accumulation_steps=1, epochs=1
    )
    assert result['steps_per_epoch'] == 10
    assert result['total_steps'] == 10


# Tests for warn_on_low_steps()

def test_warn_on_low_steps_no_warnings(capsys):
    """Test no warnings when steps are sufficient."""
    step_info = {'steps_per_epoch': 250, 'total_steps': 250, 'effective_batch_size': 4}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_low_steps(step_info, num_warmup_steps=100)
        assert len(w) == 0
    captured = capsys.readouterr()
    assert "250 total steps" in captured.out


def test_warn_on_low_steps_warmup_exceeds_total():
    """Test warning when warmup steps exceed total steps."""
    step_info = {'steps_per_epoch': 14, 'total_steps': 14, 'effective_batch_size': 256}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_low_steps(step_info, num_warmup_steps=100)
        warning_messages = [str(x.message) for x in w]
        assert any("warmup" in msg.lower() for msg in warning_messages)


def test_warn_on_low_steps_below_3x_warmup():
    """Test warning when total steps < 3 * warmup steps."""
    step_info = {'steps_per_epoch': 10, 'total_steps': 10, 'effective_batch_size': 100}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_low_steps(step_info, num_warmup_steps=5)
        warning_messages = [str(x.message) for x in w]
        assert any("3x warmup" in msg for msg in warning_messages)


def test_warn_on_low_steps_both_warnings():
    """Test both warnings fire when steps < warmup and < 3x warmup."""
    step_info = {'steps_per_epoch': 1, 'total_steps': 1, 'effective_batch_size': 256}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_low_steps(step_info, num_warmup_steps=100)
        assert len(w) == 2


def test_warn_on_low_steps_exactly_3x_warmup_no_warning():
    """Test that exactly 3x warmup steps does not trigger the low-steps warning."""
    step_info = {'steps_per_epoch': 30, 'total_steps': 30, 'effective_batch_size': 20}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_low_steps(step_info, num_warmup_steps=10)
        assert len(w) == 0


def test_warn_on_low_steps_above_50_but_below_3x_warmup():
    """Test that total > 50 but < 3*warmup still warns (would have passed old check)."""
    # warmup=20, total=50: 50 < 60 (3*20), should warn
    step_info = {'steps_per_epoch': 50, 'total_steps': 50, 'effective_batch_size': 20}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_low_steps(step_info, num_warmup_steps=20)
        warning_messages = [str(x.message) for x in w]
        assert len(w) == 1
        assert any("3x warmup" in msg for msg in warning_messages)


def test_warn_on_low_steps_below_50_but_above_3x_warmup():
    """Test that total < 50 but > 3*warmup does NOT warn (would have failed old check)."""
    # warmup=5, total=34: 34 > 15 (3*5), should not warn
    step_info = {'steps_per_epoch': 34, 'total_steps': 34, 'effective_batch_size': 3}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_on_low_steps(step_info, num_warmup_steps=5)
        assert len(w) == 0

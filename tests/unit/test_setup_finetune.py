"""Unit tests for tools/torchtune/setup_finetune.py"""

import argparse
import pytest
from cruijff_kit.tools.torchtune.setup_finetune import (
    parse_epochs,
    validate_lr_scheduler,
    validate_dataset_type,
    construct_output_dir,
    configure_dataset_for_format,
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
    "instruct_dataset",
    "chat_dataset",
    "text_completion_dataset",
])
def test_validate_dataset_type_valid(dataset_type):
    """Test that valid dataset_type names pass validation."""
    # Should not raise any exception
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

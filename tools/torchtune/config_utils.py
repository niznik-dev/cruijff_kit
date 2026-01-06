"""Shared utilities for torchtune configuration generation.

This module contains basic parsing and validation utilities used by both
recipe-based (merge_recipe_params.py) and template-based (generate_finetune_yaml.py)
workflows.

These are simple, stateless functions with no workflow-specific logic.
"""

import argparse
import os
from typing import Union, List


# Parameters that are SLURM-only and should not appear in finetune.yaml
SLURM_ONLY_PARAMS = [
    'time', 'gpus', 'conda_env', 'venv', 'modules',
    'account', 'partition', 'constraint', 'custom_recipe'
]


def parse_epochs(value: Union[str, List[int]]) -> Union[str, List[int]]:
    """Parse epochs_to_save parameter.

    Args:
        value: 'all', 'none', comma-separated string '1,2,3', or already-parsed list

    Returns:
        'all' string, empty list (for 'none'), or list of integers

    Raises:
        argparse.ArgumentTypeError: If value format is invalid
    """
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower == 'none':
            return []
        elif value_lower == 'all':
            return 'all'
        else:
            try:
                return [int(x.strip()) for x in value.split(',')]
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid epochs format: {value}")

    raise argparse.ArgumentTypeError(f"Invalid epochs type: {type(value)}")


def parse_bool(value: Union[bool, str]) -> bool:
    """Parse boolean values from strings or booleans.

    Args:
        value: String or boolean to parse

    Returns:
        Boolean value

    Raises:
        argparse.ArgumentTypeError: If value cannot be parsed as boolean

    Accepts: true/false, True/False, 1/0, yes/no (case-insensitive)
    """
    if isinstance(value, bool):
        return value

    value_lower = str(value).lower()
    if value_lower in ('true', '1', 'yes'):
        return True
    elif value_lower in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Boolean value expected. Got: '{value}'. "
            "Valid values: true/false, yes/no, 1/0 (case-insensitive)"
        )


def calculate_lora_alpha(lora_rank: int) -> int:
    """Calculate LoRA alpha from LoRA rank (always 2 * rank)."""
    return lora_rank * 2


def expand_user_in_paths(config: dict, username: str = None) -> dict:
    """Expand $USER in path values.

    Args:
        config: Configuration dictionary
        username: Username to substitute (defaults to $USER env var)

    Returns:
        Configuration with $USER expanded in path fields
    """
    if username is None:
        username = os.environ.get("USER", "")

    path_keys = ['input_dir', 'output_dir', 'models_dir', 'checkpoint_dir']

    for key in path_keys:
        if key in config and isinstance(config[key], str):
            config[key] = config[key].replace("$USER", username)

    return config


def strip_slurm_params(config: dict) -> dict:
    """Remove SLURM-only parameters before writing finetune.yaml.

    Args:
        config: Configuration dictionary possibly containing SLURM params

    Returns:
        Configuration dictionary with SLURM params removed
    """
    return {k: v for k, v in config.items() if k not in SLURM_ONLY_PARAMS}


def construct_output_dir(output_dir_base: str, experiment_name: str,
                         model_run_name: str) -> str:
    """Construct the full output directory path.

    Args:
        output_dir_base: Base directory for outputs (e.g., /scratch/.../ck-outputs/)
        experiment_name: Experiment name for grouping outputs
        model_run_name: Name of this specific model run

    Returns:
        Full output directory path: {base}/{experiment_name}/ck-out-{run_name}/
    """
    if not output_dir_base.endswith('/'):
        output_dir_base += '/'

    if experiment_name:
        return f"{output_dir_base}{experiment_name}/ck-out-{model_run_name}/"
    else:
        return f"{output_dir_base}ck-out-{model_run_name}/"


# Validation functions

VALID_LR_SCHEDULERS = [
    'get_cosine_schedule_with_warmup',
    'get_linear_schedule_with_warmup',
    'get_constant_schedule_with_warmup',
    'get_exponential_schedule_with_warmup'
]

VALID_DATASET_TYPES = [
    'instruct_dataset',
    'chat_dataset',
    'text_completion_dataset'
]


def validate_lr_scheduler(scheduler_name: str) -> None:
    """Validate learning rate scheduler name.

    Raises:
        ValueError: If scheduler_name is not valid
    """
    if scheduler_name not in VALID_LR_SCHEDULERS:
        raise ValueError(
            f"Invalid lr_scheduler: '{scheduler_name}'. "
            f"Must be one of: {', '.join(VALID_LR_SCHEDULERS)}"
        )


def validate_dataset_type(dataset_type: str) -> None:
    """Validate dataset type.

    Raises:
        ValueError: If dataset_type is not valid
    """
    if dataset_type not in VALID_DATASET_TYPES:
        raise ValueError(
            f"Invalid dataset_type: '{dataset_type}'. "
            f"Must be one of: {', '.join(VALID_DATASET_TYPES)}"
        )

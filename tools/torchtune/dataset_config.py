"""Dataset configuration builder for torchtune.

This module constructs complete dataset configuration sections for torchtune's
finetune.yaml. It handles the differences between:
- Dataset formats: json vs parquet
- Dataset types: instruct_dataset, chat_dataset, text_completion_dataset
- Training vs validation splits

Used by both recipe-based and template-based workflows to construct or replace
the dataset sections in finetune.yaml.
"""

from pathlib import Path
from typing import Dict, Any, Optional


def build_dataset_config(
    data_path: str,
    data_format: str,
    dataset_type: str = "instruct_dataset",
    split: str = "train",
    packed: bool = True,
    train_on_input: bool = False,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a complete dataset configuration for torchtune.

    Args:
        data_path: Path to the dataset file/directory (without split suffix)
                   For JSON instruct: path to .json file (e.g., /path/to/data.json)
                   For JSON chat: path to folder containing train.json, validation.json
                   For Parquet: path to folder containing train.parquet, validation.parquet
        data_format: "json" or "parquet"
        dataset_type: "instruct_dataset", "chat_dataset", or "text_completion_dataset"
        split: "train" or "validation"
        packed: Whether to pack sequences (typically True for training, ignored for val)
        train_on_input: Whether to train on input tokens
        system_prompt: Optional system prompt to inject

    Returns:
        Complete dataset configuration dict ready for finetune.yaml
    """
    config = {
        "_component_": f"torchtune.datasets.{dataset_type}"
    }

    if data_format == "json":
        config["source"] = "json"
        config = _configure_json_dataset(config, data_path, dataset_type, split)
    elif data_format == "parquet":
        config["source"] = "parquet"
        config = _configure_parquet_dataset(config, data_path, split)
    else:
        raise ValueError(f"Unsupported data_format: {data_format}. Use 'json' or 'parquet'.")

    # Add dataset type-specific fields
    if dataset_type == "chat_dataset":
        config["conversation_column"] = "messages"
        config["conversation_style"] = "openai"
    elif dataset_type == "instruct_dataset":
        config["train_on_input"] = train_on_input

    # Add packed for training (not typically for validation)
    if split == "train" and packed:
        config["packed"] = True

    # Add system prompt if specified
    if system_prompt:
        config["new_system_prompt"] = system_prompt

    return config


def _configure_json_dataset(
    config: Dict[str, Any],
    data_path: str,
    dataset_type: str,
    split: str
) -> Dict[str, Any]:
    """Configure dataset for JSON format.

    JSON datasets have two patterns:
    1. instruct_dataset: Single file with train/validation/test keys
       - data_files: /path/to/data.json
       - field: "train" or "validation"

    2. chat_dataset: Folder with separate files per split
       - data_files: /path/to/folder/train.json
       - No field key
    """
    if dataset_type == "instruct_dataset":
        # Single JSON file with splits as keys
        # Ensure path ends with .json
        if not data_path.endswith('.json'):
            data_path = f"{data_path}.json"
        config["data_files"] = data_path
        config["field"] = split
    else:
        # chat_dataset or text_completion_dataset: folder with split files
        # Remove trailing .json if present (we'll add split filename)
        if data_path.endswith('.json'):
            data_path = data_path[:-5]
        config["data_files"] = f"{data_path}/{split}.json"

    return config


def _configure_parquet_dataset(
    config: Dict[str, Any],
    data_path: str,
    split: str
) -> Dict[str, Any]:
    """Configure dataset for Parquet format.

    Parquet datasets are always in folders with split files:
    - data_dir: /path/to/folder/train.parquet
    - split: "train" or "validation"
    """
    # Remove trailing slash if present
    data_path = data_path.rstrip('/')
    config["data_dir"] = f"{data_path}/{split}.parquet"
    config["split"] = split

    return config


def build_dataset_pair(
    data_path: str,
    data_format: str,
    dataset_type: str = "instruct_dataset",
    train_on_input: bool = False,
    system_prompt: Optional[str] = None,
    include_validation: bool = True,
    packed: bool = True,
) -> Dict[str, Any]:
    """Build both training and validation dataset configurations.

    Args:
        data_path: Path to dataset (see build_dataset_config for format)
        data_format: "json" or "parquet"
        dataset_type: "instruct_dataset", "chat_dataset", etc.
        train_on_input: Whether to train on input tokens
        system_prompt: Optional system prompt
        include_validation: Whether to include validation dataset
        packed: Whether to pack training sequences

    Returns:
        Dict with 'dataset' and optionally 'dataset_val' keys
    """
    result = {
        "dataset": build_dataset_config(
            data_path=data_path,
            data_format=data_format,
            dataset_type=dataset_type,
            split="train",
            packed=packed,
            train_on_input=train_on_input,
            system_prompt=system_prompt,
        )
    }

    if include_validation:
        result["dataset_val"] = build_dataset_config(
            data_path=data_path,
            data_format=data_format,
            dataset_type=dataset_type,
            split="validation",
            packed=False,  # Validation typically not packed
            train_on_input=train_on_input,
            system_prompt=system_prompt,
        )

    return result


def infer_data_format(data_path: str) -> str:
    """Infer data format from file path.

    Args:
        data_path: Path to dataset file or directory

    Returns:
        "json" or "parquet"

    Raises:
        ValueError: If format cannot be inferred
    """
    path = Path(data_path)

    if path.suffix == '.json':
        return "json"
    elif path.suffix == '.parquet':
        return "parquet"

    # Check if it's a directory - look for common files
    if path.is_dir():
        if (path / "train.json").exists():
            return "json"
        elif (path / "train.parquet").exists():
            return "parquet"

    raise ValueError(
        f"Cannot infer data format from path: {data_path}. "
        "Specify format explicitly as 'json' or 'parquet'."
    )

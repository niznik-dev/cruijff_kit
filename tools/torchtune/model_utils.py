#!/usr/bin/env python3
"""
Model utilities for cruijff_kit torchtune integration.

Provides functions for:
- Detecting checkpoint files in model directories
- Loading model metadata from registry
- Validating model configurations
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml


def get_model_registry_path() -> Path:
    """Get path to model registry file."""
    return Path(__file__).parent / "model_registry.yaml"


def load_model_registry() -> Dict:
    """Load model registry from YAML file."""
    registry_path = get_model_registry_path()
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. "
            "This file is required for automatic model configuration."
        )

    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)

    return registry.get("models", {})


def get_model_metadata(model_name: str) -> Optional[Dict]:
    """
    Get metadata for a specific model from the registry.

    Args:
        model_name: Name of the model (e.g., "Llama-3.2-1B-Instruct")

    Returns:
        Dict with model metadata, or None if not found
    """
    registry = load_model_registry()
    return registry.get(model_name)


def detect_checkpoint_files(model_dir: Path, model_name: Optional[str] = None) -> List[str]:
    """
    Detect checkpoint files in a model directory.

    First checks the model registry if model_name is provided.
    Falls back to automatic detection if not in registry.

    Args:
        model_dir: Path to model directory
        model_name: Optional model name to look up in registry

    Returns:
        List of checkpoint file names (not full paths)

    Raises:
        ValueError: If no checkpoint files found
        FileNotFoundError: If model directory doesn't exist
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Try registry first if model name provided
    if model_name:
        metadata = get_model_metadata(model_name)
        if metadata and "checkpoint_files" in metadata:
            # Validate that the files actually exist
            checkpoint_files = metadata["checkpoint_files"]
            missing_files = []
            for ckpt_file in checkpoint_files:
                if not (model_dir / ckpt_file).exists():
                    missing_files.append(ckpt_file)

            if missing_files:
                raise ValueError(
                    f"Registry specifies checkpoint files that don't exist in {model_dir}:\n"
                    f"Missing: {missing_files}\n"
                    f"Expected from registry: {checkpoint_files}\n"
                    f"Please verify model directory or update registry."
                )

            return checkpoint_files

    # Fallback to automatic detection
    # Check for single file first
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        return ["model.safetensors"]

    # Check for sharded pattern
    sharded_files = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if sharded_files:
        return [f.name for f in sharded_files]

    # No checkpoint files found
    raise ValueError(
        f"No checkpoint files found in {model_dir}\n"
        f"Expected either:\n"
        f"  - model.safetensors (single file), or\n"
        f"  - model-XXXXX-of-YYYYY.safetensors (sharded files)\n"
        f"\n"
        f"If this is a new model, please add it to the model registry at:\n"
        f"  {get_model_registry_path()}"
    )


def get_tokenizer_path(model_dir: Path, model_name: Optional[str] = None) -> str:
    """
    Get the relative tokenizer path for a model.

    Args:
        model_dir: Path to model directory
        model_name: Optional model name to look up in registry

    Returns:
        Relative path to tokenizer file (e.g., "original/tokenizer.model")
    """
    # Try registry first
    if model_name:
        metadata = get_model_metadata(model_name)
        if metadata and "tokenizer_path" in metadata:
            tokenizer_rel_path = metadata["tokenizer_path"]
            # Validate it exists
            if (model_dir / tokenizer_rel_path).exists():
                return tokenizer_rel_path

    # Fallback: try common locations
    common_paths = [
        "original/tokenizer.model",
        "tokenizer.model",
    ]

    for rel_path in common_paths:
        if (model_dir / rel_path).exists():
            return rel_path

    raise ValueError(
        f"No tokenizer found in {model_dir}\n"
        f"Tried: {common_paths}\n"
        f"Please check model directory or add model to registry."
    )


def get_model_type(model_name: str) -> str:
    """
    Get the model_type value for torchtune checkpointer.

    Args:
        model_name: Name of the model

    Returns:
        Model type string (e.g., "LLAMA3_2", "LLAMA2")
    """
    metadata = get_model_metadata(model_name)
    if metadata and "model_type" in metadata:
        return metadata["model_type"]

    # Fallback heuristics based on name
    if "3.3" in model_name or "3_3" in model_name:
        return "LLAMA3_3"
    elif "3.2" in model_name or "3_2" in model_name:
        return "LLAMA3_2"
    elif "3.1" in model_name or "3_1" in model_name:
        return "LLAMA3_1"
    elif "3" in model_name:
        return "LLAMA3"
    elif "2" in model_name:
        return "LLAMA2"

    raise ValueError(
        f"Cannot determine model_type for {model_name}. "
        f"Please add model to registry at {get_model_registry_path()}"
    )


def get_torchtune_component(model_name: str) -> str:
    """
    Get the torchtune model component for LoRA fine-tuning.

    Args:
        model_name: Name of the model

    Returns:
        Component path (e.g., "torchtune.models.llama3_2.lora_llama3_2_1b")
    """
    metadata = get_model_metadata(model_name)
    if metadata and "torchtune_component" in metadata:
        return metadata["torchtune_component"]

    raise ValueError(
        f"Cannot determine torchtune component for {model_name}. "
        f"Please add model to registry at {get_model_registry_path()}"
    )


def validate_model_directory(
    model_dir: Path,
    model_name: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Validate a model directory and return its configuration.

    Args:
        model_dir: Path to model directory
        model_name: Optional model name
        verbose: Print validation steps

    Returns:
        Dict with validated configuration:
        {
            "checkpoint_files": [...],
            "tokenizer_path": "...",
            "model_type": "...",
            "torchtune_component": "..." (if model_name provided)
        }

    Raises:
        ValueError or FileNotFoundError if validation fails
    """
    if verbose:
        print(f"Validating model directory: {model_dir}")

    result = {}

    # Detect checkpoint files
    checkpoint_files = detect_checkpoint_files(model_dir, model_name)
    result["checkpoint_files"] = checkpoint_files
    if verbose:
        print(f"  ✓ Found checkpoint files: {checkpoint_files}")

    # Get tokenizer path
    tokenizer_path = get_tokenizer_path(model_dir, model_name)
    result["tokenizer_path"] = tokenizer_path
    if verbose:
        print(f"  ✓ Found tokenizer: {tokenizer_path}")

    # Get model type if model name provided
    if model_name:
        model_type = get_model_type(model_name)
        result["model_type"] = model_type
        if verbose:
            print(f"  ✓ Model type: {model_type}")

        torchtune_component = get_torchtune_component(model_name)
        result["torchtune_component"] = torchtune_component
        if verbose:
            print(f"  ✓ Torchtune component: {torchtune_component}")

    if verbose:
        print(f"  ✓ Model directory validated successfully")

    return result


if __name__ == "__main__":
    """
    CLI for testing model utilities.

    Usage:
        python model_utils.py /path/to/model/dir [model_name]
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_utils.py <model_dir> [model_name]")
        print("\nExample:")
        print("  python model_utils.py /scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-1B-Instruct Llama-3.2-1B-Instruct")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        config = validate_model_directory(model_dir, model_name, verbose=True)
        print("\nValidation successful!")
        print(f"\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)

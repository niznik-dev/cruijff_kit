#!/usr/bin/env python3
"""
Simple validation for torchtune configs before submission.

Checks:
- Config file exists and parses
- Model checkpoint files exist
- Dataset files exist
- Tokenizer exists

Usage:
    # Validate single run
    python validate_config.py /path/to/run_directory/

    # Validate all runs in experiment
    python validate_config.py /path/to/experiment_dir/ --all
"""

from pathlib import Path
import sys
import yaml
from typing import List, Tuple


def validate_run_config(run_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate a single run directory's finetune.yaml.

    Args:
        run_dir: Path to run directory

    Returns:
        (success, errors) tuple
    """
    errors = []

    # Check config exists
    config_path = run_dir / "finetune.yaml"
    if not config_path.exists():
        errors.append(f"finetune.yaml not found in {run_dir}")
        return False, errors

    # Parse config
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"Failed to parse finetune.yaml: {e}")
        return False, errors

    # Validate checkpoint files
    try:
        checkpoint_dir_str = config["checkpointer"]["checkpoint_dir"]

        # Handle variable substitution for ${models_dir}
        if "${models_dir}" in checkpoint_dir_str and "models_dir" in config:
            checkpoint_dir_str = checkpoint_dir_str.replace(
                "${models_dir}", config["models_dir"]
            )

        checkpoint_dir = Path(checkpoint_dir_str)
        checkpoint_files = config["checkpointer"]["checkpoint_files"]

        for ckpt_file in checkpoint_files:
            ckpt_path = checkpoint_dir / ckpt_file
            if not ckpt_path.exists():
                errors.append(f"Checkpoint file not found: {ckpt_path}")
    except KeyError as e:
        errors.append(f"Missing checkpoint config: {e}")

    # Validate tokenizer
    try:
        # Handle variable substitution in path
        tokenizer_path_str = config["tokenizer"]["path"]

        # Simple variable substitution for ${models_dir}
        if "${models_dir}" in tokenizer_path_str and "models_dir" in config:
            tokenizer_path_str = tokenizer_path_str.replace(
                "${models_dir}", config["models_dir"]
            )

        tokenizer_path = Path(tokenizer_path_str)
        if not tokenizer_path.exists():
            errors.append(f"Tokenizer not found: {tokenizer_path}")
    except KeyError as e:
        errors.append(f"Missing tokenizer config: {e}")

    # Validate dataset (handle both JSON and Parquet sources)
    try:
        dataset_source = config["dataset"]["source"]

        if dataset_source == "json":
            # JSON dataset with data_files
            data_files = config["dataset"].get("data_files")
            if data_files:
                # Handle variable substitution
                if "${input_dir}" in data_files and "input_dir" in config:
                    data_files = data_files.replace("${input_dir}", config["input_dir"])
                if "${dataset_label}" in data_files and "dataset_label" in config:
                    data_files = data_files.replace("${dataset_label}", config["dataset_label"])

                data_path = Path(data_files)
                if not data_path.exists():
                    errors.append(f"Dataset file not found: {data_path}")

        elif dataset_source == "parquet":
            # Parquet dataset with data_dir
            data_dir = config["dataset"].get("data_dir")
            if data_dir:
                # Handle variable substitution
                if "${input_dir}" in data_dir and "input_dir" in config:
                    data_dir = data_dir.replace("${input_dir}", config["input_dir"])
                if "${dataset_label}" in data_dir and "dataset_label" in config:
                    data_dir = data_dir.replace("${dataset_label}", config["dataset_label"])

                data_path = Path(data_dir)
                if not data_path.exists():
                    errors.append(f"Dataset directory not found: {data_path}")

        # For HuggingFace datasets (source is dataset name), skip file check
    except KeyError as e:
        errors.append(f"Missing dataset config: {e}")

    return len(errors) == 0, errors


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate torchtune configs")
    parser.add_argument("path", type=Path, help="Run directory or experiment directory")
    parser.add_argument("--all", action="store_true", help="Validate all runs in directory")
    args = parser.parse_args()

    if args.all:
        # Validate all run directories
        run_dirs = sorted([d for d in args.path.iterdir() if d.is_dir() and (d / "finetune.yaml").exists()])

        if not run_dirs:
            print(f"No run directories with finetune.yaml found in {args.path}")
            sys.exit(1)

        print(f"Validating {len(run_dirs)} runs in {args.path}\n")

        results = []
        for run_dir in run_dirs:
            success, errors = validate_run_config(run_dir)
            results.append((run_dir.name, success, errors))

        # Print results
        passed = 0
        for run_name, success, errors in results:
            if success:
                print(f"✓ {run_name}")
                passed += 1
            else:
                print(f"✗ {run_name}")
                for error in errors:
                    print(f"    {error}")

        print(f"\nSummary: {passed}/{len(results)} passed")

        if passed < len(results):
            sys.exit(1)

    else:
        # Validate single run
        success, errors = validate_run_config(args.path)

        if success:
            print(f"✓ {args.path.name} validation passed")
        else:
            print(f"✗ {args.path.name} validation failed:")
            for error in errors:
                print(f"  {error}")
            sys.exit(1)


if __name__ == "__main__":
    main()

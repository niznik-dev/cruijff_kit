#!/usr/bin/env python3
"""
Convert nested JSON files to Parquet format for HuggingFace datasets.

This script takes a JSON file with nested structure like:
{
    "train": [...],
    "validation": [...],
    "test": [...]
}

and converts it to Parquet files that can be loaded with load_dataset().

Usage:
    python convert_json_to_hf_dataset.py \
        --input_json /path/to/data.json \
        --output_dir /path/to/output

The output will be Parquet files (one per split) that can be loaded with:
    from datasets import load_dataset
    dataset = load_dataset('parquet', data_dir='/path/to/output')
    # Access splits: dataset['train'], dataset['validation'], etc.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset

from cruijff_kit.utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)


def convert_json_to_hf_dataset(input_json: str, output_dir: str, verbose: bool = True):
    """
    Convert nested JSON file to Parquet files for HuggingFace datasets.

    Args:
        input_json: Path to input JSON file with nested structure
        output_dir: Directory to save the Parquet files
        verbose: Whether to print progress information
    """
    # Load JSON data
    if verbose:
        logger.info(f"Loading JSON from: {input_json}")

    with open(input_json, "r") as f:
        data = json.load(f)

    # Check if data is a dict with split keys
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON to be a dict with split keys (e.g., 'train', 'validation'), "
            f"but got {type(data).__name__}"
        )

    if verbose:
        logger.info(f"Found splits: {list(data.keys())}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save each split as a separate Parquet file
    for split_name, examples in data.items():
        if not isinstance(examples, list):
            raise ValueError(
                f"Expected split '{split_name}' to contain a list of examples, "
                f"but got {type(examples).__name__}"
            )

        if verbose:
            logger.info(
                f"Converting split '{split_name}' with {len(examples)} examples"
            )

        # Convert to HF Dataset
        dataset = Dataset.from_list(examples)

        # Save as Parquet
        parquet_file = output_path / f"{split_name}.parquet"
        dataset.to_parquet(str(parquet_file))

        if verbose:
            logger.info(f"  Saved to: {parquet_file}")

    if verbose:
        logger.info(
            f"\n✓ Conversion complete!\n\n"
            f"To load this dataset:\n"
            f"  from datasets import load_dataset\n"
            f"  dataset = load_dataset('parquet', data_dir='{output_dir}')\n"
            f"  # Access splits: dataset['train'], dataset['validation'], etc."
        )

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert nested JSON to HuggingFace Dataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON file with nested structure",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the HuggingFace dataset",
    )

    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    convert_json_to_hf_dataset(
        input_json=args.input_json, output_dir=args.output_dir, verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

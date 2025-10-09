#!/usr/bin/env python3
"""
Convert JSON files with 'split' field to HuggingFace Dataset format.

This script takes a JSON file containing examples with a 'split' field
(e.g., 'train', 'validation', 'test') and converts it to a HuggingFace
DatasetDict with proper splits saved to disk.

Usage:
    python convert_json_to_hf_dataset.py \
        --input_json /path/to/data.json \
        --output_dir /path/to/output \
        --split_field split

The output will be a DatasetDict saved to disk that can be loaded with:
    from datasets import load_from_disk
    dataset = load_from_disk('/path/to/output')
    # Access splits: dataset['train'], dataset['validation'], etc.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

from datasets import Dataset, DatasetDict


def convert_json_to_hf_dataset(
    input_json: str,
    output_dir: str,
    split_field: str = "split",
    verbose: bool = True
):
    """
    Convert JSON file with split field to HuggingFace DatasetDict.

    Args:
        input_json: Path to input JSON file
        output_dir: Directory to save the HuggingFace dataset
        split_field: Name of the field containing split information
        verbose: Whether to print progress information
    """
    # Load JSON data
    if verbose:
        print(f"Loading JSON from: {input_json}")

    with open(input_json, 'r') as f:
        data = json.load(f)

    if verbose:
        print(f"Loaded {len(data)} examples")

    # Group examples by split
    split_data = defaultdict(list)
    examples_without_split = []

    for idx, example in enumerate(data):
        split_value = example.get(split_field)
        if split_value is None:
            examples_without_split.append(idx)
        else:
            split_data[split_value].append(example)

    # Warn about examples without split field
    if examples_without_split:
        print(f"WARNING: {len(examples_without_split)} examples missing '{split_field}' field")
        print(f"         These examples will be skipped: {examples_without_split[:10]}...")

    # Create DatasetDict
    dataset_dict = {}

    for split_name, examples in split_data.items():
        if verbose:
            print(f"Creating split '{split_name}' with {len(examples)} examples")

        # Convert to HF Dataset
        dataset_dict[split_name] = Dataset.from_list(examples)

    # Wrap in DatasetDict
    hf_dataset = DatasetDict(dataset_dict)

    # Save to disk
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving dataset to: {output_dir}")

    hf_dataset.save_to_disk(output_dir)

    if verbose:
        print("✓ Conversion complete!")
        print(f"\nDataset info:")
        print(hf_dataset)
        print(f"\nTo load this dataset:")
        print(f"  from datasets import load_from_disk")
        print(f"  dataset = load_from_disk('{output_dir}')")

    return hf_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON with split field to HuggingFace Dataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the HuggingFace dataset"
    )

    parser.add_argument(
        "--split_field",
        type=str,
        default="split",
        help="Name of the field containing split information (default: 'split')"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    convert_json_to_hf_dataset(
        input_json=args.input_json,
        output_dir=args.output_dir,
        split_field=args.split_field,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

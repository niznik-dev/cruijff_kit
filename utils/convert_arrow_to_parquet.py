"""
Convert Arrow format datasets (saved with save_to_disk) to Parquet format.

This script takes a dataset directory that was saved using save_to_disk()
and converts it to Parquet files that can be loaded with load_dataset().

Usage:
    python convert_arrow_to_parquet.py \
        --input_dir /path/to/arrow_dataset \
        --output_dir /path/to/output

The output will be Parquet files (one per split) that can be loaded with:
    from datasets import load_dataset
    dataset = load_dataset('parquet', data_dir='/path/to/output')
    # Access splits: dataset['train'], dataset['validation'], etc.
"""

import argparse
from pathlib import Path

from datasets import load_from_disk

from cruijff_kit.utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)


def convert_arrow_to_parquet(
    input_dir: str,
    output_dir: str,
    verbose: bool = True
):
    """
    Convert Arrow format dataset to Parquet files.

    Args:
        input_dir: Path to Arrow dataset directory (saved with save_to_disk)
        output_dir: Directory to save the Parquet files
        verbose: Whether to print progress information
    """
    # Load Arrow dataset
    if verbose:
        logger.info(f"Loading Arrow dataset from: {input_dir}")

    dataset = load_from_disk(input_dir)

    if verbose:
        logger.info(f"Found splits: {list(dataset.keys())}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save each split as a separate Parquet file
    for split_name, split_data in dataset.items():
        if verbose:
            logger.info(f"Converting split '{split_name}' with {len(split_data)} examples")

        # Save as Parquet
        parquet_file = output_path / f"{split_name}.parquet"
        split_data.to_parquet(str(parquet_file))

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
        description="Convert Arrow format dataset to Parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to Arrow dataset directory (saved with save_to_disk)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the Parquet files"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    convert_arrow_to_parquet(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
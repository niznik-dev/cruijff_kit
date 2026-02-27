#!/usr/bin/env python3
"""
Extract ACS prediction data from HuggingFace.

Downloads folktexts ACS datasets from HuggingFace and converts to
cruijff_kit-compatible JSON format.

Supported tasks:
    - ACSIncome: Predict if income > $50k
    - ACSEmployment: Predict if employed as civilian
    - ACSMobility: Predict if moved in last year
    - ACSPublicCoverage: Predict if has public health insurance
    - ACSTravelTime: Predict if commute > 20 minutes

Usage:
    python extract_acs.py --task ACSIncome --output acs_income.json
    python extract_acs.py --task ACSEmployment --train-size 40000 --val-size 5000 --test-size 5000
    python extract_acs.py --task ACSIncome --balanced  # outputs acs_income_verbose_balanced.json
"""

import json
import argparse
from pathlib import Path

from datasets import load_dataset

# Pin to a specific revision for reproducibility.
# Last modified 2024-11-28; all existing experiments used this revision.
FOLKTEXTS_REVISION = "ad89c177c7cf428152494c90150cce7011a6d960"


# Task configurations: dataset name -> binary question
ACS_TASKS = {
    "ACSIncome": "Is this person's income above $50,000?",
    "ACSEmployment": "Is this person employed as a civilian?",
    "ACSMobility": "Did this person move in the last year?",
    "ACSPublicCoverage": "Does this person have public health insurance?",
    "ACSTravelTime": "Is this person's commute longer than 20 minutes?",
}


def extract_acs(
    task: str,
    output_path: Path,
    train_size: int = 5000,
    val_size: int = 500,
    test_size: int = 500,
    random_seed: int = 42,
    balanced: bool = False,
):
    """
    Extract ACS data from HuggingFace and save to JSON.

    Args:
        task: ACS task name (e.g., "ACSIncome", "ACSEmployment")
        output_path: Path to save JSON output
        train_size: Number of training examples to sample
        val_size: Number of validation examples to sample
        test_size: Number of test examples to sample
        random_seed: Random seed for reproducibility
        balanced: If True, sample equal numbers from each class (50/50 split)
    """
    if task not in ACS_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {list(ACS_TASKS.keys())}")

    binary_question = ACS_TASKS[task]

    print(f"Loading {task} dataset from HuggingFace...")
    print(f"  Subset: {task}")
    print(f"  Revision: {FOLKTEXTS_REVISION}")
    print(f"  Splits: train, validation, test")

    # Load dataset from HuggingFace
    dataset = load_dataset("acruz/folktexts", task, revision=FOLKTEXTS_REVISION)

    print(f"Dataset loaded!")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")
    print(f"  Test: {len(dataset['test'])} examples")

    # Sample from each split
    print(f"\nSampling data (seed={random_seed}, balanced={balanced})...")

    def sample_split(split_data, size):
        """Sample from a split, optionally with class balancing."""
        if not balanced:
            return split_data.shuffle(seed=random_seed).select(range(size))

        # Balanced sampling: equal numbers from each class
        half_size = size // 2
        class_0 = split_data.filter(lambda x: x['label'] == 0).shuffle(seed=random_seed)
        class_1 = split_data.filter(lambda x: x['label'] == 1).shuffle(seed=random_seed)

        # Check we have enough samples
        if len(class_0) < half_size:
            raise ValueError(f"Not enough class 0 samples: need {half_size}, have {len(class_0)}")
        if len(class_1) < half_size:
            raise ValueError(f"Not enough class 1 samples: need {half_size}, have {len(class_1)}")

        # Sample equal from each class and concatenate
        from datasets import concatenate_datasets
        sampled = concatenate_datasets([
            class_0.select(range(half_size)),
            class_1.select(range(half_size))
        ]).shuffle(seed=random_seed)
        return sampled

    train_sample = sample_split(dataset['train'], train_size)
    val_sample = sample_split(dataset['validation'], val_size)
    test_sample = sample_split(dataset['test'], test_size)

    # Convert to cruijff_kit format
    print("Converting to cruijff_kit JSON format...")

    def convert_split(split_data):
        """Convert a dataset split to cruijff_kit format."""
        converted = []
        for example in split_data:
            # Combine instruction, description, and question for full context
            full_input = f"{example['instruction']}\n{example['description']}\n\n{binary_question}"

            converted.append({
                "input": full_input,
                "output": str(example['label'])  # "1" = positive class, "0" = negative class
            })
        return converted

    output_data = {
        "train": convert_split(train_sample),
        "validation": convert_split(val_sample),
        "test": convert_split(test_sample)
    }

    # Save to JSON
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nExtraction complete!")
    print(f"  Task: {task}")
    print(f"  Question: {binary_question}")
    print(f"  Balanced: {balanced}")
    print(f"  Train: {len(output_data['train'])} examples")
    print(f"  Validation: {len(output_data['validation'])} examples")
    print(f"  Test: {len(output_data['test'])} examples")
    print(f"  Total: {len(output_data['train']) + len(output_data['validation']) + len(output_data['test'])} examples")
    print(f"  Output: {output_path}")

    # Print a sample example
    if output_data['train']:
        print(f"\nSample example:")
        print(f"  Input: {output_data['train'][0]['input'][:150]}...")
        print(f"  Output: {output_data['train'][0]['output']}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ACS prediction data from HuggingFace"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(ACS_TASKS.keys()),
        help=f"ACS task to extract. Choices: {list(ACS_TASKS.keys())}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: acs_<task>.json)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=5000,
        help="Number of training examples to sample (default: 5000)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=500,
        help="Number of validation examples to sample (default: 500)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Number of test examples to sample (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Sample equal numbers from each class (50/50 split)",
    )

    args = parser.parse_args()

    # Default output filename based on task
    if args.output is None:
        task_lower = args.task.lower().replace("acs", "acs_")
        suffix = "_balanced" if args.balanced else ""
        args.output = Path(f"{task_lower}_verbose{suffix}.json")

    extract_acs(
        task=args.task,
        output_path=args.output,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_seed=args.seed,
        balanced=args.balanced,
    )


if __name__ == "__main__":
    main()

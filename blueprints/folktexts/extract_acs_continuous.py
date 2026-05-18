#!/usr/bin/env python3
"""
Extract ACS data with continuous target variables from HuggingFace.

Instead of binary labels (0/1), extracts actual numeric values
(e.g., age=42, income=65000) for continuous prediction experiments.

Supported targets:
    - AGEP: Age in years (all tasks)
    - PINCP: Yearly income in dollars (ACSMobility, ACSPublicCoverage)
    - WKHP: Hours worked per week (ACSIncome, ACSMobility)
    - JWMNP: Commute time in minutes (ACSMobility)

Usage:
    python extract_acs_continuous.py --target AGEP
    python extract_acs_continuous.py --target PINCP --source-task ACSMobility
    python extract_acs_continuous.py --target AGEP --train-size 40000 --val-size 5000 --test-size 5000
"""

import json
import re
import argparse
from pathlib import Path
from datasets import load_dataset

FOLKTEXTS_REVISION = "ad89c177c7cf428152494c90150cce7011a6d960"

# target_col -> (friendly name, question, regex to strip from description, default source task)
TARGETS = {
    "AGEP": ("age", "How old is this person? Respond with only a number.", r"- The age is: .+\n?", "ACSMobility"),
    "PINCP": ("income", "What is this person's yearly income in dollars? Respond with only a number.", r"- The yearly income is: .+\n?", "ACSMobility"),
    "WKHP": ("hours_worked", "How many hours per week does this person usually work? Respond with only a number.", r"- The usual number of hours worked per week is: .+\n?", "ACSMobility"),
    "JWMNP": ("commute_time", "What is this person's commute time to work in minutes? Respond with only a number.", r"- The commute time is: .+\n?", "ACSMobility"),
}


def extract_acs_continuous(target_col, output_path, source_task, train_size, val_size, test_size, seed):
    friendly_name, question, strip_pattern, _ = TARGETS[target_col]

    print(f"Target: {target_col} ({friendly_name})")
    print(f"Source: {source_task}")
    print(f"Loading dataset...")

    dataset = load_dataset("acruz/folktexts", source_task, revision=FOLKTEXTS_REVISION)

    def sample_split(split_data, size):
        # Filter NaN values
        filtered = split_data.filter(lambda x: x[target_col] is not None and x[target_col] == x[target_col])
        size = min(size, len(filtered))
        return filtered.shuffle(seed=seed).select(range(size))

    def convert_split(split_data):
        converted = []
        for ex in split_data:
            value = ex[target_col]
            if float(value) == int(value):
                value = int(value)

            # Strip target info from description so model can't just read the answer
            desc = re.sub(strip_pattern, "", ex["description"]).strip()
            desc = re.sub(r"\n\n+", "\n", desc)

            full_input = f"{ex['instruction']}\n{desc}\n\n{question}"
            converted.append({"input": full_input, "output": str(value)})
        return converted

    train = sample_split(dataset["train"], train_size)
    val = sample_split(dataset["validation"], val_size)
    test = sample_split(dataset["test"], test_size)

    output_data = {
        "train": convert_split(train),
        "validation": convert_split(val),
        "test": convert_split(test),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Quick stats
    vals = [float(x["output"]) for x in output_data["train"]]
    print(f"Train: {len(output_data['train'])}, Val: {len(output_data['validation'])}, Test: {len(output_data['test'])}")
    print(f"Target range: [{min(vals):.0f}, {max(vals):.0f}], mean: {sum(vals)/len(vals):.1f}")
    print(f"Saved: {output_path}")
    print(f"\nSample input:\n{output_data['train'][0]['input'][:300]}...")
    print(f"\nSample output: {output_data['train'][0]['output']}")


def main():
    parser = argparse.ArgumentParser(description="Extract ACS data with continuous targets")
    parser.add_argument("--target", type=str, required=True, choices=list(TARGETS.keys()))
    parser.add_argument("--source-task", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.source_task is None:
        args.source_task = TARGETS[args.target][3]
    if args.output is None:
        args.output = Path(f"acs_{TARGETS[args.target][0]}_continuous.json")

    extract_acs_continuous(
        args.target, args.output, args.source_task,
        args.train_size, args.val_size, args.test_size, args.seed,
    )


if __name__ == "__main__":
    main()

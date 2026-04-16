"""Shuffle the order of survey variables independently per training sample.

Unlike shuffle_acs_variables.py (which applies one fixed permutation to all
rows), this script gives each training sample its own random permutation.
Validation and test splits are left in original order so that evaluation
comparisons remain clean.  Works with any ACS task (Income, PublicCoverage,
etc.) regardless of the number of variables.

Usage:
    python experiments/folktexts/per_sample_shuffle_acs_variables.py \
        --input  data/green/acs/acs_income_verbose_50000_80P.json \
        --output data/green/acs/acs_income_verbose_50000_80P_per_sample_shuffle.json \
        --seed 42
"""

import argparse
import json
import random

from shuffle_acs_variables import parse_input, reassemble


def main():
    parser = argparse.ArgumentParser(
        description="Per-sample shuffle of variable order in ACS prompts."
    )
    parser.add_argument("--input", required=True, help="Path to source JSON")
    parser.add_argument("--output", required=True, help="Path to write shuffled JSON")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    rng = random.Random(args.seed)

    result = {}
    for split, samples in data.items():
        if split == "train":
            shuffled_samples = []
            for sample in samples:
                header, variables, question = parse_input(sample["input"])
                perm = list(range(len(variables)))
                rng.shuffle(perm)
                shuffled_variables = [variables[i] for i in perm]
                shuffled_samples.append(
                    {
                        "input": reassemble(header, shuffled_variables, question),
                        "output": sample["output"],
                    }
                )
            result[split] = shuffled_samples
            print(f"  {split}: {len(shuffled_samples)} samples (per-sample shuffled)")
        else:
            # Keep val/test in original order
            result[split] = samples
            print(f"  {split}: {len(samples)} samples (original order)")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSeed: {args.seed}")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()

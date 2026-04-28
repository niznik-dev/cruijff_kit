"""Shuffle the order of survey variables in ACS prompts.

Produces a new dataset where the survey variables appear in a
random (but consistent across all rows) order.  Everything else — the
header paragraph, the closing question, and the output label — stays
identical.  Works with any ACS task (Income, PublicCoverage, etc.)
regardless of the number of variables.

Usage:
    python blueprints/folktexts/modifiers/shuffle_acs_variables.py \
        --input  {ck_data_dir}/folktexts/acs_income_verbose_50000_80P.json \
        --output {ck_data_dir}/folktexts/acs_income_verbose_50000_80P_shuffled_s14.json \
        --seed 14
"""

import argparse
import json
import random
import re


def parse_input(text: str) -> tuple[str, list[str], str]:
    """Split a prompt into (header, variable_lines, question)."""
    lines = text.split("\n")

    variable_indices = [i for i, line in enumerate(lines) if line.startswith("- ")]
    if len(variable_indices) < 2:
        raise ValueError(
            f"Expected at least 2 variable lines, found {len(variable_indices)}:\n{text[:200]}"
        )

    first_variable = variable_indices[0]
    last_variable = variable_indices[-1]

    header = "\n".join(lines[:first_variable])
    variables = [lines[i] for i in variable_indices]
    question = "\n".join(lines[last_variable + 1 :])

    return header, variables, question


def reassemble(header: str, variables: list[str], question: str) -> str:
    """Reassemble the three parts into a single prompt string."""
    return header + "\n" + "\n".join(variables) + "\n" + question


def shuffle_sample(sample: dict, permutation: list[int]) -> dict:
    """Return a copy of the sample with variable lines reordered."""
    header, variables, question = parse_input(sample["input"])
    shuffled_variables = [variables[i] for i in permutation]
    return {
        "input": reassemble(header, shuffled_variables, question),
        "output": sample["output"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle variable order in ACS prompts."
    )
    parser.add_argument("--input", required=True, help="Path to source JSON")
    parser.add_argument("--output", required=True, help="Path to write shuffled JSON")
    parser.add_argument(
        "--seed", type=int, default=14, help="Random seed (default: 14)"
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    # Detect number of variables from first sample
    first_sample = data[next(iter(data))][0]
    _, original_variables, _ = parse_input(first_sample["input"])
    n_vars = len(original_variables)
    print(f"Detected {n_vars} variables")

    # Generate a derangement (no element stays in its original position)
    rng = random.Random(args.seed)
    permutation = list(range(n_vars))
    attempts = 0
    while True:
        rng.shuffle(permutation)
        attempts += 1
        if all(permutation[i] != i for i in range(n_vars)):
            break
    print(f"Found derangement after {attempts} attempt(s)")

    # Log the resulting order
    var_names = []
    for v in original_variables:
        m = re.match(r"- The (.+?) is:", v)
        var_names.append(m.group(1) if m else v[:60])

    print(f"Seed: {args.seed}")
    print(f"Permutation: {permutation}")
    print("Resulting variable order:")
    for i, idx in enumerate(permutation):
        print(f"  {i + 1:2d}. {var_names[idx]}")

    # Shuffle all splits
    shuffled = {}
    for split, samples in data.items():
        shuffled[split] = [shuffle_sample(s, permutation) for s in samples]
        print(f"  {split}: {len(shuffled[split])} samples")

    with open(args.output, "w") as f:
        json.dump(shuffled, f, indent=2)

    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()

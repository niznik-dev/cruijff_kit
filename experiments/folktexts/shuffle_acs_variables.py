"""Shuffle the order of bullet-point variables in ACS Income prompts.

Produces a new dataset where the 10 survey-variable bullets appear in a
random (but consistent across all rows) order.  Everything else — the
header paragraph, the closing question, and the output label — stays
identical.

Usage:
    python experiments/folktexts/shuffle_acs_variables.py \
        --input  data/green/acs/acs_income_verbose_50000_80P.json \
        --output data/green/acs/acs_income_verbose_50000_80P_shuffled_s14.json \
        --seed 14
"""

import argparse
import json
import random
import re


def parse_input(text: str) -> tuple[str, list[str], str]:
    """Split a prompt into (header, bullet_lines, question)."""
    lines = text.split("\n")

    bullet_indices = [i for i, line in enumerate(lines) if line.startswith("- The ")]
    if len(bullet_indices) != 10:
        raise ValueError(
            f"Expected 10 bullet lines, found {len(bullet_indices)}:\n{text[:200]}"
        )

    first_bullet = bullet_indices[0]
    last_bullet = bullet_indices[-1]

    header = "\n".join(lines[:first_bullet])
    bullets = [lines[i] for i in bullet_indices]
    question = "\n".join(lines[last_bullet + 1 :])

    return header, bullets, question


def reassemble(header: str, bullets: list[str], question: str) -> str:
    """Reassemble the three parts into a single prompt string."""
    return header + "\n" + "\n".join(bullets) + "\n" + question


def shuffle_sample(sample: dict, permutation: list[int]) -> dict:
    """Return a copy of the sample with bullet lines reordered."""
    header, bullets, question = parse_input(sample["input"])
    shuffled_bullets = [bullets[i] for i in permutation]
    return {
        "input": reassemble(header, shuffled_bullets, question),
        "output": sample["output"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle variable order in ACS Income prompts."
    )
    parser.add_argument("--input", required=True, help="Path to source JSON")
    parser.add_argument("--output", required=True, help="Path to write shuffled JSON")
    parser.add_argument(
        "--seed", type=int, default=14, help="Random seed (default: 14)"
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    # Generate a derangement (no element stays in its original position)
    rng = random.Random(args.seed)
    permutation = list(range(10))
    attempts = 0
    while True:
        rng.shuffle(permutation)
        attempts += 1
        if all(permutation[i] != i for i in range(10)):
            break
    print(f"Found derangement after {attempts} attempt(s)")

    # Log the resulting order by extracting variable names from the first sample
    first_sample = data[next(iter(data))][0]
    _, original_bullets, _ = parse_input(first_sample["input"])
    var_names = [re.match(r"- The (.+?) is:", b).group(1) for b in original_bullets]

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

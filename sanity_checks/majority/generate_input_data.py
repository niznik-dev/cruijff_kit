"""
Generate majority classification data.

Each sample is a comma-delimited sequence of 0s and 1s.
The label is whichever digit appears more often.
Ties (equal count) are excluded.

Usage:
    python generate_input_data.py
    python generate_input_data.py --min_len 5 --max_len 15 --n_train 2000
"""

import argparse
import json
import random
from pathlib import Path


def generate_sample(seq_len):
    """Generate one sample: random 0/1 sequence, label = majority digit."""
    sequence = [random.choice([0, 1]) for _ in range(seq_len)]
    n_ones = sum(sequence)
    n_zeros = seq_len - n_ones

    # Skip ties
    if n_ones == n_zeros:
        return None

    label = "1" if n_ones > n_zeros else "0"
    margin = abs(n_ones - n_zeros) / seq_len

    input_str = ",".join(str(x) for x in sequence)

    return {
        "input": input_str,
        "output": label,
        "metadata": {
            "seq_len": seq_len,
            "n_ones": n_ones,
            "n_zeros": n_zeros,
            "margin": round(margin, 4),
        },
    }


def generate_samples(n, min_len, max_len):
    """Generate n samples with lengths uniformly drawn from [min_len, max_len]."""
    samples = []
    while len(samples) < n:
        seq_len = random.randint(min_len, max_len)
        sample = generate_sample(seq_len)
        if sample is not None:
            samples.append(sample)
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate majority classification data"
    )
    parser.add_argument("--min_len", type=int, default=5, help="Min sequence length")
    parser.add_argument("--max_len", type=int, default=15, help="Max sequence length")
    parser.add_argument(
        "--ood_max_len", type=int, default=25, help="Max OOD sequence length for eval"
    )
    parser.add_argument(
        "--n_train", type=int, default=2000, help="Number of training samples"
    )
    parser.add_argument(
        "--n_val", type=int, default=500, help="Number of validation samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: data/green/majority/)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    if args.output_dir is None:
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent.parent
        args.output_dir = repo_root / "data" / "green" / "majority"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Training: lengths min_len to max_len
    train = generate_samples(args.n_train, args.min_len, args.max_len)

    # Validation: same range as training
    val = generate_samples(args.n_val, args.min_len, args.max_len)

    # OOD validation: lengths max_len+1 to ood_max_len
    val_ood = generate_samples(args.n_val, args.max_len + 1, args.ood_max_len)

    for name, data in [("train", train), ("val", val), ("val_ood", val_ood)]:
        n_ones = sum(1 for s in data if s["output"] == "1")
        margins = [s["metadata"]["margin"] for s in data]
        lens = [s["metadata"]["seq_len"] for s in data]
        print(
            f"{name}: {len(data)} samples ({n_ones} majority-1, {len(data) - n_ones} majority-0)"
        )
        print(f"  Length range: [{min(lens)}, {max(lens)}]")
        print(f"  Margin range: [{min(margins):.3f}, {max(margins):.3f}]")

    filename = f"majority_len{args.min_len}-{args.max_len}.json"
    output_path = args.output_dir / filename
    output = {"train": train, "validation": val, "validation_ood": val_ood}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

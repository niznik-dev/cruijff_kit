"""
Generate count_digits classification data.

Each sample is a comma-delimited sequence of single digits (0-9).
The label is the count of digits in the sequence, as a string.

Three splits are produced:
  - train:          seq_len uniform in [min_len, max_len]
  - validation:     same length range as train (in-distribution)
  - validation_ood: seq_len uniform in [max_len + 1, ood_max_len]

Usage:
    python generate_input_data.py
    python generate_input_data.py --min_len 3 --max_len 6 --ood_max_len 10
"""

import argparse
import json
import random
from pathlib import Path


def generate_sample(seq_len):
    """Generate one sample: random digit sequence, label = count as string."""
    sequence = [random.randint(0, 9) for _ in range(seq_len)]
    input_str = ",".join(str(x) for x in sequence)
    return {
        "input": input_str,
        "output": str(seq_len),
        "metadata": {
            "seq_len": seq_len,
        },
    }


def generate_samples(n, min_len, max_len):
    """Generate n samples with lengths uniformly drawn from [min_len, max_len]."""
    return [generate_sample(random.randint(min_len, max_len)) for _ in range(n)]


def main():
    parser = argparse.ArgumentParser(
        description="Generate count_digits classification data"
    )
    parser.add_argument("--min_len", type=int, default=3, help="Min sequence length")
    parser.add_argument("--max_len", type=int, default=6, help="Max sequence length")
    parser.add_argument(
        "--ood_max_len", type=int, default=10, help="Max OOD sequence length for eval"
    )
    parser.add_argument(
        "--n_train", type=int, default=1000, help="Number of training samples"
    )
    parser.add_argument(
        "--n_val", type=int, default=400, help="Number of validation samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: data/green/count_digits/)",
    )
    args = parser.parse_args()

    if args.min_len < 1:
        raise ValueError("min_len must be >= 1")
    if args.max_len < args.min_len:
        raise ValueError("max_len must be >= min_len")
    if args.ood_max_len <= args.max_len:
        raise ValueError("ood_max_len must be > max_len")

    random.seed(args.seed)

    if args.output_dir is None:
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent.parent
        args.output_dir = repo_root / "data" / "green" / "count_digits"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = generate_samples(args.n_train, args.min_len, args.max_len)
    val = generate_samples(args.n_val, args.min_len, args.max_len)
    val_ood = generate_samples(args.n_val, args.max_len + 1, args.ood_max_len)

    for name, data in [("train", train), ("val", val), ("val_ood", val_ood)]:
        lens = [s["metadata"]["seq_len"] for s in data]
        outputs = [int(s["output"]) for s in data]
        print(f"{name}: {len(data)} samples")
        print(f"  Length range: [{min(lens)}, {max(lens)}]")
        print(f"  Output range: [{min(outputs)}, {max(outputs)}]")

    filename = f"count_digits_len{args.min_len}-{args.max_len}.json"
    output_path = args.output_dir / filename
    output = {"train": train, "validation": val, "validation_ood": val_ood}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

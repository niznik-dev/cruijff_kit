"""
Generate Bernoulli classification data for calibration testing.

Two coins with different bias:
  Coin A: P(heads=1) = p_a (default 0.7)
  Coin B: P(heads=1) = p_b (default 0.3)

Each sample is a sequence of flips from one coin. The model must classify
which coin generated the sequence. Ground-truth Bayesian posterior is stored
in metadata for calibration analysis.

Usage:
    python generate_input_data.py
    python generate_input_data.py --p_a 0.6 --p_b 0.4 --seq_len 15
"""

import argparse
import json
import math
import random
from pathlib import Path


def bayesian_posterior_a(sequence, p_a, p_b):
    """Compute P(Coin A | sequence) assuming equal priors."""
    k = sum(sequence)
    n = len(sequence)
    # Log-likelihood to avoid underflow
    ll_a = k * math.log(p_a) + (n - k) * math.log(1 - p_a)
    ll_b = k * math.log(p_b) + (n - k) * math.log(1 - p_b)
    # Numerically stable softmax
    max_ll = max(ll_a, ll_b)
    p_a_posterior = math.exp(ll_a - max_ll) / (
        math.exp(ll_a - max_ll) + math.exp(ll_b - max_ll)
    )
    return p_a_posterior


def generate_sample(p_a, p_b, seq_len):
    """Generate one sample: pick a coin, flip it, compute ground truth."""
    coin = random.choice(["A", "B"])
    p = p_a if coin == "A" else p_b
    sequence = [1 if random.random() < p else 0 for _ in range(seq_len)]

    posterior_a = bayesian_posterior_a(sequence, p_a, p_b)

    input_str = " ".join(str(x) for x in sequence)

    return {
        "input": input_str,
        "output": coin,
        "metadata": {
            "p_a": p_a,
            "p_b": p_b,
            "seq_len": seq_len,
            "num_ones": sum(sequence),
            "bayesian_posterior_a": round(posterior_a, 6),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Bernoulli classification data"
    )
    parser.add_argument("--p_a", type=float, default=0.7, help="P(1) for Coin A")
    parser.add_argument("--p_b", type=float, default=0.3, help="P(1) for Coin B")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length")
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
        help="Output directory (default: data/green/bernoulli/)",
    )
    args = parser.parse_args()

    if not (0 < args.p_a < 1) or not (0 < args.p_b < 1):
        raise ValueError("Probabilities must be between 0 and 1 (exclusive)")
    if args.p_a == args.p_b:
        raise ValueError("p_a and p_b must differ (otherwise the task is impossible)")

    random.seed(args.seed)

    if args.output_dir is None:
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent.parent
        args.output_dir = repo_root / "data" / "green" / "bernoulli"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = [
        generate_sample(args.p_a, args.p_b, args.seq_len) for _ in range(args.n_train)
    ]
    val = [generate_sample(args.p_a, args.p_b, args.seq_len) for _ in range(args.n_val)]

    # Summary stats
    for name, data in [("train", train), ("val", val)]:
        n_a = sum(1 for s in data if s["output"] == "A")
        n_b = len(data) - n_a
        posteriors = [s["metadata"]["bayesian_posterior_a"] for s in data]
        print(f"{name}: {len(data)} samples ({n_a} A, {n_b} B)")
        print(f"  Posterior P(A) range: [{min(posteriors):.4f}, {max(posteriors):.4f}]")
        print(f"  Posterior P(A) mean:  {sum(posteriors) / len(posteriors):.4f}")

    filename = f"bernoulli_p{args.p_a}_vs_p{args.p_b}_len{args.seq_len}.json"
    output_path = args.output_dir / filename
    output = {"train": train, "validation": val}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

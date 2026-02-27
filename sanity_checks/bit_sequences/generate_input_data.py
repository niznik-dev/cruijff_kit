#!/usr/bin/env python3
"""
Generate binary-sequence datasets with parity-based or probabilistic labelling.

Outputs a single JSON file with nested 'train' and 'validation' keys.

Note: Deduplication ensures no sequence string appears in both train and validation.
Use bit_length such that 2^bit_length >> val_size to avoid removing all training data.

Examples
--------
Parity dataset, ~30K samples of length 15, noiseless, 4 000-example validation set
    python generate_input_data.py --bit_length 15 --N 33000 --p 0 --bit_parity True --val_size 4000 --output parity.json

Probabilistic dataset, p = 0.5, ~30K samples of length 15, 4 000-example validation set
    python generate_input_data.py --bit_length 15 --N 33000 --p 0.5 --bit_parity False --val_size 4000 --output prob_p05.json
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import List, Dict

import numpy as np

from cruijff_kit.utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def compute_parity(bitstr: str) -> str:
    """Return '0' if the bit-string has an even number of “1”s, else '1'."""
    return str(bitstr.count("1") % 2)


def parse_bool(value: str) -> bool:
    """Parse booleans passed on the command line."""
    val = value.lower()
    if val in {"true", "t", "1"}:
        return True
    if val in {"false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate binary-sequence datasets (parity or Bernoulli labelling)."
    )
    p.add_argument(
        "--bit_length", type=int, required=True, help="Sequence length n ≥ 1"
    )
    p.add_argument(
        "--N", type=int, required=True, help="Number of samples to generate (N ≥ 1)"
    )
    p.add_argument(
        "--val_size",
        type=int,
        required=True,
        help="Exact number of examples to put in the validation set",
    )
    p.add_argument(
        "--bit_parity",
        type=parse_bool,
        required=True,
        choices=[True, False],
        help="True → parity labels (with optional noise p); False → Bernoulli(p) labels",
    )
    p.add_argument(
        "--p",
        type=float,
        default=0.0,
        help=(
            "If --bit_parity True: probability of *flipping* the true parity label.\n"
            "If False: probability of label 1 in Bernoulli labelling."
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output filename (e.g., parity.json, prob_p05.json)",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: data/green/bit_sequences/)",
    )
    return p.parse_args()


def label_sequences(
    sequences: List[str], bit_parity: bool, p: float, rng: np.random.RandomState
) -> List[Dict[str, str]]:
    """Attach labels to every sequence and return instruction/input/output dicts."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Probability p must be in [0, 1], got {p}")

    labelled = []
    for seq in sequences:
        if bit_parity:
            true_lbl = int(compute_parity(seq))
            lbl = true_lbl ^ int(rng.rand() < p)  # flip with prob p
        else:
            lbl = int(rng.rand() < p)
        labelled.append({"instruction": "", "input": seq, "output": str(lbl)})
    return labelled


# -----------------------------------------------------------------------------#
# Script entry point
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    args = parse_args()

    n = args.bit_length
    N = args.N
    val_size = args.val_size

    assert n > 0, "--bit_length must be ≥ 1"
    assert 0 < val_size < N, "--val_size must be in (0, N)"

    # Set default output directory relative to repository root
    if args.output_dir is None:
        # Find repository root (assuming script is in sanity_checks/bit_sequences/)
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent.parent
        args.output_dir = repo_root / "data" / "green" / "bit_sequences"

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    # ---------------------------------------------------------------------#
    # 1. Generate the population and draw N samples WITH replacement
    # ---------------------------------------------------------------------#
    universe = ["".join(bits) for bits in itertools.product("01", repeat=n)]
    sampled_seqs = rng.choice(universe, size=N, replace=True)

    # ---------------------------------------------------------------------#
    # 2. Pick EXACTLY `val_size` sample indices for the validation split
    #    (no replacement → indices are unique)
    # ---------------------------------------------------------------------#
    val_indices = rng.choice(N, size=val_size, replace=False)
    val_mask = np.zeros(N, dtype=bool)
    val_mask[val_indices] = True

    # ---------------------------------------------------------------------#
    # 3. Ensure the TRAIN split contains *no* sequence that appears in VALIDATION
    #    • Build a set of the sequences found in the validation subset.
    #    • Keep only those training rows whose sequence is NOT in that set.
    # ---------------------------------------------------------------------#
    val_seqs = sampled_seqs[val_mask].tolist()
    val_seq_set = set(val_seqs)

    train_seqs = [seq for seq in sampled_seqs[~val_mask] if seq not in val_seq_set]

    # ---------------------------------------------------------------------#
    # 4. strict separation at the sequence level
    # ---------------------------------------------------------------------#
    assert val_seq_set.isdisjoint(train_seqs), (
        "Leakage: sequence appears in both splits"
    )

    # ---------------------------------------------------------------------#
    # 5. Label and create nested structure
    # ---------------------------------------------------------------------#
    train_data = label_sequences(train_seqs, args.bit_parity, args.p, rng)
    val_data = label_sequences(val_seqs, args.bit_parity, args.p, rng)

    # Create single JSON file with train/validation splits as top-level keys
    output = {"train": train_data, "validation": val_data}

    # Write to specified output file in output directory
    output_path = args.output_dir / args.output
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        f"✅  Generated {output_path}: {len(train_data)} train, {len(val_data)} validation "
        f"(bit_parity={args.bit_parity}, p={args.p})"
    )

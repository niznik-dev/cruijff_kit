"""Generate model-organism datasets (inputs x rules x formats x designs).

Chunk 1a supports only ``design="memorization"``. Additional rules, input
types, formats, and designs land in later chunks.

CLI::

    python -m sanity_checks.model_organisms.generate \\
        --input_type bits --rule parity --k 8 \\
        --N 100 --seed 1729 --design memorization \\
        --output bits_parity_k8_memo.json
"""

import argparse
import json
import random
from pathlib import Path

from .formats import get_format
from .inputs import get_input
from .rules import get_rule


SUPPORTED_DESIGNS = {"memorization"}


def _sample_sequences(
    alphabet: tuple[str, ...], k: int, N: int, rng: random.Random
) -> list[tuple[str, ...]]:
    return [tuple(rng.choices(alphabet, k=k)) for _ in range(N)]


def _build_rows(sequences, rule_fn, formatter) -> list[dict]:
    return [
        {"input": formatter(seq), "output": rule_fn(seq), "metadata": {}}
        for seq in sequences
    ]


def generate(
    *,
    input_type: str,
    rule: str,
    k: int,
    N: int,
    seed: int,
    design: str,
    fmt: str = "spaced",
) -> dict:
    """Generate a model-organism dataset and return it as a dict.

    For ``design="memorization"``, train and validation contain the same rows.
    """
    if design not in SUPPORTED_DESIGNS:
        raise NotImplementedError(
            f"design={design!r} not supported in chunk 1a. "
            f"Supported: {sorted(SUPPORTED_DESIGNS)}"
        )

    input_def = get_input(input_type)
    rule_def = get_rule(rule)
    formatter = get_format(fmt)

    if input_def.name not in rule_def.applicable:
        raise ValueError(
            f"Rule {rule!r} is not applicable to input type {input_type!r}. "
            f"Applicable: {sorted(rule_def.applicable)}"
        )

    rng = random.Random(seed)
    sequences = _sample_sequences(input_def.alphabet, k, N, rng)
    rows = _build_rows(sequences, rule_def.fn, formatter)

    metadata = {
        "generator": "sanity_checks/model_organisms/generate.py",
        "input_type": input_type,
        "rule": rule,
        "k": k,
        "format": fmt,
        "N": N,
        "seed": seed,
        "design": design,
    }

    return {"train": rows, "validation": rows, "metadata": metadata}


def _default_output_dir() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    return repo_root / "data" / "green" / "model_organisms"


def main():
    parser = argparse.ArgumentParser(description="Generate a model-organism dataset.")
    parser.add_argument("--input_type", required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--k", type=int, required=True, help="Sequence length")
    parser.add_argument("--N", type=int, required=True, help="Number of samples")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--design", required=True, choices=sorted(SUPPORTED_DESIGNS))
    parser.add_argument("--format", dest="fmt", default="spaced")
    parser.add_argument(
        "--output",
        required=True,
        help="Output filename (written under data/green/model_organisms/).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Override output directory (default: data/green/model_organisms/).",
    )
    args = parser.parse_args()

    dataset = generate(
        input_type=args.input_type,
        rule=args.rule,
        k=args.k,
        N=args.N,
        seed=args.seed,
        design=args.design,
        fmt=args.fmt,
    )

    output_dir = (
        args.output_dir if args.output_dir is not None else _default_output_dir()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(
        f"Wrote {len(dataset['train'])} train / "
        f"{len(dataset['validation'])} val rows to {output_path}"
    )


if __name__ == "__main__":
    main()

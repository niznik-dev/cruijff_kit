"""Generate model-organism datasets (inputs x rules x formats x designs).

Current breadth:
- Input types: ``bits``, ``digits``, ``letters``
- Formats: ``spaced``, ``dense``, ``comma``, ``tab``, ``pipe``
- Rules: universal (``first``, ``last``, ``nth``, ``length``, ``constant``,
  ``coin``) plus domain-specific (``parity`` / ``majority`` on bits,
  ``min`` / ``max`` on digits / letters)
- Designs: ``memorization``, ``in_distribution``, ``ood``

A unified inspect task and the ``design-experiment`` integration land in
later chunks.

CLI::

    python -m cruijff_kit.sanity_checks.model_organisms.generate \\
        --input_type bits --rule parity --k 8 \\
        --N 100 --seed 1729 --design memorization \\
        --output bits_parity_k8_memo.json

    python -m cruijff_kit.sanity_checks.model_organisms.generate \\
        --input_type bits --rule coin --k 8 \\
        --rule_kwargs '{"p": 0.7}' \\
        --N 200 --seed 1729 --design in_distribution --split 0.8 \\
        --output bits_coin_p70_k8_indist.json

    python -m cruijff_kit.sanity_checks.model_organisms.generate \\
        --input_type bits --rule parity --k 8 \\
        --N 200 --seed 1729 --design ood --split 0.8 \\
        --ood_tests '[{"k": 12, "N": 50}, {"k": 16, "N": 50}]' \\
        --output bits_parity_k8_ood.json
"""

import argparse
import itertools
import json
import random
from pathlib import Path

from .formats import get_format
from .inputs import get_input
from .rules import get_rule


SUPPORTED_DESIGNS = {"memorization", "in_distribution", "ood"}

# If the full sequence space fits under this size, enumerate it and shuffle
# rather than sampling-with-replacement-and-dedup. 2**20 ~ 1M sequences;
# above that, enumeration costs more than the dedup loop.
ENUMERATION_CEILING = 1 << 20


def _sample_sequences_unique(
    alphabet: tuple[str, ...], k: int, N: int, rng: random.Random
) -> list[tuple[str, ...]]:
    """Return N distinct sequences of length k drawn from ``alphabet``."""
    space_size = len(alphabet) ** k
    if N > space_size:
        raise ValueError(
            f"Cannot sample {N} unique sequences of length {k} from an "
            f"alphabet of size {len(alphabet)} (space size is {space_size})."
        )
    if space_size <= ENUMERATION_CEILING:
        population = list(itertools.product(alphabet, repeat=k))
        rng.shuffle(population)
        return population[:N]

    seen: set[tuple[str, ...]] = set()
    out: list[tuple[str, ...]] = []
    while len(out) < N:
        cand = tuple(rng.choices(alphabet, k=k))
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _build_rows(sequences, rule_fn, rule_kwargs, formatter) -> list[dict]:
    return [
        {
            "input": formatter(seq),
            "output": rule_fn(seq, **rule_kwargs),
            "metadata": {},
        }
        for seq in sequences
    ]


def _split_indices(N: int, split: float) -> tuple[int, int]:
    if not 0 < split < 1:
        raise ValueError(f"split must be in (0, 1); got {split}")
    n_train = int(round(N * split))
    if n_train == 0 or n_train == N:
        raise ValueError(
            f"split={split} with N={N} yields an empty train or validation "
            "set; pick a split that keeps both sides non-empty."
        )
    return n_train, N - n_train


def _resolve_ood_spec(spec: dict, base: dict) -> dict:
    """Merge an ood_tests entry with defaults from the primary config."""
    if "N" not in spec:
        raise ValueError(f"ood_tests entry missing required field 'N': {spec}")
    resolved = {
        "input_type": spec.get("input_type", base["input_type"]),
        "k": spec.get("k", base["k"]),
        "N": spec["N"],
        "format": spec.get("format", base["format"]),
        "rule_kwargs": {**base["rule_kwargs"], **spec.get("rule_kwargs", {})},
    }
    return resolved


def generate(
    *,
    input_type: str,
    rule: str,
    k: int,
    N: int,
    seed: int,
    design: str,
    fmt: str = "spaced",
    rule_kwargs: dict | None = None,
    split: float = 0.8,
    ood_tests: list[dict] | None = None,
) -> dict:
    """Generate a model-organism dataset and return it as a dict.

    - ``memorization``: train and validation contain the same rows.
    - ``in_distribution``: N unique sequences sampled from one distribution
      and split ``split`` / ``1 - split`` into train and validation.
    - ``ood``: primary train/val (as in_distribution) plus one
      ``validation_ood_{i}`` per entry in ``ood_tests``. Each OOD entry
      may override ``input_type``, ``k``, ``format``, and ``rule_kwargs``
      (``N`` is required).
    """
    if design not in SUPPORTED_DESIGNS:
        raise NotImplementedError(
            f"design={design!r} not yet supported. "
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

    rule_kwargs = dict(rule_kwargs or {})
    # Seed is always available to rules (coin needs it for determinism);
    # explicit user-supplied seed in rule_kwargs wins so a caller can
    # decouple the label seed from the sampling seed if they want.
    rule_kwargs.setdefault("seed", seed)

    primary_rng = random.Random(seed)
    sequences = _sample_sequences_unique(input_def.alphabet, k, N, primary_rng)

    metadata = {
        "generator": "sanity_checks/model_organisms/generate.py",
        "input_type": input_type,
        "rule": rule,
        "rule_kwargs": {key: val for key, val in rule_kwargs.items() if key != "seed"},
        "k": k,
        "format": fmt,
        "N": N,
        "seed": seed,
        "design": design,
    }

    if design == "memorization":
        rows = _build_rows(sequences, rule_def.fn, rule_kwargs, formatter)
        return {"train": rows, "validation": rows, "metadata": metadata}

    # Both in_distribution and ood start with a primary train/val split.
    n_train, _ = _split_indices(N, split)
    metadata["split"] = split
    train_rows = _build_rows(sequences[:n_train], rule_def.fn, rule_kwargs, formatter)
    val_rows = _build_rows(sequences[n_train:], rule_def.fn, rule_kwargs, formatter)
    result: dict = {"train": train_rows, "validation": val_rows}

    if design == "in_distribution":
        result["metadata"] = metadata
        return result

    # design == "ood"
    if not ood_tests:
        raise ValueError("design='ood' requires a non-empty ood_tests list")

    base = {
        "input_type": input_type,
        "k": k,
        "format": fmt,
        "rule_kwargs": {key: val for key, val in rule_kwargs.items() if key != "seed"},
    }
    resolved_specs = []
    for i, spec in enumerate(ood_tests):
        r = _resolve_ood_spec(spec, base)
        ood_input_def = get_input(r["input_type"])
        if ood_input_def.name not in rule_def.applicable:
            raise ValueError(
                f"ood_tests[{i}]: rule {rule!r} is not applicable to "
                f"input type {r['input_type']!r}. "
                f"Applicable: {sorted(rule_def.applicable)}"
            )
        ood_formatter = get_format(r["format"])
        ood_rule_kwargs = dict(r["rule_kwargs"])
        ood_rule_kwargs.setdefault("seed", seed)
        # Per-index RNG keeps samples reproducible even if another OOD spec
        # is added to the list later.
        ood_rng = random.Random(f"{seed}.ood.{i}")
        ood_seqs = _sample_sequences_unique(
            ood_input_def.alphabet, r["k"], r["N"], ood_rng
        )
        ood_rows = _build_rows(ood_seqs, rule_def.fn, ood_rule_kwargs, ood_formatter)
        result[f"validation_ood_{i}"] = ood_rows
        resolved_specs.append(r)

    metadata["ood_tests"] = resolved_specs
    result["metadata"] = metadata
    return result


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
        "--rule_kwargs",
        default="{}",
        help="JSON dict of extra rule parameters, e.g. '{\"p\": 0.7}' or "
        '\'{"v": "A"}\'. Default: "{}".',
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train fraction for design=in_distribution or design=ood "
        "(default: 0.8). Ignored for design=memorization.",
    )
    parser.add_argument(
        "--ood_tests",
        default="[]",
        help="JSON list of OOD specs for design=ood. Each entry needs 'N' "
        "and may override 'input_type', 'k', 'format', 'rule_kwargs'. "
        'Example: \'[{"k": 12, "N": 50}, {"k": 16, "N": 50}]\'.',
    )
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

    try:
        parsed_rule_kwargs = json.loads(args.rule_kwargs)
    except json.JSONDecodeError as exc:
        parser.error(f"--rule_kwargs must be valid JSON: {exc}")
    if not isinstance(parsed_rule_kwargs, dict):
        parser.error("--rule_kwargs must be a JSON object (dict)")

    try:
        parsed_ood_tests = json.loads(args.ood_tests)
    except json.JSONDecodeError as exc:
        parser.error(f"--ood_tests must be valid JSON: {exc}")
    if not isinstance(parsed_ood_tests, list):
        parser.error("--ood_tests must be a JSON list")

    dataset = generate(
        input_type=args.input_type,
        rule=args.rule,
        k=args.k,
        N=args.N,
        seed=args.seed,
        design=args.design,
        fmt=args.fmt,
        rule_kwargs=parsed_rule_kwargs,
        split=args.split,
        ood_tests=parsed_ood_tests or None,
    )

    output_dir = (
        args.output_dir if args.output_dir is not None else _default_output_dir()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    ood_counts = ", ".join(
        f"{key}={len(dataset[key])}"
        for key in sorted(dataset)
        if key.startswith("validation_ood_")
    )
    summary = f"Wrote {len(dataset['train'])} train / {len(dataset['validation'])} val"
    if ood_counts:
        summary += f" / {ood_counts}"
    summary += f" rows to {output_path}"
    print(summary)


if __name__ == "__main__":
    main()

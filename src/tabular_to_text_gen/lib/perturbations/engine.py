"""Perturbation compose engine.

Parses a perturbation config (list of type names) into a chain of
perturbation functions and applies them sequentially.

Each perturbation function receives a seeded random.Random instance
derived deterministically from the global seed, perturbation name,
and row index — ensuring reproducibility and independence.
"""

import random
from typing import Callable

from ..segments import Segment
from .clause_addition import clause_addition_perturbation
from .reorder import reorder_perturbation
from .shorthand import shorthand_perturbation
from .synonym import synonym_perturbation

PERTURBATION_REGISTRY: dict[str, Callable] = {
    "synonym": synonym_perturbation,
    "shorthand": shorthand_perturbation,
    "reorder": reorder_perturbation,
    "clause_addition": clause_addition_perturbation,
}


def _make_row_rng(
    seed: int | None, perturbation_name: str, row_index: int
) -> random.Random:
    """Create a deterministic Random instance for a specific
    perturbation on a specific row.

    Different perturbation types get independent random streams.
    Different rows get independent random streams.
    """
    if seed is None:
        return random.Random()
    combined = f"{seed}:{perturbation_name}:{row_index}"
    return random.Random(combined)


def build_perturbation_chain(
    perturbation_names: list[str],
    seed: int | None = None,
) -> Callable[[list[Segment], int], list[Segment]]:
    """Build a composed perturbation function from a list of names.

    Args:
        perturbation_names: e.g., ["synonym", "reorder"]
        seed: Optional seed for reproducible perturbations

    Returns:
        A function that takes (list[Segment], row_index) and returns
        list[Segment].
    """
    # Validate names
    unknown = [n for n in perturbation_names if n not in PERTURBATION_REGISTRY]
    if unknown:
        available = ", ".join(sorted(PERTURBATION_REGISTRY.keys()))
        raise ValueError(f"Unknown perturbation(s): {unknown}. Available: {available}")

    funcs = [(name, PERTURBATION_REGISTRY[name]) for name in perturbation_names]

    def chain(segments: list[Segment], row_index: int) -> list[Segment]:
        result = segments
        for name, func in funcs:
            rng = _make_row_rng(seed, name, row_index)
            result = func(result, rng)
        return result

    return chain


def apply_perturbations(
    segments: list[Segment],
    chain: Callable[[list[Segment], int], list[Segment]],
    row_index: int,
) -> list[Segment]:
    """Apply a perturbation chain to a segment list."""
    return chain(segments, row_index)

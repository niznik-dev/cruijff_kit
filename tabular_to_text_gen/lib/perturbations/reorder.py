"""Segment reordering perturbation.

Randomly shuffles the order of Segments in the list. For dictionary
templates this shuffles feature order; for narrative templates this
shuffles sentence order.
"""

import random

from ..segments import Segment


def reorder_perturbation(
    segments: list[Segment],
    rng: random.Random,
) -> list[Segment]:
    """Randomly shuffle segment order."""
    result = list(segments)
    rng.shuffle(result)
    return result

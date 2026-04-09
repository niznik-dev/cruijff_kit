"""Shorthand substitution perturbation.

Replaces full-form values with abbreviated forms (or vice versa)
using the column's shorthand_map.
"""

import re
import random

from ..segments import Segment


def shorthand_perturbation(
    segments: list[Segment],
    rng: random.Random,
    direction: str = "full_to_short",
) -> list[Segment]:
    """Replace values using the shorthand map.

    Args:
        segments: Input segments.
        rng: Random instance (unused here but kept for consistent interface).
        direction: "full_to_short" (default) or "short_to_full".
    """
    result = []
    for seg in segments:
        shorthand_map = seg.metadata.get("shorthand_map", {})
        if not shorthand_map:
            result.append(seg)
            continue

        if direction == "full_to_short":
            lookup = shorthand_map
        else:
            # Invert the map: abbreviated -> full
            lookup = {v: k for k, v in shorthand_map.items()}

        new_value = lookup.get(seg.value, seg.value)
        if new_value == seg.value:
            result.append(seg)
            continue

        # Replace the old value in the existing text,
        # preserving whatever format the template produced.
        new_text = re.sub(re.escape(seg.value), new_value, seg.text, count=1)
        result.append(
            Segment(
                field=seg.field,
                display_name=seg.display_name,
                value=new_value,
                text=new_text,
                metadata=seg.metadata,
                is_added=seg.is_added,
            )
        )
    return result

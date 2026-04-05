"""Synonym substitution perturbation.

For each Segment, randomly selects an alternative display name from
the column's synonyms list and re-renders the segment text.
"""

import random

from ..segments import Segment


def synonym_perturbation(
    segments: list[Segment],
    rng: random.Random,
) -> list[Segment]:
    """Swap display names with random synonyms.

    Segments with no synonyms (or only one entry) are left unchanged.
    """
    result = []
    for seg in segments:
        synonyms = seg.metadata.get("synonyms", [])
        if len(synonyms) < 2:
            result.append(seg)
            continue

        new_name = rng.choice(synonyms)
        # Re-render text with the new display name
        new_text = f"The {new_name} is: {seg.value}."

        result.append(
            Segment(
                field=seg.field,
                display_name=new_name,
                value=seg.value,
                text=new_text,
                metadata=seg.metadata,
                is_added=seg.is_added,
            )
        )
    return result

"""Redundant restatement insertion perturbation.

Selects random features, picks a restatement template from the
column's restatements list, fills it with the row's value, and
inserts the new Segment at a random position.
"""

import random

from ..segments import Segment


def clause_addition_perturbation(
    segments: list[Segment],
    rng: random.Random,
    n_clauses: int = 1,
) -> list[Segment]:
    """Insert redundant restatement clauses.

    Args:
        segments: Input segments.
        rng: Seeded Random instance.
        n_clauses: Number of clauses to insert.
    """
    # Only features with non-empty restatements are eligible
    eligible = [s for s in segments if s.metadata.get("restatements")]
    if not eligible:
        return list(segments)

    result = list(segments)
    for _ in range(n_clauses):
        source_seg = rng.choice(eligible)
        restatement_template = rng.choice(source_seg.metadata["restatements"])

        # Fill placeholders
        text = _fill_restatement(restatement_template, source_seg)

        new_seg = Segment(
            field=source_seg.field,
            display_name=source_seg.display_name,
            value=source_seg.value,
            text=text,
            metadata=source_seg.metadata,
            is_added=True,
        )

        # Insert at a random position
        insert_pos = rng.randint(0, len(result))
        result.insert(insert_pos, new_seg)

    return result


def _fill_restatement(template: str, segment: Segment) -> str:
    """Fill restatement template placeholders.

    Supported placeholders:
        {value}  - the raw value
        {decade} - for numeric values, the decade (e.g., 51 -> "50")
    """
    text = template.replace("{value}", segment.value)

    if "{decade}" in text:
        try:
            # Extract numeric part (strip units)
            numeric_str = segment.value.split()[0]
            decade = str((int(float(numeric_str)) // 10) * 10)
            text = text.replace("{decade}", decade)
        except (ValueError, IndexError):
            text = text.replace("{decade}", segment.value)

    return text

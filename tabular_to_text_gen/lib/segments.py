"""Segment dataclass and rendering utilities.

A Segment is the intermediate representation between templates and
perturbations. Templates produce a list[Segment]; perturbations
transform list[Segment] -> list[Segment]; the final render joins
segment text with the appropriate delimiter.
"""

from dataclasses import dataclass, field


@dataclass
class Segment:
    """A single unit of text representing one feature or clause.

    Attributes:
        field: Schema column key, e.g., "AGEP"
        display_name: Current display name, e.g., "age" (may be perturbed)
        value: Current value string, e.g., "51 years old" (may be perturbed)
        text: Rendered text, e.g., "The age is: 51 years old."
        metadata: Schema info for perturbation lookups (synonyms,
            shorthand_map, restatements, type, unit)
        is_added: True if this segment was inserted by clause_addition
    """

    field: str
    display_name: str
    value: str
    text: str
    metadata: dict = field(default_factory=dict)
    is_added: bool = False


def render_segments(
    segments: list[Segment],
    template_type: str,
    separator: str | None = None,
) -> str:
    """Render segments into final text.

    For 'dictionary' template: joins with newline, prefixed by '- '
      e.g., "- The age is: 51 years old.\\n- The state is: New York."

    For 'narrative' or 'llm_narrative' template: joins with space
      e.g., "The respondent is 51 years old. They live in New York."

    Custom separator overrides default behavior.
    """
    if separator is not None:
        return separator.join(seg.text for seg in segments)

    if template_type == "dictionary":
        return "\n".join(f"- {seg.text}" for seg in segments)
    else:
        # narrative and llm_narrative: join with space
        return " ".join(seg.text for seg in segments)

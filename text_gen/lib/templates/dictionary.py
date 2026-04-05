"""Dictionary (key-value) template — folktexts-style bulleted list.

Produces output like:
  - The age is: 51 years old.
  - The class of worker is: Working for a for-profit private company.
  - The highest educational attainment is: Bachelor's degree.
"""

from ..schema import ColumnSchema, Schema
from ..segments import Segment
from .base import BaseTemplate


class DictionaryTemplate(BaseTemplate):
    """Key-value bulleted list template."""

    @property
    def template_type(self) -> str:
        return "dictionary"

    def render_row(
        self,
        features: list[tuple[ColumnSchema, str]],
        schema: Schema,
    ) -> list[Segment]:
        segments = []
        for col_schema, raw_value in features:
            display_name = col_schema.display_name
            value = self._format_value(raw_value, col_schema)
            text = f"The {display_name} is: {value}."

            segments.append(
                Segment(
                    field=col_schema.key,
                    display_name=display_name,
                    value=value,
                    text=text,
                    metadata={
                        "type": col_schema.type,
                        "unit": col_schema.unit,
                        "synonyms": col_schema.synonyms,
                        "shorthand_map": col_schema.shorthand_map,
                        "restatements": col_schema.restatements,
                    },
                )
            )
        return segments

    @staticmethod
    def _format_value(raw_value: str, col_schema: ColumnSchema) -> str:
        """Format a raw value with its unit if applicable."""
        if col_schema.type == "numeric" and col_schema.unit:
            return f"{raw_value} {col_schema.unit}"
        return raw_value

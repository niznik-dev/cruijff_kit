"""Abstract base class for text generation templates."""

from abc import ABC, abstractmethod

from ..schema import ColumnSchema, Schema
from ..segments import Segment


class BaseTemplate(ABC):
    """Base class for all text generation templates."""

    @abstractmethod
    def render_row(
        self,
        features: list[tuple[ColumnSchema, str]],
        schema: Schema,
    ) -> list[Segment]:
        """Convert a single row's features into a list of Segments.

        Args:
            features: List of (column_schema, raw_value) pairs
            schema: The full schema (for additional lookups if needed)

        Returns:
            List of Segment objects, one per feature (dictionary) or
            one per sentence (narrative).
        """
        ...

    @staticmethod
    def _format_value(raw_value: str, col_schema: ColumnSchema) -> str:
        """Format a raw value with its unit if applicable."""
        if col_schema.type == "numeric" and col_schema.unit:
            return f"{raw_value} {col_schema.unit}"
        return raw_value

    @property
    @abstractmethod
    def template_type(self) -> str:
        """Return template type identifier: 'dictionary', 'narrative',
        or 'llm_narrative'."""
        ...

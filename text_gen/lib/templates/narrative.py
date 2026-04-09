"""Deterministic Jinja2 narrative template.

Produces prose that reads more naturally than a bulleted list, using
Jinja2 templates that live in separate .j2 files for easy editing.

Templates must emit ``|||`` between segments. Each delimited block
becomes one Segment, allowing a single segment to reference multiple
features (e.g., "lived in {city} from {year1} to {year2}"). The
segment is tagged with the feature at the corresponding index position.
"""

from pathlib import Path

import jinja2

from ..schema import ColumnSchema, Schema
from ..segments import Segment
from .base import BaseTemplate

BUILTIN_DIR = Path(__file__).parent / "builtin_templates"


class NarrativeTemplate(BaseTemplate):
    """Deterministic Jinja2-based narrative template."""

    def __init__(self, template_file: str | None = None):
        """Initialize with a Jinja2 template file.

        Args:
            template_file: Path to a .j2 file, or a filename within
                builtin_templates/. Defaults to default_narrative.j2.
        """
        if template_file is None:
            template_path = BUILTIN_DIR / "default_narrative.j2"
        elif Path(template_file).is_absolute():
            template_path = Path(template_file)
        else:
            # Try builtin directory first, then treat as relative path
            candidate = BUILTIN_DIR / template_file
            template_path = candidate if candidate.exists() else Path(template_file)

        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        self._template_path = template_path
        self._template = jinja2.Template(template_path.read_text())

    @property
    def template_type(self) -> str:
        return "narrative"

    def render_row(
        self,
        features: list[tuple[ColumnSchema, str]],
        schema: Schema,
    ) -> list[Segment]:
        # Build template context
        feat_dicts = []
        for col_schema, raw_value in features:
            value = self._format_value(raw_value, col_schema)
            feat_dicts.append(
                {
                    "field": col_schema.key,
                    "display_name": col_schema.display_name,
                    "value": value,
                    "type": col_schema.type,
                    "unit": col_schema.unit,
                }
            )

        rendered = self._template.render(features=feat_dicts).strip()

        chunks = [c.strip() for c in rendered.split("|||") if c.strip()]

        segments = []
        for i, chunk in enumerate(chunks):
            if i < len(features):
                col_schema, raw_value = features[i]
                value = self._format_value(raw_value, col_schema)
                segments.append(
                    Segment(
                        field=col_schema.key,
                        display_name=col_schema.display_name,
                        value=value,
                        text=chunk,
                        metadata={
                            "type": col_schema.type,
                            "unit": col_schema.unit,
                            "synonyms": col_schema.synonyms,
                            "shorthand_map": col_schema.shorthand_map,
                            "restatements": col_schema.restatements,
                        },
                    )
                )
            else:
                segments.append(
                    Segment(
                        field="",
                        display_name="",
                        value="",
                        text=chunk,
                        metadata={},
                    )
                )

        return segments

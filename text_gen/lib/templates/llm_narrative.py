"""LLM-generated narrative template (optional dependency).

Produces natural-language descriptions by calling the Anthropic API.
Requires the 'anthropic' package and ANTHROPIC_API_KEY environment
variable. Maintains a JSON cache for consistency across re-runs.

LLM outputs are non-deterministic. The seed parameter does not apply.
The cache provides run-to-run consistency for identical inputs.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

from ..schema import ColumnSchema, Schema
from ..segments import Segment
from .base import BaseTemplate

logger = logging.getLogger(__name__)


def _check_dependencies():
    """Check that anthropic is installed and API key is set."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        raise ImportError(
            "LLM narrative mode requires:\n"
            "  1. pip install anthropic\n"
            "  2. export ANTHROPIC_API_KEY=your-key\n\n"
            "Run with --template dictionary or --template narrative "
            "for non-API modes."
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it before using LLM narrative mode:\n"
            "  export ANTHROPIC_API_KEY=your-key"
        )


class LLMNarrativeTemplate(BaseTemplate):
    """LLM-generated narrative template using the Anthropic API."""

    MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self, cache_path: str | None = None, style_guidance: str | None = None
    ):
        """Initialize with optional cache path and style guidance.

        Args:
            cache_path: Path to JSON cache file. If None, caching is
                disabled (each call hits the API).
            style_guidance: Optional user-provided instructions for how
                the text should be styled (e.g., tone, token density,
                intended use). Appended to the API prompt.
        """
        _check_dependencies()
        import anthropic

        self._client = anthropic.Anthropic()
        self._cache_path = Path(cache_path) if cache_path else None
        self._style_guidance = style_guidance
        self._cache: dict[str, str] = {}

        if self._cache_path and self._cache_path.exists():
            with open(self._cache_path) as f:
                self._cache = json.load(f)
            logger.info(
                "Loaded %d cached entries from %s", len(self._cache), self._cache_path
            )

    @property
    def template_type(self) -> str:
        return "llm_narrative"

    def render_row(
        self,
        features: list[tuple[ColumnSchema, str]],
        schema: Schema,
    ) -> list[Segment]:
        cache_key = self._make_cache_key(features)

        if cache_key in self._cache:
            text = self._cache[cache_key]
        else:
            text = self._call_api(features)
            self._cache[cache_key] = text
            self._save_cache()

        # Split into sentences, one Segment per sentence
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        segments = []
        for i, sentence in enumerate(sentences):
            sentence_text = sentence + "."
            if i < len(features):
                col_schema, raw_value = features[i]
                value = raw_value
                if col_schema.type == "numeric" and col_schema.unit:
                    value = f"{raw_value} {col_schema.unit}"
                segments.append(
                    Segment(
                        field=col_schema.key,
                        display_name=col_schema.display_name,
                        value=value,
                        text=sentence_text,
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
                        text=sentence_text,
                        metadata={},
                    )
                )

        return segments

    def _call_api(self, features: list[tuple[ColumnSchema, str]]) -> str:
        """Call the Anthropic API to generate a narrative description."""
        feature_lines = []
        for col_schema, raw_value in features:
            value = raw_value
            if col_schema.type == "numeric" and col_schema.unit:
                value = f"{raw_value} {col_schema.unit}"
            feature_lines.append(f"- {col_schema.display_name}: {value}")

        feature_text = "\n".join(feature_lines)
        prompt = (
            "Write a natural-language description of a person based on "
            "the following data. Use the provided attribute names and "
            "values naturally in your prose. Do not add information "
            "that is not present in the data. Write 2-4 sentences.\n\n"
            f"{feature_text}"
        )

        if self._style_guidance:
            prompt += f"\n\nAdditional instructions: {self._style_guidance}"

        response = self._client.messages.create(
            model=self.MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _make_cache_key(self, features: list[tuple[ColumnSchema, str]]) -> str:
        """Create a deterministic cache key from features and style guidance."""
        parts = [(cs.key, v) for cs, v in features]
        raw = json.dumps(parts, sort_keys=True)
        if self._style_guidance:
            raw += f"|style={self._style_guidance}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _save_cache(self):
        """Persist cache to disk."""
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)

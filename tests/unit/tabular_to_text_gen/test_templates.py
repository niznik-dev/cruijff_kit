"""Tests for tabular_to_text_gen/lib/templates/ — dictionary, narrative, and factory."""

import pytest

from cruijff_kit.tabular_to_text_gen.lib.templates import get_template
from cruijff_kit.tabular_to_text_gen.lib.templates.dictionary import DictionaryTemplate
from cruijff_kit.tabular_to_text_gen.lib.templates.narrative import NarrativeTemplate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(schema):
    """Build a feature list from schema columns (excluding PINCP)."""
    row = {"AGEP": "51", "ST": "New York", "OCCP": "Teacher"}
    return [(schema.get_column(k), row[k]) for k in ["AGEP", "ST", "OCCP"]]


# ---------------------------------------------------------------------------
# DictionaryTemplate
# ---------------------------------------------------------------------------


class TestDictionaryTemplate:
    def test_template_type(self):
        assert DictionaryTemplate().template_type == "dictionary"

    def test_segments_per_feature(self, schema):
        t = DictionaryTemplate()
        segments = t.render_row(_make_features(schema), schema)
        assert len(segments) == 3

    def test_segment_text_format(self, schema):
        t = DictionaryTemplate()
        segments = t.render_row(_make_features(schema), schema)
        # Numeric with unit
        assert segments[0].text == "The age is: 51 years old."
        # Categorical without unit
        assert segments[1].text == "The state is: New York."
        assert segments[2].text == "The occupation is: Teacher."

    def test_segment_metadata_carries_schema(self, schema):
        t = DictionaryTemplate()
        segments = t.render_row(_make_features(schema), schema)
        assert "synonyms" in segments[0].metadata
        assert "shorthand_map" in segments[1].metadata

    def test_segment_field_and_display_name(self, schema):
        t = DictionaryTemplate()
        segments = t.render_row(_make_features(schema), schema)
        assert segments[0].field == "AGEP"
        assert segments[0].display_name == "age"


# ---------------------------------------------------------------------------
# NarrativeTemplate
# ---------------------------------------------------------------------------


class TestNarrativeTemplate:
    def test_template_type(self, narrative_template):
        assert (
            NarrativeTemplate(template_file=narrative_template).template_type
            == "narrative"
        )

    def test_produces_segments(self, schema, narrative_template):
        t = NarrativeTemplate(template_file=narrative_template)
        segments = t.render_row(_make_features(schema), schema)
        assert len(segments) > 0

    def test_no_bullet_prefix(self, schema, narrative_template):
        t = NarrativeTemplate(template_file=narrative_template)
        segments = t.render_row(_make_features(schema), schema)
        for seg in segments:
            assert not seg.text.startswith("- ")

    def test_segments_end_with_period(self, schema, narrative_template):
        t = NarrativeTemplate(template_file=narrative_template)
        segments = t.render_row(_make_features(schema), schema)
        for seg in segments:
            assert seg.text.endswith(".")

    def test_custom_template_file(self, tmp_path, schema):
        j2 = tmp_path / "custom.j2"
        j2.write_text(
            "{% for feat in features %}{{ feat.display_name }}: {{ feat.value }}. {% endfor %}"
        )
        t = NarrativeTemplate(template_file=str(j2))
        segments = t.render_row(_make_features(schema), schema)
        assert len(segments) > 0

    def test_missing_template_raises(self):
        with pytest.raises(FileNotFoundError):
            NarrativeTemplate(template_file="/nonexistent/template.j2")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestGetTemplate:
    def test_dictionary(self):
        t = get_template("dictionary")
        assert t.template_type == "dictionary"

    def test_narrative(self, narrative_template):
        t = get_template("narrative", template_file=narrative_template)
        assert t.template_type == "narrative"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown template type"):
            get_template("nonexistent")

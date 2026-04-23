"""Tests for tabular_to_text_gen/lib/segments.py — Segment dataclass and rendering."""

from tabular_to_text_gen.lib.segments import Segment, render_segments


class TestSegmentDataclass:
    def test_defaults(self):
        seg = Segment(field="X", display_name="x", value="1", text="x=1")
        assert seg.metadata == {}
        assert seg.is_added is False

    def test_explicit_fields(self):
        seg = Segment(
            field="AGEP",
            display_name="age",
            value="51",
            text="age: 51",
            metadata={"unit": "years"},
            is_added=True,
        )
        assert seg.field == "AGEP"
        assert seg.is_added is True
        assert seg.metadata["unit"] == "years"


class TestRenderSegments:
    def test_dictionary_format(self, sample_segments):
        """Dictionary format: newline-separated with '- ' prefix."""
        result = render_segments(sample_segments, "dictionary")
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "- The age is: 51 years old."
        assert lines[1] == "- The state is: New York."
        assert lines[2] == "- The occupation is: Teacher."

    def test_narrative_format(self, make_segment):
        """Narrative format: space-separated, no prefix."""
        segments = [
            make_segment(text="The respondent is 51 years old."),
            make_segment(text="They live in New York."),
            make_segment(text="They work as a teacher."),
        ]
        result = render_segments(segments, "narrative")
        assert result == (
            "The respondent is 51 years old. "
            "They live in New York. "
            "They work as a teacher."
        )
        assert "\n" not in result

    def test_llm_narrative_uses_space_join(self, make_segment):
        """llm_narrative uses the same join as narrative."""
        segments = [
            make_segment(text="Sentence one."),
            make_segment(text="Sentence two."),
        ]
        assert render_segments(segments, "llm_narrative") == render_segments(
            segments, "narrative"
        )

    def test_custom_separator(self, sample_segments):
        result = render_segments(sample_segments, "dictionary", separator=" | ")
        assert " | " in result
        # Custom separator overrides the '- ' prefix
        assert not result.startswith("- ")

    def test_empty_segments(self):
        assert render_segments([], "dictionary") == ""
        assert render_segments([], "narrative") == ""

    def test_single_segment(self, make_segment):
        seg = make_segment(text="Hello.")
        assert render_segments([seg], "dictionary") == "- Hello."
        assert render_segments([seg], "narrative") == "Hello."

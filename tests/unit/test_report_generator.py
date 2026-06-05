"""Unit tests for tools/inspect/report_generator.py — PDF preprocessing helpers.

The module no longer assembles reports (explore-experiment authors ``report.md``
directly). What remains is the markdown-to-PDF preprocessing used by the
analyze-to-pdf skill.
"""

from cruijff_kit.tools.inspect.report_generator import expand_details_for_pdf


# =============================================================================
# expand_details_for_pdf()
# =============================================================================


class TestExpandDetailsForPdf:
    def test_summary_becomes_bold(self):
        """<summary> text becomes a **bold** label."""
        text = "<details>\n<summary>My Section</summary>\n\nSome content.\n\n</details>"
        result = expand_details_for_pdf(text)
        assert "**My Section**" in result
        assert "<details>" not in result
        assert "<summary>" not in result

    def test_content_preserved(self):
        """Inner content is kept after expansion."""
        text = "<details>\n<summary>Logs</summary>\n\n- `log1.eval`\n- `log2.eval`\n\n</details>"
        result = expand_details_for_pdf(text)
        assert "log1.eval" in result
        assert "log2.eval" in result

    def test_pre_code_to_fenced(self):
        """<pre><code> blocks become fenced code blocks."""
        text = (
            "<details>\n<summary>Commands</summary>\n\n"
            "<pre><code>echo hello\necho world</code></pre>\n\n</details>"
        )
        result = expand_details_for_pdf(text)
        assert "```\necho hello\necho world\n```" in result
        assert "<pre>" not in result
        assert "<code>" not in result

    def test_multiple_details_blocks(self):
        """Multiple <details> blocks are all expanded."""
        text = (
            "<details>\n<summary>First</summary>\n\nAAA\n\n</details>\n\n"
            "<details>\n<summary>Second</summary>\n\nBBB\n\n</details>"
        )
        result = expand_details_for_pdf(text)
        assert "**First**" in result
        assert "**Second**" in result
        assert "AAA" in result
        assert "BBB" in result
        assert "<details>" not in result

    def test_no_details_passthrough(self):
        """Text without <details> blocks is returned unchanged."""
        text = "# Heading\n\nSome regular markdown.\n"
        assert expand_details_for_pdf(text) == text

    def test_authored_report_round_trip(self):
        """An authored report's provenance <details> blocks fully expand."""
        report = (
            "# Claude's Exploration\n\n"
            "Bottom line: the 3B model wins.\n\n"
            "<details>\n<summary>Source eval logs</summary>\n\n"
            "- `/exp/run1/eval/logs/log1.eval`\n\n</details>\n\n"
            "<details>\n<summary>Inspect view commands</summary>\n\n"
            "<pre><code>inspect view start --log-dir /exp/run1/eval/logs</code></pre>\n\n"
            "</details>\n"
        )
        expanded = expand_details_for_pdf(report)
        assert "<details>" not in expanded
        assert "</details>" not in expanded
        assert "<summary>" not in expanded
        # Content survives the expansion.
        assert "log1.eval" in expanded
        assert "inspect view start" in expanded

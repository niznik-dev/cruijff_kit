"""Unit tests for tools/inspect/report_generator.py — PDF preprocessing helpers.

The module no longer assembles reports (explore-experiment authors ``report.md``
directly). What remains is the markdown-to-PDF preprocessing used by the
analyze-to-pdf skill.
"""

from cruijff_kit.tools.inspect.report_generator import (
    PDF_LATEX_HEADER,
    expand_details_for_pdf,
    sanitize_unicode_for_pdf,
)


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


# =============================================================================
# sanitize_unicode_for_pdf()
# =============================================================================


class TestSanitizeUnicodeForPdf:
    def test_relation_glyphs_become_latex_math(self):
        """Dropped relation symbols map to core LaTeX math."""
        assert sanitize_unicode_for_pdf("a ≥ b") == r"a $\geq$  b"
        assert sanitize_unicode_for_pdf("c ≈ d") == r"c $\approx$  d"

    def test_decorative_glyphs_become_ascii(self):
        """Checkmarks/warnings the font drops become self-contained ASCII."""
        assert sanitize_unicode_for_pdf("done ✓") == "done [x]"
        assert sanitize_unicode_for_pdf("heads up ⚠") == "heads up [!]"

    def test_glyph_glued_to_digit_still_converts(self):
        """Regression: a closing ``$`` before a digit trips pandoc's currency
        heuristic, so the trailing space is required (#551)."""
        out = sanitize_unicode_for_pdf("p≈0.6 and x≥0.95")
        assert "≈" not in out and "≥" not in out
        assert r"$\approx$ 0" in out
        assert r"$\geq$ 0" in out

    def test_superscript_uses_text_mode(self):
        """Superscripts avoid ``$`` entirely (no currency-heuristic risk)."""
        assert sanitize_unicode_for_pdf("x⁻") == r"x\textsuperscript{-}"
        assert sanitize_unicode_for_pdf("m³") == r"m\textsuperscript{3}"

    def test_glyphs_in_inline_code_left_verbatim(self):
        """A ``$\\geq$`` inside backticks would print literally, so code spans
        keep their raw symbol."""
        assert sanitize_unicode_for_pdf("see `x ≥ y`") == "see `x ≥ y`"

    def test_fenced_code_block_untouched(self):
        """Fenced blocks are verbatim — no glyph or break rewriting."""
        block = "```\n≥ ✓ /a/b/c\n```"
        assert sanitize_unicode_for_pdf(block) == block

    def test_long_inline_code_gets_break_points(self):
        """Long unbreakable paths in inline code get zero-width breaks so they
        wrap instead of overflowing the page (#551)."""
        path = "`/home/user/projects/folktexts/run_TedEaHgxai7c6nPx35.eval`"
        out = sanitize_unicode_for_pdf(path)
        assert "​" in out
        # the visible characters are all preserved (break is zero-width)
        assert out.replace("​", "") == path

    def test_short_inline_code_not_littered(self):
        """Short code spans stay clean — no spurious break characters."""
        assert sanitize_unicode_for_pdf("`raw`") == "`raw`"
        assert "​" not in sanitize_unicode_for_pdf("`raw`")

    def test_no_special_chars_passthrough(self):
        """Plain ASCII markdown is returned unchanged."""
        text = "# Heading\n\nNormal prose with no symbols.\n"
        assert sanitize_unicode_for_pdf(text) == text

    def test_typographic_chars_preserved(self):
        """Glyphs the default font *does* render (em dash, arrows) are left
        alone so typography is not degraded."""
        text = "1B → 3B — a fine result"
        assert sanitize_unicode_for_pdf(text) == text


class TestPdfLatexHeader:
    def test_header_has_image_and_break_directives(self):
        """The header caps images and enables long-line breaking."""
        assert "keepaspectratio" in PDF_LATEX_HEADER
        assert "graphicx" in PDF_LATEX_HEADER
        assert "emergencystretch" in PDF_LATEX_HEADER

    def test_header_only_uses_core_packages(self):
        """Stays portable: no packages missing from a minimal TeX Live."""
        for risky in ("hyphenat", "fvextra", "seqsplit", "xurl"):
            assert risky not in PDF_LATEX_HEADER

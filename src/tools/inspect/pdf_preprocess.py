"""
Markdown preprocessing helpers for turning exploration reports into PDFs.

Exploration reports (``report.md``) are authored directly by the
explore-experiment skill — Claude reads ``summary.md`` and the evaluation logs
and writes the narrative, tables, and figures itself. This module no longer
assembles reports; it only holds the small, reusable transforms that prepare an
authored ``report.md`` for ``pandoc`` (see the analyze-to-pdf skill).

Example usage:
    from cruijff_kit.tools.inspect.pdf_preprocess import expand_details_for_pdf

    pdf_ready = expand_details_for_pdf(Path("report.md").read_text())
"""

import re


def expand_details_for_pdf(text: str) -> str:
    """Expand HTML ``<details>`` blocks into plain markdown for PDF conversion.

    Collapsible sections are a browser concept; in print they should just be
    visible.  This function:

    * Converts ``<summary>`` text to a bold label.
    * Converts ``<pre><code>`` blocks to fenced code blocks.
    * Strips the wrapping ``<details>``/``</details>`` tags.

    Args:
        text: Markdown report text (may contain ``<details>`` blocks).

    Returns:
        Markdown text with all ``<details>`` blocks expanded.
    """

    def _expand_block(match: re.Match) -> str:
        content = match.group(1)

        # Extract and remove <summary>
        summary_match = re.search(r"<summary>(.*?)</summary>", content)
        summary = summary_match.group(1) if summary_match else ""
        content = re.sub(r"<summary>.*?</summary>\s*", "", content)

        # Convert <pre><code>...</code></pre> to fenced code blocks
        content = re.sub(
            r"<pre><code>(.*?)</code></pre>",
            lambda m: f"```\n{m.group(1)}\n```",
            content,
            flags=re.DOTALL,
        )

        return f"**{summary}**\n\n{content.strip()}\n"

    return re.sub(
        r"<details>\s*\n?(.*?)</details>",
        _expand_block,
        text,
        flags=re.DOTALL,
    )


# LaTeX preamble injected via pandoc ``-H`` when converting reports to PDF.
# Every directive uses only packages present in a minimal TeX Live install
# (graphicx and etoolbox are always available); nothing exotic is required.
#
# * ``\setkeys{Gin}`` caps every image at the text width and ~85% of the text
#   height with aspect ratio locked, so an oversized matplotlib PNG scales down
#   to fit instead of bleeding past the margin.
# * ``\AtBeginEnvironment{longtable}`` shrinks wide tables a notch.
# * ``\sloppy`` plus ``\emergencystretch`` let TeX break lines at separators
#   that would otherwise push long inline code (filenames separated by spaces or
#   middots) into the right margin.
PDF_LATEX_HEADER = r"""\usepackage{graphicx}
\setkeys{Gin}{width=\linewidth,height=0.85\textheight,keepaspectratio}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\footnotesize}
\setlength{\tabcolsep}{4pt}
\sloppy
\setlength{\emergencystretch}{3em}
"""

# Zero-width space — a line-break opportunity that renders invisibly under
# xelatex. Injected into long inline-code spans so unbreakable tokens (e.g.
# absolute eval-log paths with hash suffixes) can wrap instead of running off
# the page.
_ZWSP = "\u200b"

# Glyphs that the default LaTeX font (Latin Modern) silently drops, leaving
# blank gaps in the PDF. Math symbols map to LaTeX math (core, no extra
# package); decorative marks map to self-contained ASCII so the result needs no
# header support to render.
#
# The trailing space on the math entries is load-bearing: pandoc treats a
# closing ``$`` immediately followed by a digit as a literal dollar (its
# currency heuristic), which would silently break "≥0.95" and similar.
# A trailing space dodges that. A *leading* space is deliberately avoided —
# it would break adjacent markdown emphasis (``**≈4pp**`` would stop bolding).
# Superscripts use text-mode ``\textsuperscript`` to sidestep math/``$`` entirely.
_PDF_GLYPH_MAP = {
    # relation / binary math symbols -> LaTeX math (core, no extra package)
    "≥": r"$\geq$ ",  # ≥
    "≤": r"$\leq$ ",  # ≤
    "≠": r"$\neq$ ",  # ≠
    "≈": r"$\approx$ ",  # ≈
    "≡": r"$\equiv$ ",  # ≡
    "⊗": r"$\otimes$ ",  # ⊗
    "√": r"$\surd$ ",  # √
    "≳": ">~ ",  # ≳ greater-than-or-equivalent (no core math symbol)
    # superscripts -> text mode
    "⁻": r"\textsuperscript{-}",  # ⁻ superscript minus
    "³": r"\textsuperscript{3}",  # ³ superscript three
    # decorative marks -> ASCII
    "✓": "[x]",  # ✓
    "✗": "[ ]",  # ✗
    "✘": "[ ]",  # ✘
    "⚠": "[!]",  # ⚠
}


def _inject_inline_code_breaks(code: str) -> str:
    """Add zero-width break points after path separators in long code spans."""
    if len(code) <= 30:
        return code
    return re.sub(r"([/._\-\\])", r"\1" + _ZWSP, code)


def sanitize_unicode_for_pdf(text: str) -> str:
    """Make a markdown report safe for LaTeX-based PDF conversion.

    Two transformations, applied only where they are safe:

    * **Glyph substitution** — replaces Unicode symbols that the default LaTeX
      font drops (``≥ ≈ ✓ ⚠`` …) with LaTeX-math or ASCII
      equivalents, so they render instead of vanishing. Applied to prose only;
      code spans and fenced code blocks are left verbatim (a ``$\\geq$`` inside
      backticks would print literally).
    * **Inline-code line breaks** — injects zero-width spaces into long inline
      code so unbreakable tokens (absolute paths, hashed filenames) wrap rather
      than overflow the right margin. Applied to inline code only.

    Args:
        text: Markdown report text.

    Returns:
        Sanitized markdown ready for ``pandoc --pdf-engine=xelatex``.
    """

    def _replace_glyphs(prose: str) -> str:
        for glyph, repl in _PDF_GLYPH_MAP.items():
            prose = prose.replace(glyph, repl)
        return prose

    out: list[str] = []
    # Split on fenced code blocks first and leave them untouched.
    for block_idx, block in enumerate(re.split(r"(```.*?```)", text, flags=re.DOTALL)):
        if block_idx % 2 == 1:
            out.append(block)
            continue
        # Within prose, separate inline code spans from surrounding text.
        for span_idx, span in enumerate(re.split(r"(`[^`\n]+`)", block)):
            if span_idx % 2 == 1:
                out.append("`" + _inject_inline_code_breaks(span[1:-1]) + "`")
            else:
                out.append(_replace_glyphs(span))
    return "".join(out)

"""
Markdown preprocessing helpers for turning exploration reports into PDFs.

Exploration reports (``report.md``) are authored directly by the
explore-experiment skill — Claude reads ``summary.md`` and the evaluation logs
and writes the narrative, tables, and figures itself. This module no longer
assembles reports; it only holds the small, reusable transforms that prepare an
authored ``report.md`` for ``pandoc`` (see the analyze-to-pdf skill).

Example usage:
    from cruijff_kit.tools.inspect.report_generator import expand_details_for_pdf

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

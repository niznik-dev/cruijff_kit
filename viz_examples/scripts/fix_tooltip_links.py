"""
Post-process inspect-viz HTML to make tooltip links more clickable.

This script adds CSS to make tooltips more stable and keep them visible
when hovering to click links.
"""

import re
import sys


def fix_tooltip_config(html_content: str) -> str:
    """
    Add CSS to keep tooltips visible and stable when hovering over them.

    This uses CSS pointer-events and transitions to prevent flickering.
    """

    # Add custom CSS at the beginning of the file
    custom_css = """<style>
/* Make tooltips stay visible longer for clicking links */
.tippy-box[data-theme~='inspect'] {
    pointer-events: all !important;
    transition: opacity 0.3s ease-in-out !important;
}

/* Extend the tooltip hover area */
.tippy-box[data-theme~='inspect'] .tippy-content {
    pointer-events: all !important;
}

/* Make links in tooltips more clickable */
.tippy-box[data-theme~='inspect'] .tippy-content a {
    pointer-events: all !important;
    cursor: pointer !important;
    text-decoration: underline !important;
    color: #0066cc !important;
    padding: 2px 4px !important;
}

.tippy-box[data-theme~='inspect'] .tippy-content a:hover {
    background-color: rgba(0, 102, 204, 0.1) !important;
}

/* Keep tooltip visible during interaction */
[data-tippy-root] {
    pointer-events: auto !important;
}
</style>

"""

    # Insert at the beginning of the file
    html_content = custom_css + html_content

    return html_content


def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_tooltip_links.py <html_file>")
        sys.exit(1)

    html_file = sys.argv[1]

    # Read HTML
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Fix tooltip config
    fixed_html = fix_tooltip_config(html_content)

    # Write back
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(fixed_html)

    print(f"✓ Added CSS to improve tooltip interaction in {html_file}")
    print("  - Added pointer-events: all for tooltips")
    print("  - Added transition for smoother interaction")
    print("  - Styled links to be more clickable")


if __name__ == "__main__":
    main()

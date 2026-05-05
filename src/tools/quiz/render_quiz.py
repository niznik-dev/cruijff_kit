"""Render a quiz spec (JSON) into a self-contained HTML file.

The renderer is dumb on purpose — all the question authoring happens in the
LLM step driving the create-quiz skill. This module just pours JSON into a
Jinja2 template and base64-encodes any image assets so the resulting HTML is
portable.
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import jinja2
import markdown as md

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_FILE = "quiz.html.j2"

EQUATION_RE = re.compile(r"(?<!\\)\$\$|(?<!\\)\$")
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

MD_EXTENSIONS = ["fenced_code", "tables", "sane_lists"]


def _md_to_html(text: str | None) -> str:
    if not text:
        return ""
    return md.markdown(text, extensions=MD_EXTENSIONS)


def _embed_markdown_images(text: str, base_dir: Path | None) -> str:
    """Replace ``![alt](relative.png)`` with base64 data URIs.

    Leaves external URLs and already-embedded data URIs untouched. Skips
    references that can't be resolved on disk (renders the original markdown
    so the broken-image marker is visible).
    """
    if not text or base_dir is None:
        return text

    def replace(match: re.Match) -> str:
        alt, path = match.group(1), match.group(2)
        if path.startswith(("http://", "https://", "data:")):
            return match.group(0)
        if not path.lower().endswith(".png"):
            return match.group(0)
        png_path = Path(path)
        if not png_path.is_absolute():
            png_path = (base_dir / path).resolve()
        if not png_path.exists():
            return match.group(0)
        return f"![{alt}]({_embed_png(png_path)})"

    return MD_IMAGE_RE.sub(replace, text)


def _embed_png(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _needs_mathjax(spec: dict) -> bool:
    haystacks: list[str] = [spec.get("intro", "")]
    for q in spec.get("questions", []):
        haystacks.append(q.get("prompt", ""))
        haystacks.append(q.get("explanation", ""))
        haystacks.append(q.get("model_answer", ""))
        for choice in q.get("choices", []) or []:
            haystacks.append(choice)
        for item in q.get("items", []) or []:
            haystacks.append(item)
    return any(EQUATION_RE.search(text) for text in haystacks if text)


def _prepare_questions(spec: dict, spec_path: Path) -> list[dict]:
    """Resolve image assets, attach base64 data URLs, and pre-render markdown.

    Drops the original ``asset.path`` once a ``data_url`` is attached so the
    output HTML doesn't leak local filesystem paths. Pre-renders ``prompt``
    and ``explanation`` fields as HTML (stored under ``prompt_html`` /
    ``explanation_html``) so the template + JS can drop them in directly.
    """
    prepared = []
    for q in spec.get("questions", []):
        q = dict(q)  # shallow copy
        asset = q.get("asset")
        if asset and asset.get("kind") == "png":
            asset_path = Path(asset["path"])
            if not asset_path.is_absolute():
                asset_path = (spec_path.parent / asset_path).resolve()
            if not asset_path.exists():
                raise FileNotFoundError(
                    f"Question {q.get('id', '?')} asset not found: {asset_path}"
                )
            asset = {k: v for k, v in asset.items() if k != "path"}
            asset["data_url"] = _embed_png(asset_path)
            q["asset"] = asset
        q["prompt_html"] = _md_to_html(q.get("prompt"))
        q["explanation_html"] = _md_to_html(q.get("explanation"))
        prepared.append(q)
    return prepared


def render(spec_path: str | Path, out_path: str | Path) -> Path:
    """Render a quiz spec to HTML.

    Args:
        spec_path: Path to a quiz.json file (see generation.md for schema).
        out_path: Where to write the resulting quiz.html.

    Returns:
        The output path.
    """
    spec_path = Path(spec_path)
    out_path = Path(out_path)

    spec = json.loads(spec_path.read_text())

    if not isinstance(spec.get("questions"), list) or not spec["questions"]:
        raise ValueError(f"{spec_path} has no questions array")
    for q in spec["questions"]:
        if not q.get("explanation"):
            raise ValueError(
                f"Question {q.get('id', '?')} is missing required 'explanation' field"
            )

    prepared_questions = _prepare_questions(spec, spec_path)
    mathjax_needed = _needs_mathjax(spec)
    intro_html = _md_to_html(spec.get("intro"))

    writeup_md = spec.get("full_writeup_md")
    writeup_dir = spec.get("full_writeup_image_dir")
    if writeup_md:
        base_dir = Path(writeup_dir) if writeup_dir else None
        writeup_html = _md_to_html(_embed_markdown_images(writeup_md, base_dir))
    else:
        writeup_html = ""

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(TEMPLATE_FILE)
    html = template.render(
        spec=spec,
        intro_html=intro_html,
        questions=prepared_questions,
        mathjax_needed=mathjax_needed,
        questions_json=json.dumps(prepared_questions),
        writeup_html=writeup_html,
        experiment_summary_yaml=spec.get("experiment_summary_yaml", ""),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to quiz.json")
    parser.add_argument("--out", required=True, help="Output quiz.html path")
    args = parser.parse_args(argv)

    out = render(args.spec, args.out)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Tests for cruijff_kit.tools.quiz.render_quiz."""

import base64
import json
import struct
import zlib
from pathlib import Path

import pytest

from cruijff_kit.tools.quiz.render_quiz import _needs_mathjax, render


def _tiny_png() -> bytes:
    """Return a minimal valid 1x1 PNG (red pixel) without external deps."""
    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    # IHDR: 1x1, 8-bit, RGB
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    # IDAT: one filtered scanline of one RGB pixel (255,0,0), zlib-compressed
    raw = b"\x00\xff\x00\x00"
    idat = zlib.compress(raw)
    iend = b""
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", iend)


@pytest.fixture
def fixture_dir(tmp_path: Path) -> Path:
    png_path = tmp_path / "headline.png"
    png_path.write_bytes(_tiny_png())

    spec = {
        "version": "1",
        "title": "Test experiment",
        "intro": "## Setup\n\nA short intro with **markdown**.\n\nExample: input `1 2 3` → output `3`.",
        "experiments": [{"name": "test_exp"}],
        "questions": [
            {
                "id": "q1",
                "type": "multiple_choice",
                "prompt": "Pick the right one.",
                "asset": None,
                "choices": ["A", "B", "C"],
                "answer_index": 1,
                "explanation": "B is correct: the table row 'foo' shows **B = 0.42**, which is the highest of the three.",
            },
            {
                "id": "q2",
                "type": "numerical_estimate",
                "prompt": "Predict accuracy at k=50, N=500. Answer 0-100, ±5pp.",
                "asset": None,
                "answer": 98.7,
                "tolerance": 5,
                "min": 0,
                "max": 100,
                "step": 0.1,
                "unit": "%",
                "explanation": "The cell value was 98.7%.",
            },
            {
                "id": "q3",
                "type": "ranking",
                "prompt": "Order from smallest to largest.",
                "asset": None,
                "items": ["c", "a", "b"],
                "answer_order": ["a", "b", "c"],
                "explanation": "Alphabetical.",
            },
            {
                "id": "q4",
                "type": "image_read",
                "prompt": "What does the figure show?",
                "asset": {
                    "kind": "png",
                    "path": str(png_path),
                    "alt": "headline",
                },
                "answer_type": "string",
                "answer": "red",
                "explanation": "The image is a red pixel.",
            },
            {
                "id": "q5",
                "type": "equation_or_baseline",
                "prompt": "Random-guess accuracy for $k=10$ classes?",
                "asset": None,
                "answer": 0.10,
                "tolerance": 0.001,
                "unit": "accuracy",
                "explanation": "Cite the hypothesis. $1/k = 0.1$.",
            },
            {
                "id": "q6",
                "type": "free_text_intuition",
                "prompt": "What would you do next?",
                "asset": None,
                "model_answer": "Run a follow-up at higher N.",
                "key_points": ["scale N", "fixed seed"],
                "explanation": "Future-directions section suggests N=8000.",
            },
        ],
    }

    spec_path = tmp_path / "quiz.json"
    spec_path.write_text(json.dumps(spec))
    return tmp_path


def test_render_writes_file(fixture_dir: Path):
    out = render(fixture_dir / "quiz.json", fixture_dir / "quiz.html")
    assert out.exists()
    assert out.stat().st_size > 0


def test_html_contains_each_prompt(fixture_dir: Path):
    out = render(fixture_dir / "quiz.json", fixture_dir / "quiz.html")
    html = out.read_text()
    assert "Pick the right one." in html
    assert "Predict accuracy at k=50, N=500" in html
    assert "Order from smallest to largest." in html
    assert "What does the figure show?" in html
    assert "Random-guess accuracy for" in html
    assert "What would you do next?" in html


def test_png_embedded_as_base64(fixture_dir: Path):
    out = render(fixture_dir / "quiz.json", fixture_dir / "quiz.html")
    html = out.read_text()
    # Must contain a base64-encoded PNG; must not contain the original path
    assert "data:image/png;base64," in html
    assert str(fixture_dir / "headline.png") not in html
    # The base64 should decode to the same bytes we wrote
    encoded = html.split("data:image/png;base64,")[1].split('"')[0].split("'")[0]
    decoded = base64.b64decode(encoded)
    assert decoded.startswith(b"\x89PNG")


def test_mathjax_loaded_when_equations_present(fixture_dir: Path):
    out = render(fixture_dir / "quiz.json", fixture_dir / "quiz.html")
    html = out.read_text()
    assert "MathJax" in html  # CDN script should be present


def test_no_external_refs_except_whitelisted(fixture_dir: Path):
    out = render(fixture_dir / "quiz.json", fixture_dir / "quiz.html")
    html = out.read_text()
    # External http(s) refs must be on the whitelist:
    #   - MathJax CDN + its polyfill dependency (only if intro/questions use math)
    #   - The cruijff_kit GitHub brand link in the header banner
    import re

    refs = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    allowed_substrings = ("mathjax", "polyfill", "github.com/niznik-dev/cruijff_kit")
    for ref in refs:
        ref_lc = ref.lower()
        assert any(s in ref_lc for s in allowed_substrings), (
            f"Unexpected external reference: {ref}"
        )


def test_missing_explanation_raises(tmp_path: Path):
    spec = {
        "version": "1",
        "title": "x",
        "intro": "y",
        "questions": [
            {
                "id": "q1",
                "type": "multiple_choice",
                "prompt": "?",
                "choices": ["a", "b"],
                "answer_index": 0,
                # NO explanation
            }
        ],
    }
    spec_path = tmp_path / "quiz.json"
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="explanation"):
        render(spec_path, tmp_path / "quiz.html")


def test_missing_questions_raises(tmp_path: Path):
    spec = {"version": "1", "title": "x", "intro": "y", "questions": []}
    spec_path = tmp_path / "quiz.json"
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="no questions"):
        render(spec_path, tmp_path / "quiz.html")


def test_missing_png_raises(tmp_path: Path):
    spec = {
        "version": "1",
        "title": "x",
        "intro": "y",
        "questions": [
            {
                "id": "q1",
                "type": "image_read",
                "prompt": "?",
                "asset": {
                    "kind": "png",
                    "path": str(tmp_path / "missing.png"),
                    "alt": "",
                },
                "answer_type": "string",
                "answer": "",
                "explanation": "x",
            }
        ],
    }
    spec_path = tmp_path / "quiz.json"
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(FileNotFoundError):
        render(spec_path, tmp_path / "quiz.html")


def test_needs_mathjax_detection():
    assert _needs_mathjax({"intro": "no math", "questions": []}) is False
    assert _needs_mathjax({"intro": r"price \$5", "questions": []}) is False
    assert (
        _needs_mathjax(
            {"intro": "", "questions": [{"prompt": "$x = 1$", "explanation": ""}]}
        )
        is True
    )
    assert (
        _needs_mathjax(
            {
                "intro": "",
                "questions": [{"prompt": "$$E=mc^2$$", "explanation": ""}],
            }
        )
        is True
    )


def test_questions_json_embedded_for_grading(fixture_dir: Path):
    out = render(fixture_dir / "quiz.json", fixture_dir / "quiz.html")
    html = out.read_text()
    assert "const QUESTIONS = " in html
    # All question ids should appear in the embedded JSON
    for qid in ["q1", "q2", "q3", "q4", "q5", "q6"]:
        assert f'"{qid}"' in html


def test_intro_renders_markdown(fixture_dir: Path):
    out = render(fixture_dir / "quiz.json", fixture_dir / "quiz.html")
    html = out.read_text()
    # The intro had `## Setup`, `**markdown**`, and a code span — all should render
    assert "<h2>Setup</h2>" in html
    assert "<strong>markdown</strong>" in html
    assert "<code>1 2 3</code>" in html


def test_numeric_input_has_min_max(fixture_dir: Path):
    out = render(fixture_dir / "quiz.json", fixture_dir / "quiz.html")
    html = out.read_text()
    # Q2 has min=0, max=100 in spec; the rendered input should carry them as attrs
    import re

    q2_input = re.search(r'<input type="number"[^>]*name="q2"[^>]*>', html)
    assert q2_input is not None, "q2 numeric input not found"
    tag = q2_input.group(0)
    assert 'min="0"' in tag
    assert 'max="100"' in tag


def test_experiments_rendered_after_questions(fixture_dir: Path):
    out = render(fixture_dir / "quiz.json", fixture_dir / "quiz.html")
    html = out.read_text()
    # Footer credit must appear AFTER the form close tag
    form_close = html.find("</form>")
    experiments_block = html.find('class="experiments"')
    assert form_close > 0
    assert experiments_block > form_close, (
        "experiments block should render below the form, not above"
    )


def test_experiments_no_research_question(fixture_dir: Path):
    out = render(fixture_dir / "quiz.json", fixture_dir / "quiz.html")
    html = out.read_text()
    # Even if a fixture had research_question, the template should never render it
    assert "research_question" not in html
    assert "Research question" not in html


def test_explanation_html_in_questions_json(fixture_dir: Path):
    out = render(fixture_dir / "quiz.json", fixture_dir / "quiz.html")
    html = out.read_text()
    # explanation_html (markdown-rendered) should appear in the embedded JSON for JS grading
    assert "explanation_html" in html
    # Q1's explanation has **B = 0.42** which should render to <strong>
    assert "<strong>B = 0.42</strong>" in html


def test_brand_banner_with_link(fixture_dir: Path):
    out = render(fixture_dir / "quiz.json", fixture_dir / "quiz.html")
    html = out.read_text()
    assert 'class="brand"' in html
    assert "github.com/niznik-dev/cruijff_kit" in html
    # The page <title> should mention cruijff_kit so browser tabs advertise it
    assert "cruijff_kit</title>" in html


def test_writeup_appendix_renders_with_embedded_image(tmp_path: Path):
    png_path = tmp_path / "headline.png"
    png_path.write_bytes(_tiny_png())
    spec = {
        "version": "1",
        "title": "x",
        "intro": "y",
        "experiments": [{"name": "x"}],
        "questions": [
            {
                "id": "q1",
                "type": "multiple_choice",
                "prompt": "?",
                "choices": ["a", "b"],
                "answer_index": 0,
                "explanation": "x",
            }
        ],
        "full_writeup_md": (
            "## Results\n\n"
            "Headline accuracy was 100%.\n\n"
            "![Headline plot](headline.png)\n\n"
            "| col | val |\n|---|---|\n| a | 1 |\n"
        ),
        "full_writeup_image_dir": str(tmp_path),
    }
    spec_path = tmp_path / "quiz.json"
    spec_path.write_text(json.dumps(spec))
    out = render(spec_path, tmp_path / "quiz.html")
    html = out.read_text()
    # Appendix wrapper present (writeup details has id="writeup-appendix")
    assert 'id="writeup-appendix"' in html
    assert "Full experiment write-up" in html
    # Markdown rendered
    assert "<h2>Results</h2>" in html
    assert "<table>" in html
    # Image base64-embedded; original relative path NOT in HTML
    assert html.count("data:image/png;base64,") >= 1
    # The relative ref shouldn't survive
    assert 'src="headline.png"' not in html


def test_writeup_omitted_when_field_missing(tmp_path: Path):
    spec = {
        "version": "1",
        "title": "x",
        "intro": "y",
        "experiments": [{"name": "x"}],
        "questions": [
            {
                "id": "q1",
                "type": "multiple_choice",
                "prompt": "?",
                "choices": ["a"],
                "answer_index": 0,
                "explanation": "x",
            }
        ],
    }
    spec_path = tmp_path / "quiz.json"
    spec_path.write_text(json.dumps(spec))
    out = render(spec_path, tmp_path / "quiz.html")
    html = out.read_text()
    # No writeup appendix when field not provided
    assert 'id="writeup-appendix"' not in html


def test_yaml_appendix_renders_escaped(tmp_path: Path):
    yaml_text = "experiment:\n  name: foo\n  question: 'why?'\n  bad: <script>alert(1)</script>\n"
    spec = {
        "version": "1",
        "title": "x",
        "intro": "y",
        "experiments": [{"name": "x"}],
        "questions": [
            {
                "id": "q1",
                "type": "multiple_choice",
                "prompt": "?",
                "choices": ["a"],
                "answer_index": 0,
                "explanation": "x",
            }
        ],
        "experiment_summary_yaml": yaml_text,
    }
    spec_path = tmp_path / "quiz.json"
    spec_path.write_text(json.dumps(spec))
    out = render(spec_path, tmp_path / "quiz.html")
    html = out.read_text()
    assert "Full experimental specification" in html
    assert '<pre class="yaml-block">' in html
    # Script tags must be escaped, not executed
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_yaml_appendix_appears_before_questions(tmp_path: Path):
    spec = {
        "version": "1",
        "title": "x",
        "intro": "y",
        "experiments": [{"name": "x"}],
        "questions": [
            {
                "id": "q1",
                "type": "multiple_choice",
                "prompt": "?",
                "choices": ["a"],
                "answer_index": 0,
                "explanation": "x",
            }
        ],
        "experiment_summary_yaml": "experiment:\n  name: foo\n",
    }
    spec_path = tmp_path / "quiz.json"
    spec_path.write_text(json.dumps(spec))
    out = render(spec_path, tmp_path / "quiz.html")
    html = out.read_text()
    yaml_pos = html.find("Full experimental specification")
    form_pos = html.find('<form id="quiz">')
    assert yaml_pos > 0 and form_pos > 0
    assert yaml_pos < form_pos, (
        "YAML spec should appear before the form (part of the setup, not bottom of page)"
    )


def test_writeup_hidden_until_finalize(tmp_path: Path):
    png_path = tmp_path / "g.png"
    png_path.write_bytes(_tiny_png())
    spec = {
        "version": "1",
        "title": "x",
        "intro": "y",
        "experiments": [{"name": "x"}],
        "questions": [
            {
                "id": "q1",
                "type": "multiple_choice",
                "prompt": "?",
                "choices": ["a"],
                "answer_index": 0,
                "explanation": "x",
            }
        ],
        "full_writeup_md": "## Results\n\nHeadline 100%.",
        "full_writeup_image_dir": str(tmp_path),
    }
    spec_path = tmp_path / "quiz.json"
    spec_path.write_text(json.dumps(spec))
    out = render(spec_path, tmp_path / "quiz.html")
    html = out.read_text()
    # Writeup details element MUST have the `hidden` attribute by default
    import re

    m = re.search(r'<details\b[^>]*id="writeup-appendix"[^>]*>', html)
    assert m is not None, "writeup-appendix details element not found"
    assert "hidden" in m.group(0), (
        "writeup must be hidden by default; recipient sees it only after finalize"
    )
    # The finalize handler must reveal it
    assert "writeup-appendix" in html
    assert "writeup.hidden = false" in html or "writeup.hidden=false" in html

# Generation: quiz.json schema and rendering

## quiz.json schema

The canonical spec the renderer consumes. Path: `{experiment_dir}/quiz/quiz.json`.

```json
{
  "version": "1",
  "title": "string — usually the experiment name(s)",
  "intro": "string (markdown allowed) — must be self-contained: task description, example input → output, experimental axes with definitions, chance baseline. See SKILL.md §4 for what must be present.",
  "experiments": [
    {
      "name": "string"
      // 'directory' field is accepted but ignored; rendered name-only as a footer credit
    }
    // optional second entry for two-experiment mode
  ],
  "questions": [
    {
      "id": "q1",
      "type": "multiple_choice | numerical_estimate | ranking | image_read | equation_or_baseline | free_text_intuition | anomaly_hunt",
      "prompt": "string (markdown allowed; equations as $$...$$ or $...$)",
      "asset": null | {
        "kind": "png",
        "path": "absolute path to the PNG, embedded as base64 by the renderer",
        "alt": "string — alt text for accessibility"
      },

      // Type-specific fields (see question_catalog.md):
      "choices": ["A", "B", "C", "D"],          // multiple_choice, anomaly_hunt
      "answer_index": 2,                         // multiple_choice, anomaly_hunt

      "answer": 98.7,                            // numerical_estimate, equation_or_baseline
      "tolerance": 5,                            // numerical_estimate, equation_or_baseline
      "min": 0,                                  // numerical_estimate (HTML5 input min; rejects out-of-range values)
      "max": 100,                                // numerical_estimate (HTML5 input max)
      "step": 0.1,                               // numerical_estimate (HTML5 input step)
      "unit": "%",                               // numerical_estimate, equation_or_baseline (display only — placeholder text)

      "items": ["k=15", "k=30", ...],            // ranking
      "answer_order": ["k=100", ...],            // ranking

      "answer_type": "string|numeric_with_tolerance|multiple_choice",  // image_read sub-type

      "model_answer": "string",                  // free_text_intuition
      "key_points": ["...", "..."],              // free_text_intuition

      "explanation": "string (markdown). Must be SELF-SUFFICIENT: quote the relevant numbers / facts inline so the recipient can learn the answer without any other file. REQUIRED for every question."
    }
  ],

  // experiment_summary_yaml: rendered as a `<details>` panel labeled "Full experimental specification",
  // placed BETWEEN the intro and the first question (part of the setup, not the answer key).
  "experiment_summary_yaml": "raw text of experiment_summary.yaml. Rendered inside a <pre> block, escaped, scrollable.",

  // full_writeup_md: rendered as a `<details>` panel hidden until the recipient clicks "See Final Score".
  // Contains the answers; must not be available before submission. Auto-opens after reveal.
  "full_writeup_md": "raw markdown content of analysis/report.md. The renderer pre-renders it to HTML. Image references like ![alt](headline.png) are resolved against full_writeup_image_dir and base64-embedded so the file remains portable.",
  "full_writeup_image_dir": "absolute path to the analysis/ directory (the dir containing the PNGs that report.md references)"
}
```

`explanation` is required on every question. The acceptance check in `SKILL.md` enforces this — write a check during validation that `all(q['explanation'] for q in spec['questions'])`.

## Render command

```bash
python -m cruijff_kit.tools.quiz.render_quiz \
  --spec {experiment_dir}/quiz/quiz.json \
  --out {experiment_dir}/quiz/quiz.html
```

The renderer:
1. Loads the JSON spec.
2. Reads each `asset.path` and base64-encodes it.
3. Renders the Jinja2 template at `src/tools/quiz/templates/quiz.html.j2`.
4. Writes a single self-contained HTML file.

The HTML contains:
- Inline CSS (no external stylesheets)
- Inline vanilla JS (per-question reveal, final scoring, no frameworks)
- Embedded base64 images
- One external reference: MathJax CDN, loaded only if any question uses `$...$` or `$$...$$`.

## Authoring sequence

1. Read everything per `parsing.md`.
2. Decide question portfolio per `question_catalog.md`.
3. Construct the spec dict in Python (or assemble JSON directly). Don't hand-write nested JSON in the skill body — build it programmatically and dump it.
4. Validate before writing: `assert len(spec["questions"]) >= 6`, every question has an explanation, every PNG path exists.
5. Write `quiz.json` and run the renderer.
6. Spot-check the resulting HTML — open it (or `head -100`) and confirm the first question renders, the MathJax script tag appears if any equations exist, and base64 images are present (`grep -c "data:image/png;base64" quiz.html`).

## Output layout

```
{experiment_dir}/quiz/
├── quiz.json     # ~5-50 KB, the canonical spec
└── quiz.html     # ~50KB-2MB depending on number of embedded images
```

`quiz.html` is the deliverable the user shares.

### Experiments block renders as a footer, name-only

The `experiments` array is rendered at the **bottom** of `quiz.html` as a small "source experiments" credit. Only the `name` field is shown to the recipient. Do not include the research question, hypothesis, or any other description — those would either give away answers or assume the recipient has prior context they don't have.

`asset.path` (when used) should be an absolute path to the PNG. The renderer reads the bytes and drops the path from the embedded JS to keep the HTML portable.

## Idempotency

Re-running the skill overwrites both files (after asking per the SKILL.md user-question flow). This is intentional — re-running after editing `report.md` or fixing a question.

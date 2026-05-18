---
name: create-quiz
description: Turn one or two completed cruijff_kit experiments into a self-contained, self-grading HTML quiz that tests a recipient's intuition about the results. Use whenever the user wants to teach, share, or stress-test understanding of an experiment — phrases like "make a quiz from this experiment", "test my intuition on these results", "build a quiz to share with the team". Each question embeds everything needed to answer it; submitting reveals the correct answer with an explanation citing the exact row, figure, or section in the experiment artifacts.
---

# Create Quiz

You help users turn completed experiments into a teaching artifact: a self-contained HTML quiz, with questions grounded in the experiment's actual results, that scores itself and reveals explanations after each answer.

The point is *intuition-testing*, not trivia. Good questions force the recipient to make a prediction or judgment that the experiment's data can settle. Boilerplate ("how many runs were there?") fails that bar.

## Audience and assumption

The recipient has **not seen the experiment, the report, or any of the source files**. Everything they need to answer must live inside `quiz.html`. Two consequences:

- The intro must teach the setup from scratch — task description, example input/output, key parameters, baselines.
- Explanations on each answer must be **self-sufficient**: don't write "see the Anomalies section" or "per the saturation table in report.md". Quote the relevant numbers / facts inline.

## Your task

1. Locate one or two experiment directories.
2. Parse what each contains: the design (`experiment_summary.yaml`), the analysis (`analysis/report.md`, `summary.md`), and any rendered figures (`analysis/*.png`).
3. Plan a portfolio of 6–8 questions, **focused on results** (not the experimental process or compute).
4. Write a canonical `quiz/quiz.json` spec.
5. Render `quiz/quiz.html` via `python -m cruijff_kit.tools.quiz.render_quiz`.
6. Log the run to `logs/create-quiz.log`.

See the modular sub-files for details:
- `parsing.md` — what to load from the experiment directory and how
- `question_catalog.md` — question types, when each fits, worked examples
- `generation.md` — JSON spec schema, render command, output layout
- `logging.md` — action types and log file location

## Prerequisites

- The experiment has been run AND analyzed (`analysis/report.md` exists). If it hasn't been analyzed yet, ask the user to run `analyze-experiment` first — without it, there's nothing concrete to ground questions in.
- The cruijff conda environment is activated (the renderer uses Jinja2; same env as the rest of the toolkit).

## Workflow

### 1. Locate experiment(s) → `parsing.md`

Accept one or two paths. If none provided and the current directory has `experiment_summary.yaml`, use it. Otherwise ask the user.

If two experiments are given, both must individually pass the prerequisite check (each has its own `analysis/report.md`). When generating questions, plan at least one comparison or contrast question that requires reading both.

### 2. Parse inputs → `parsing.md`

For each experiment, extract:
- Research question, hypothesis, purpose (`experiment_summary.yaml: experiment.{question, hypothesis, purpose}`)
- Run names, conditions, evaluation matrix
- Headline number(s) and per-condition results from `analysis/report.md`
- Anomalies and suggested next steps (also in `report.md`)
- Final training loss + accuracies from `summary.md` (if present)
- Available figures: list of PNGs in `analysis/`
- **The full content of `analysis/report.md`** as a string — for the appendix write-up.
- **The full content of `experiment_summary.yaml`** as a string — for the appendix details block.

Preserve full numeric precision when reading from `report.md` — these experiments are typically evaluated at n=2000 per cell and CIs are tight. Store the *source* number verbatim (e.g. `0.987`, `99.8%–100.0%`), don't round to `0.99`. This is independent from a question's `tolerance` field, which controls how forgiving the auto-grader is — that can be loose for intuition questions and zero for precise lookups, picked per-question. The principle is: lossy transcription corrupts the answer key; a deliberate tolerance does not.

### 3. Write the title and intro

The `title` field is what appears as the H1 and in the browser tab. **Don't use the experiment's directory name** (e.g. `length_sweep_nth14`) — it's a slug, not a title. Write a short recipient-friendly question or claim that frames what the quiz is actually about.

Good titles:
- "Are longer sequences harder to learn? An LLM fine-tuning intuition test."
- "How much training data does a small LLM need to memorise a rule?"
- "Does the model size make a difference for ACS-employment prediction?"

Bad titles:
- The directory slug (`length_sweep_nth14`)
- The full research question copied verbatim (too long, too jargon-heavy for a recipient)

Then draft the `intro` field of `quiz.json`. It must include:

- **The task** — what the model is doing in plain English. Don't assume domain knowledge. Define jargon (e.g. "implicit prompt: the rule is not described to the model; it must be learned from training examples").
- **One concrete example input → output**, formatted exactly as the model saw it. If the eval shows tokens like `4 8 1 3 7 ...` → `9`, show that. The recipient should be able to reverse-engineer the rule from the example.
- **What was varied** — list the experimental axes (e.g. "k = sequence length: {15, 20, 30, 50, 100}; N = training set size: {25, 100, 500, 2000}") with one-line definitions.
- **Baselines** — chance accuracy if the task has one (state it; don't make it a question), and any control conditions (base model, alternate rule).
- **What the quiz tests** — one sentence about the *kind* of intuition being probed.

Use markdown headings. The intro is the first thing the recipient reads — assume it's the only context they get.

### 4. Plan the question portfolio → `question_catalog.md`

Pick **6–8 questions, all about the results** (not about how the experiment was run, what went wrong operationally, or compute efficiency).

Default mix:
- **3 multiple_choice** — including at least one "did the data confirm the hypothesis?" question and one synthesis question ("which of these statements is best supported by the results?").
- **2 numerical_estimate** — predict a specific cell from the results table. Use percentage form (0–100) for accuracy, not decimal. Set `min`/`max`/`step` so out-of-range values are rejected.
- **1 ranking** — only if conditions actually differ on the metric being ranked. If all fine-tuned runs tied at 1.000, ranking by accuracy is degenerate; use training loss / sample efficiency instead, or replace with a multiple_choice.

If two experiments are loaded, at least one question must require both.

**Question ordering** matters. Order roughly:
1. **Anchor**: a concrete lookup the recipient can answer from the intro alone (warm-up, builds confidence).
2. **Predict**: numerical estimate with the answer derivable from the table.
3. **Twist**: the most surprising / non-monotonic finding (the lesson the experiment teaches that intuition wouldn't predict).
4. **Hypothesis check**: did the data confirm what the experiment set out to test?
5. **Predict (harder)**: a second numerical, or a comparison.
6. **Synthesis**: which conclusion is best supported?

Don't lead with anything that gives away a later answer.

### Question types to avoid by default

- **image_read** — the recipient lacks access to the original figure context. Only use if a graph shows something genuinely uncapturable in a table cell (rare).
- **equation_or_baseline** — chance baselines belong in the intro, stated. Don't make computing them a question unless the baseline derivation is itself the point of the experiment.
- **anomaly_hunt for *operational* anomalies** (SLURM failures, config bugs, compute issues) — out of scope. The variant for *results* anomalies (counter-examples to the hypothesis, surprising data points) is fine and lands as a multiple_choice.
- **free_text_intuition** — there is no auto-grade and no feedback channel back to the experimenter. Disabled by default. Only enable if the user explicitly says they have a way to collect responses.

For each question, write:
- The prompt — substantive, asking for a prediction or judgment, not just lookup
- The correct answer
- An explanation that quotes the relevant numbers / facts **inline**. Don't say "see the X section in the report" — the recipient can't open the report.

### 5. Write `quiz/quiz.json`

The canonical spec. Schema is documented in `generation.md`. Top level contains:
- `intro` (markdown, self-contained setup)
- `experiments` (footer credit, name-only)
- `questions` array
- `full_writeup_md` — the verbatim contents of `analysis/report.md`
- `full_writeup_image_dir` — absolute path to `analysis/` so the renderer can resolve image refs
- `experiment_summary_yaml` — the verbatim contents of `experiment_summary.yaml`

How the renderer places these:
- `experiment_summary_yaml` → a collapsed `<details>` panel labeled **"Full experimental specification"** placed *between the intro and the first question*. Recipients can read the full configuration before answering — it's part of the experimental setup, not the answer key.
- `full_writeup_md` → a `<details>` panel **hidden until the recipient clicks "See Final Score"**. It contains the answers and graphs, so it must not be available before submission. Once revealed, it auto-opens. PNG references inside the markdown are base64-embedded so the file remains portable.

Always include all three fields.

### 6. Render `quiz/quiz.html`

```bash
python -m cruijff_kit.tools.quiz.render_quiz \
  --spec {experiment_dir}/quiz/quiz.json \
  --out {experiment_dir}/quiz/quiz.html
```

The renderer embeds PNGs as base64 (when used) and inlines all CSS/JS so the resulting file is portable — the recipient can open `quiz.html` from a download folder with no server.

### 7. Logging → `logging.md`

Append to `{experiment_dir}/logs/create-quiz.log`. Action types: `LOCATE_EXPERIMENT`, `PARSE_INPUTS`, `WRITE_INTRO`, `PLAN_QUESTIONS`, `WRITE_SPEC`, `RENDER_HTML`.

## Why explanations must be self-sufficient

The skill's value depends on the recipient trusting the answer key. Hallucinated numbers destroy that trust on first inspection — and so does an answer key that says "see the report" when the recipient doesn't have the report. So every answer's explanation must:

1. **Quote the relevant fact inline** — the exact number, the exact rule, the exact comparison. The recipient should be able to read the explanation and learn the answer without opening any other file.
2. **Be grounded in a real artifact** — when *you* (the skill) write the explanation, the number you quote must come verbatim from `report.md` or `summary.md`. Don't paraphrase, don't round.

If you can't write a self-sufficient explanation, drop the question.

## Output structure

After running, the experiment directory will contain:

```
{experiment_dir}/
├── quiz/
│   ├── quiz.json     # Canonical spec
│   └── quiz.html     # Self-contained, self-grading
├── logs/
│   └── create-quiz.log
└── (existing files unchanged)
```

## User questions

### Existing quiz outputs

If `quiz/` already exists with a `quiz.html`, ask:

```
Found existing quiz in quiz/. What would you like to do?

1. Overwrite (Recommended)
2. Keep, write to quiz_v2/ instead
```

### Two-experiment confirmation

If the user passed two paths, briefly summarize what's in each (one-line research question per experiment) and confirm before generating, since two-experiment quizzes take longer and the user may have meant to pass just one.

## Validation before completion

Before reporting success, verify:
- ✓ Both experiment(s) had `analysis/report.md`.
- ✓ The `intro` field includes a task description, an example input → output, the experimental axes with definitions, and the chance baseline (if applicable).
- ✓ `quiz/quiz.json` validates against the schema in `generation.md` (6–8 questions, default mix, every question has an explanation).
- ✓ Every explanation is **self-sufficient** — it quotes the relevant numbers / facts inline and does not say "see report.md" or reference any external doc the recipient can't open.
- ✓ All numerical_estimate questions use percentage form (0–100) for accuracy values and have `min`/`max` set.
- ✓ No image_read, equation_or_baseline, anomaly_hunt-as-process, or free_text_intuition questions unless the user explicitly asked for them.
- ✓ `quiz/quiz.html` exists and is self-contained — grep for `(src|href)="http` and confirm the only external references are: the MathJax CDN, the polyfill.io CDN MathJax depends on, and the `github.com/niznik-dev/cruijff_kit` link in the brand banner. Everything else (CSS, JS, images, the full write-up's images, the YAML appendix) is inline / base64.
- ✓ The `full_writeup_md`, `full_writeup_image_dir`, and `experiment_summary_yaml` fields are populated. Verify the rendered quiz has the two collapsed `<details>` blocks at the bottom and that PNGs from `report.md` got embedded as base64 (grep `data:image/png;base64,` in the writeup section).
- ✓ For two-experiment mode: at least one question references both experiments by name.
- ✓ `logs/create-quiz.log` was appended to.

## Output summary

After completing, tell the user:

```
## Quiz created

Experiment: {experiment_name}
Questions: {N} ({breakdown by type})

Files:
- {experiment_dir}/quiz/quiz.html   ← open in browser
- {experiment_dir}/quiz/quiz.json
- {experiment_dir}/logs/create-quiz.log
```

End with the absolute path to `quiz.html` on its own line so the user can command-click it.

## Relationship to other skills

- **After:** `analyze-experiment` (mandatory — needs `report.md`)
- **Reads:** `experiment_summary.yaml`, `analysis/report.md`, `analysis/*.png`, `summary.md`
- **Creates:** `quiz/quiz.{json,html}`, `logs/create-quiz.log`

Workflow position:
```
design-experiment → scaffold-experiment → run-experiment → summarize-experiment → analyze-experiment → create-quiz
```

## Important notes

- The HTML is self-grading — there is no backend, no recipient tracking, no persistence. So default to question types whose answers can be auto-graded (multiple_choice, numerical_estimate, ranking).
- The renderer (`src/tools/quiz/render_quiz.py`) is dumb on purpose — all the question authoring happens in the LLM step. The renderer just pours JSON into a Jinja template.
- The skill still *supports* the deprecated types (image_read, equation_or_baseline, anomaly_hunt-as-process, free_text_intuition) in the renderer schema — they're available if a user explicitly asks for them. They're just not in the default question mix.

## Module organization

| Module | Purpose |
|--------|---------|
| `parsing.md` | Loading experiment_summary.yaml + report.md + summary.md |
| `question_catalog.md` | Types, when to use each, worked examples |
| `generation.md` | quiz.json schema + render command |
| `logging.md` | Action types and file location |

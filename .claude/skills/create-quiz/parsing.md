# Parsing: Locate Experiment(s) and Read Inputs

## Locating experiment directories

If the user passes paths, use them. If not, check the current directory:

```bash
ls {cwd}/experiment_summary.yaml
```

If found, treat `{cwd}` as the (single) experiment directory. If not, ask the user.

For each directory provided, verify:
- `experiment_summary.yaml` exists
- `analysis/report.md` exists (this skill requires `analyze-experiment` to have run; if missing, stop and tell the user)

## Loading `experiment_summary.yaml`

```python
import yaml
from pathlib import Path

with open(Path(experiment_dir) / "experiment_summary.yaml") as f:
    config = yaml.safe_load(f)
```

Pull these fields (all live under `experiment.`):
- `name` — display name
- `project` — for context
- `question` — the research question, used in the quiz intro
- `hypothesis` — a rich source of intuition questions ("the hypothesis predicted X — did the data agree?")
- `purpose` — reason the experiment exists, useful for "why does this matter" questions
- `date`

Also pull:
- `runs[]` — list of conditions, each with `name`, `model`, `parameters`
- `evaluation.matrix[]` — task names, vis_labels, epoch lists per run
- `variables` — independent variable axes (often the cleanest source for ranking and prediction questions)
- `controls` — fixed parameters (helpful for "what was held constant?" framing)

The hypothesis text often pre-states predictions in plain English (e.g. "k=15 saturates trivially by N=25 or N=100"). These are gold for "did the data confirm the hypothesis here?" questions.

## Loading `analysis/report.md`

The richest input. Read the whole file (typically a few hundred lines) and extract:

- **Executive summary** — one or two bolded headline numbers + best/worst performer
- **Model comparison table** — every numeric result, with full precision. Don't truncate.
- **Per-condition summary** (sometimes named "Per-k summary", "Saturation check", etc.) — these are pivot tables that question stems can read directly
- **Anomalies** section — operational quirks (failed submissions, wrong configs) that make great "spot the problem" questions
- **Suggested next steps** — what experiments would come next; useful for free-text intuition questions ("what experiment would you propose?")
- **Visualizations** — list of `analysis/*.png` referenced in the report

Don't paraphrase numbers. Store them as strings exactly as they appear (`"0.987"`, `"99.8%–100.0%"`) so the answer key matches what the recipient would see if they cracked open `report.md`.

## Loading `summary.md` (optional)

If present (`{experiment_dir}/summary.md` from `summarize-experiment`), use it for:
- Run status table (which runs completed)
- Final training loss per run
- Quick accuracy table
- Datasets used

This data overlaps with `report.md` but is sometimes formatted more usefully (e.g. training loss only appears in `summary.md`, not `report.md`).

## Discovering figures

```python
analysis_dir = Path(experiment_dir) / "analysis"
pngs = sorted(analysis_dir.glob("*.png"))
```

Each PNG is a candidate for an image-read question. Skip the question type if no PNGs exist — don't try to synthesize one.

When embedding into the quiz, the renderer will read the PNG bytes and base64-encode them. You only need to record the file path in `quiz.json`.

## Two-experiment mode

Run all of the above for each experiment independently and store the results under separate keys (e.g. `experiment_a`, `experiment_b`). When planning questions, deliberately allocate at least one comparison question that requires both — e.g. "Both experiments tested k=20 with N=500. Which got higher accuracy?" or "Experiment A's hypothesis predicted X. Did experiment B's results agree?".

## Error handling

- **Missing `experiment_summary.yaml`:** stop, tell the user the directory doesn't look like a cruijff_kit experiment.
- **Missing `analysis/report.md`:** stop, suggest running `analyze-experiment` first.
- **Malformed YAML:** report the error, don't try to recover.
- **No PNGs:** continue, just don't generate image-read questions.

## Logging

Log a `LOCATE_EXPERIMENT` entry per directory found, then a `PARSE_INPUTS` entry summarizing what was extracted (number of runs, number of report rows, number of PNGs). See `logging.md`.

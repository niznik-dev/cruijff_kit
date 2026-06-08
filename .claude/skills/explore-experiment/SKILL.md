---
name: explore-experiment
description: Open-ended, Claude-driven exploration of a completed experiment — you decide per-experiment which figures matter and what the results mean. Optional and runnable any time after run-experiment (summarize-experiment is the required post-run step). Produces "Claude's Exploration": a non-deterministic report of what you examined and concluded, with an audit log.
---

# Explore Experiment

You are Claude, exploring a completed experiment. **You** decide — for this specific experiment — which figures are worth making and what the results mean. There is no fixed plot menu: the value of this step is your per-experiment judgment, not a deterministic pipeline.

The artifact you produce is **"Claude's Exploration"** — a deliberately non-deterministic record of what you looked at and concluded *this time*. The human usually does not intervene while you work, so the name is honest about what this is: your exploration, not a reproducible deterministic report. To keep it auditable anyway (Principle #1: Scientific), you log what you did.

## The summarize / explore boundary

`summarize-experiment` is the required, script-driven, deterministic backbone — the always-correct facts that hold for *every* experiment (run status, loss, accuracy with CIs, provenance including eval-set class balance, and the structure-free cross-cell fact of saturation). **Read its `summary.md` first.**

> **If something is meaningful for *every* experiment, it belongs to `summarize`. Everything else is yours.**

You *interpret* the facts summarize establishes — which figures matter here, what the results mean, what to do next. You do not recompute the backbone.

## Your Task

1. Read `summary.md` (from summarize-experiment) and `experiment_summary.yaml` — understand the question / hypothesis and the deterministic facts summarize already established.
2. Decide what, if anything, is worth visualizing for *this* experiment (see "Choose what to show" — it is a toolkit, not a checklist).
3. Interpret the results to the bar in "Interpret — clear the bar."
4. Write "Claude's Exploration" (`exploration/report.md`) and an audit log (`logs/explore-experiment.log`).

## Prerequisites

- experiment_summary.yaml exists (from design-experiment)
- Evaluations complete (from run-experiment)
- Plotting toolkit installed — inspect-viz, matplotlib, seaborn, plotly are all bundled by `pip install -e .`
- Conda environment activated (from claude.local.md)
- **Optional:** playwright for PNG export (`pip install playwright && playwright install chromium`)

## Dependency version check

Run before proceeding to catch stale envs (user pulled new pins but didn't re-run `pip install -e .`):

```bash
python scripts/check_env.py
```

- **Exit 0**: proceed.
- **Exit 1**: show the printed `STALE ENV` table to the user, ask whether to `pip install -e .` first or continue anyway.

## Workflow

### 1. Locate Experiment → `parsing.md`

Find experiment directory and parse experiment_summary.yaml for:
- Experimental variables (model, task, factors)
- Run structure and naming conventions
- Evaluation matrix

**See `parsing.md` for:**
- How to find the experiment directory
- YAML parsing logic
- Extracting variable types for inference

### 2. Load Evaluation Logs → `data_loading.md`

Use helper functions from `src/tools/inspect/viz_helpers.py`:
- `deduplicate_eval_files()` - Remove duplicate evals (keeps most recent per model+epoch)
- `evals_df_prep()` - Prepare eval-level dataframes
- `parse_eval_metadata()` - Extract epoch/finetuned/source_model from JSON metadata
- `detect_metrics()` - Dynamically detect available score columns

**See `data_loading.md` for:**
- Automatic deduplication of re-run evaluations
- Generic metadata extraction using vis_label and JSON metadata
- How to construct subdirs list from experiment_summary.yaml
- Preparing data for each view type

### 3. Choose what to show — a toolkit, not a menu

There is no fixed plot menu. Decide what, if anything, is worth showing for *this* experiment. Generating **zero** plots is a valid outcome if `summary.md` already tells the story; so is a bespoke figure no pre-built view covers. But when a result has any shape worth seeing, lean toward showing it — a figure the reader can scan beats a sentence describing one, and a near-empty report under-serves them. Err toward the figure when in doubt. Two craft habits worth applying as you choose (detail in `generation.md` → "Figure craft"): prefer a figure that shows the *mechanism* — the raw quantity the metrics are computed from — over one that re-expresses a metric you already have; and use color to carry a dimension (map an ordinal axis to a sequential colormap) rather than defaulting to the sparest palette.

The environment ships four plotting tools — peers chosen by fit, not a ranking:

| Tool | Reach for it when… | Output |
|------|---------------------|--------|
| **inspect-viz** | a pre-built eval view fits as-is (scores by task / model / factor, heatmap, radar) | interactive HTML |
| **matplotlib** | you want full control of a static figure (already how the ROC / calibration / histogram overlays are built) | PNG |
| **seaborn** | statistical cuts with good defaults — distributions, regressions, annotated heatmaps | PNG (matplotlib-backed) |
| **plotly** | you want custom interactivity a pre-built inspect-viz view can't give | interactive HTML |

`pandas` (`df.plot`, `df.style`), `scipy`, and `scikit-learn` are also on hand for quick cuts and the stats beneath the plots. Nothing here is mandatory or exhaustive — if something else in the environment fits the question better, use it.

### 4. Generate the figures → `generation.md`

Build whatever you chose. `generation.md` holds reference *recipes* — not a required sequence. A bespoke matplotlib / seaborn / plotly figure written from scratch is as valid as a pre-built view.

**See `generation.md` for:**
- inspect-viz pre-built view signatures and parameters
- matplotlib overlay helpers (ROC, calibration, prediction histogram) for `risk_scorer` evals
- Dynamic metric detection, PNG export, and output-file naming

### 5. Interpret — clear the bar → `generation.md`

**Required.** Interpretation is the heart of explore: a hypothesis-first re-read of the data, not a restatement of the tables. Write five sections, each **mandatory or an explicit "n/a — reason"** (no silent skips):

1. **What the data does, vs. what we expected** — decompose `experiment.hypothesis` into individual falsifiable claims and walk through them one at a time, each as its own short paragraph: describe the phenomenon and mechanism first, anchor with a one-word verdict (Confirmed / Violated / Inconclusive) plus cell-level evidence. A compact table is fine as at-a-glance reference, but the prose must stand alone (repeat every number/verdict in the narrative) — tables are reference, never the focal artifact. Lead with description; chase the pattern nobody pre-registered. If there is no hypothesis, infer predictions from `experiment.question` + `variables` and flag the gap. (See `generation.md` → "Describe, don't just adjudicate".)
2. **Cross-cell pattern audit** — scan the results grid, split by what a script can honestly do:
   - **Saturation** (accuracy ≥0.95) is pure counting — structure-free, true for every experiment. This is summarize's to compute; `summary.md` carries it, so read it from there instead of recomputing.
   - **Base-rate floor** is *not* structure-free. The arithmetic is — the eval-set class balance, which summarize reports as provenance — but whether that split is the *meaningful* baseline depends on intent. A deliberately balanced split's 50/50 says nothing; the truer floor in a fine-tuning experiment is usually the base-model eval, not label prevalence. Reading the class balance is free; deciding what the floor actually is stays your judgment, made here.
   - **Monotonicity** (does a metric move one direction as an *ordinal* variable climbs?) and **equivalent-cell agreement** (do cells the design means to be the same actually agree?) depend on knowing your experiment's shape — which variables are ordered, which cells are meant to match. A generic script would have to guess that, so they stay your judgment: make the calls here.
3. **Mechanistic interpretation** — what each variable's variation actually tests; a parsimonious explanation for each surprise; name confounds explicitly.
4. **Self-consistency audit** — check every numerical claim in your prose against the source table (`claim → table value`), literally, not interpretively.
5. **Calibrated next steps** — each specifies what to vary / what mechanism it tests / what outcome is informative. "The design didn't actually test X — fix the design first" is a valid item.

**Adversarial pre-pass:** before reading the data, state from the hypothesis alone what cell-level evidence *would* falsify each claim — then check whether it appears. This guards against post-hoc rationalization.

Record what you concluded — and why each figure was worth making — in the audit log (`logs/explore-experiment.log`). Only omit interpretation if the user explicitly asks for plots-only.

### 6. Write Report → `generation.md`

**Author `exploration/report.md` yourself** — there is no report-assembling
function. Compose the markdown directly (prose, inline figures, provenance), the
way you wrote `summary.md`:
1. Open with a jargon-free **Bottom line**, then the five interpretation sections.
2. **Cite `../summary.md`** for the canonical metric tables — don't re-render
   them; quote only the specific numbers your argument turns on.
3. Embed figures **inline with descriptive captions** at the point you discuss
   them (no dumped "Visualizations" appendix).
4. Close with a **summarized** provenance footer (count + glob, not every
   absolute path) in a collapsible `<details>` block.

See `generation.md` → "Report Generation" for the full shape and rationale.

### 6b. Compute Utilization Report → `generation.md`

Generate a compute utilization section. Always call `harvest_jids_from_run_logs(experiment_dir)` from `src/tools/slurm/compute_metrics.py` first — it returns `(jids_dict, warnings)`:

1. Call `harvest_jids_from_run_logs(experiment_dir)`. It already prints `WARNING:` lines to stderr for any missing or malformed log files.
2. **If `warnings` is non-empty**: append a visible "**Compute Utilization unavailable:** ..." note to the rendered report listing each warning, instead of silently skipping. Do NOT omit the section header. (Closes the silent-skip gap from issue #451.)
3. **If JIDs are present**: for each, run `seff` and parse with helpers in `src/tools/slurm/compute_metrics.py`; read `gpu_metrics.csv` per run and summarize with `summarize_gpu_metrics()`; format with `format_compute_table()`; save raw metrics to `exploration/compute_metrics.json`; pass the formatted table as `compute_section` to `generate_report()`.

**Loud-warn, do not silently skip.** When the logs are genuinely absent (e.g., analyzing a colleague's experiment without local logs), the warning text in `report.md` tells the operator how to recover (typically: re-run `run-experiment`, or re-create the canonical log via `src/tools/run/submit_*.py`).

### 7. Logging → `logging.md`

Document the process in `{experiment_dir}/logs/explore-experiment.log`. This log is what keeps a deliberately non-deterministic step auditable (Principle #1): record which data you read, which figures you chose to make and **why**, and what you concluded.

**See `logging.md` for:**
- Plain text format specification
- Action types (`LOAD_EVALS`, `CHOOSE_FIGURE`, `GENERATE_PLOT`, `RECORD_INTERPRETATION`, …)
- Example log entries

---

## inspect-viz pre-built views (one tool's catalog)

These are the ready-made views **inspect-viz** offers — convenient when one fits as-is, but only one of the four tools in "Choose what to show." For anything they don't cover, drop to matplotlib / seaborn / plotly.

| View | Use Case | Required Data |
|------|----------|---------------|
| `scores_by_task` | Multiple tasks/conditions | task_name column |
| `scores_heatmap` | Model × task matrix | model + task columns |
| `scores_radar_by_task` | Multiple metrics | Multiple score columns |
| `scores_by_factor` | Binary factors | Boolean factor column |
| `scores_by_model` | Cross-model comparison (single task only) | model column, single task |

**Note:** `scores_by_model` requires a single-task experiment. For multi-task experiments, use `scores_by_task`, `scores_heatmap`, or `scores_radar_by_task` instead.

## User Questions

### Existing Exploration Outputs

If `exploration/` directory exists with files, **ask the user**:

```
Found existing exploration outputs in exploration/. What would you like to do?

1. Keep existing files, add new outputs (Recommended)
2. Clean exploration directory first
```

If user chooses option 2, delete contents of `exploration/` before generating new outputs.

### Visualization Selection

When `vis_label` creates multiple task variants (conditions), **ask the user** which visualization to generate:

```
Found {N} conditions via vis_label: {list}

Which visualization would you like?

1. scores_by_task - Compare conditions side-by-side (Recommended)
2. scores_heatmap - Model × condition matrix
3. Both
```

### Tracking Generated Files

**Important:** Track which PNG files you generate during this run, and embed only those inline in `report.md` — never stale outputs from a previous session.

```python
generated_pngs = []

# After each successful PNG export
generated_pngs.append(png_path)

# Embed only these inline in report.md (with descriptive captions)
```

**Smart defaults for everything else:**
- Deduplication: Automatic (keep most recent per model+epoch)
- Epochs: Use latest only (unless user explicitly asks for all)
- Metrics: Plot all available (dynamically detected)
- PNG: Auto-detect playwright; generate if available, skip with warning if not

## Output Structure

After running, the experiment directory will contain:

```
{experiment_dir}/
├── exploration/
│   ├── report.md               # Markdown report with metrics
│   ├── scores_by_task.html
│   ├── scores_by_task.png      (if playwright available)
│   ├── scores_heatmap.html
│   ├── scores_heatmap.png      (if playwright available)
│   ├── roc_curves.png          (if risk_scorer used)
│   ├── calibration_curves.png  (if risk_scorer used)
│   ├── prediction_histogram.png (if risk_scorer used)
│   └── ...
├── logs/
│   └── explore-experiment.log
└── experiment_summary.yaml
```

## Error Handling

**If experiment_summary.yaml not found:**
- Report error to user
- Suggest running design-experiment skill first
- Do not proceed

**If no evaluation logs found:**
- Report error to user
- Suggest running run-experiment skill first
- Do not proceed

**If visualization generation fails:**
- Log error details in explore-experiment.log
- Continue with remaining visualizations
- Report which visualizations failed in summary

**If inference cannot determine view type:**
- Log warning
- Ask user which view to generate
- Or skip and note in summary

## Validation Before Completion

Before reporting success, verify:
- ✓ experiment_summary.yaml was found and parsed
- ✓ Evaluation logs were loaded successfully
- ✓ At least one visualization was generated
- ✓ HTML files exist in exploration/ directory
- ✓ report.md was generated in exploration/ directory
- ✓ Log file created (logs/explore-experiment.log)

## Output Summary

After completing analysis, provide a summary:

```markdown
## Explore Experiment Complete

Experiment: `{experiment_dir}`

### Report Generated

✓ Markdown report: `exploration/report.md`
  - Executive summary with best performer
  - Model comparison table with 95% CIs
  - Calibration metrics (if available)

### Visualizations Generated

✓ {N} visualizations created in `exploration/`

**Created:**
- scores_by_task.html - Task comparison (match accuracy)
- scores_heatmap.html - Model × task matrix
- scores_by_model.html - Model comparison

### Viewing Results

Open HTML files in a browser:
```bash
open {experiment_dir}/exploration/scores_by_task.html
```

Or start a local server:
```bash
cd {experiment_dir}/exploration && python -m http.server 8080
```

### Log

Details recorded in `logs/explore-experiment.log`

### Report Path

`{experiment_dir}/exploration/report.md`
```

**IMPORTANT:** Always end your output summary with the **full absolute path** to `report.md` on its own line, so the user can command-click it in their terminal/IDE. Example:

```
Full report: /scratch/gpfs/user/ck-projects/{project}/my_experiment/exploration/report.md
```

## Relationship to Other Skills

- **After:** run-experiment — optional, and can be run any time afterward. (`summarize-experiment` is the required post-run step; explore-experiment is an optional deeper pass.)
- **Reads:** experiment_summary.yaml, .eval files
- **Creates:** HTML visualizations, analysis log

**Workflow position:**
```
design-experiment → scaffold-experiment → run-experiment → summarize-experiment → (optional) explore-experiment
```

## Important Notes

- **Metadata-driven** - uses `vis_label` metadata for task names, JSON metadata for epoch/finetuned/source_model
- **Automatic deduplication** - when multiple evals exist for the same model+epoch, keeps only the most recent (logs skipped files)
- **Dynamic metric detection** - plots all available score columns (no hardcoding)
- **Must run after evaluations complete** - requires .eval files to exist
- Visualizations are independent - one failing doesn't stop others
- Output directory (`exploration/`) is created if it doesn't exist
- HTML files are standalone (no external dependencies to view)
- PNG files require playwright (skipped with warning if not installed)
- Uses helper functions from `src/tools/inspect/viz_helpers.py`

## Module Organization

| Module | Purpose |
|--------|---------|
| parsing.md | Experiment location and YAML parsing |
| data_loading.md | Helper functions for loading logs |
| generation.md | Plot creation workflow |
| logging.md | Plain text audit trail specification |

## Implementation Reference

Helper functions are implemented in `src/tools/inspect/viz_helpers.py`. Live usage is shown inline throughout `generation.md`.

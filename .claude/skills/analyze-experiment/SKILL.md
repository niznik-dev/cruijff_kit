---
name: analyze-experiment
description: Generate visualizations from completed experiment evaluations using inspect-viz. Use after run-experiment to create interactive HTML plots from inspect-ai evaluation logs.
---

# Analyze Experiment

You help users visualize and analyze results from completed experiments by generating interactive HTML and PNG plots using inspect-viz pre-built views.

## Your Task

Generate visualizations from evaluation results:

1. Read experiment_summary.yaml to understand experimental design
2. Load evaluation logs from run directories
3. Infer appropriate visualization types based on experimental variables
4. Generate interactive HTML plots using inspect-viz
5. Log the process in analyze-experiment.log

## Prerequisites

- experiment_summary.yaml exists (from design-experiment)
- Evaluations complete (from run-experiment)
- inspect-viz installed (`pip install inspect-viz`)
- Conda environment activated (from claude.local.md)
- **Optional:** playwright for PNG export (`pip install playwright && playwright install chromium`)

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

Use helper functions from `tools/inspect/viz_helpers.py`:
- `deduplicate_eval_files()` - Remove duplicate evals (keeps most recent per model+epoch)
- `evals_df_prep()` - Prepare eval-level dataframes
- `parse_eval_metadata()` - Extract epoch/finetuned/source_model from JSON metadata
- `detect_metrics()` - Dynamically detect available score columns

**See `data_loading.md` for:**
- Automatic deduplication of re-run evaluations
- Generic metadata extraction using vis_label and JSON metadata
- How to construct subdirs list from experiment_summary.yaml
- Preparing data for each view type

### 3. Select Visualizations

Map experimental variables to appropriate pre-built views:

| Variable Type | View |
|---------------|------|
| Multiple models | `scores_by_model` |
| Binary factor | `scores_by_factor` |
| Multiple tasks/conditions | `scores_by_task` |
| Model × task matrix | `scores_heatmap` |
| Multiple metrics | `scores_radar_by_task` |

### 4. Generate Plots → `generation.md`

Create visualizations using inspect-viz:
1. Wrap dataframes with `Data.from_dataframe()`
2. Call appropriate pre-built view functions
3. Save HTML with `write_html()`
4. Auto-export PNG with `write_png()` if playwright is available

**See `generation.md` for:**
- Dynamic metric detection
- View function signatures and parameters
- PNG export with playwright auto-detection
- Output file naming conventions
- Creating multiple plots per experiment

### 5. Generate Analysis & Interpretation → `generation.md`

**This step is required by default.** Write a `future_directions` string that interprets the results and suggests next steps. This section should:

1. Interpret the key findings (what do the metrics mean for the research question?)
2. Note any surprises or anomalies in the results
3. Suggest concrete next experiments or analyses

Only omit this section if the user explicitly asks to skip it.

### 6. Generate Report → `generation.md`

Create markdown report with metrics and comparisons:
1. Extract metrics from evaluation dataframe
2. Compute Wilson score confidence intervals
3. Generate narrative summary and comparison tables
4. Include `future_directions` from step 5
5. Write report to `analysis/report.md`

Uses `tools/inspect/report_generator.py`:
```python
from tools.inspect.report_generator import generate_report

report = generate_report(
    df=logs_df,
    experiment_name=experiment_name,
    output_path=Path(experiment_dir) / "analysis" / "report.md",
    config=experiment_config,  # Optional, for research question metadata
    future_directions=future_directions,  # From step 5
)
```

### 7. Logging → `logging.md`

Document process in `{experiment_dir}/analyze-experiment.log`

**See `logging.md` for:**
- JSONL format specification
- Action types (LOCATE, PARSE, LOAD, INFER, GENERATE)
- Example log entries

---

## Supported Pre-built Views

| View | Use Case | Required Data |
|------|----------|---------------|
| `scores_by_task` | Multiple tasks/conditions | task_name column |
| `scores_heatmap` | Model × task matrix | model + task columns |
| `scores_radar_by_task` | Multiple metrics | Multiple score columns |
| `scores_by_factor` | Binary factors | Boolean factor column |
| `scores_by_model` | Cross-model comparison (single task only) | model column, single task |

**Note:** `scores_by_model` requires a single-task experiment. For multi-task experiments, use `scores_by_task`, `scores_heatmap`, or `scores_radar_by_task` instead.

## User Questions

### Existing Analysis Outputs

If `analysis/` directory exists with files, **ask the user**:

```
Found existing analysis outputs in analysis/. What would you like to do?

1. Keep existing files, add new outputs (Recommended)
2. Clean analysis directory first
```

If user chooses option 2, delete contents of `analysis/` before generating new outputs.

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

**Important:** Track which PNG files you generate during this run. Only pass those to `generate_report()` so the report embeds only the visualizations created in this session, not stale outputs from previous runs.

```python
generated_pngs = []

# After each successful PNG export
generated_pngs.append(png_path)

# Pass to report generator
generate_report(..., generated_pngs=generated_pngs)
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
├── analysis/
│   ├── report.md               # Markdown report with metrics
│   ├── scores_by_task.html
│   ├── scores_by_task.png      (if playwright available)
│   ├── scores_heatmap.html
│   ├── scores_heatmap.png      (if playwright available)
│   ├── roc_curves.png          (if risk_scorer used)
│   ├── calibration_curves.png  (if risk_scorer used)
│   ├── prediction_histogram.png (if risk_scorer used)
│   └── ...
├── analyze-experiment.log
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
- Log error details in analyze-experiment.log
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
- ✓ HTML files exist in analysis/ directory
- ✓ report.md was generated in analysis/ directory
- ✓ Log file created (analyze-experiment.log)

## Output Summary

After completing analysis, provide a summary:

```markdown
## Analyze Experiment Complete

Experiment: `{experiment_dir}`

### Report Generated

✓ Markdown report: `analysis/report.md`
  - Executive summary with best performer
  - Model comparison table with 95% CIs
  - Calibration metrics (if available)

### Visualizations Generated

✓ {N} visualizations created in `analysis/`

**Created:**
- scores_by_task.html - Task comparison (match accuracy)
- scores_heatmap.html - Model × task matrix
- scores_by_model.html - Model comparison

### Viewing Results

Open HTML files in a browser:
```bash
open {experiment_dir}/analysis/scores_by_task.html
```

Or start a local server:
```bash
cd {experiment_dir}/analysis && python -m http.server 8080
```

### Log

Details recorded in `analyze-experiment.log`
```

## Relationship to Other Skills

- **After:** run-experiment, summarize-experiment
- **Reads:** experiment_summary.yaml, .eval files
- **Creates:** HTML visualizations, analysis log

**Workflow position:**
```
design-experiment → scaffold-experiment → run-experiment → summarize-experiment → analyze-experiment
```

## Important Notes

- **Metadata-driven** - uses `vis_label` metadata for task names, JSON metadata for epoch/finetuned/source_model
- **Automatic deduplication** - when multiple evals exist for the same model+epoch, keeps only the most recent (logs skipped files)
- **Dynamic metric detection** - plots all available score columns (no hardcoding)
- **Must run after evaluations complete** - requires .eval files to exist
- Visualizations are independent - one failing doesn't stop others
- Output directory (`analysis/`) is created if it doesn't exist
- HTML files are standalone (no external dependencies to view)
- PNG files require playwright (skipped with warning if not installed)
- Uses helper functions from `tools/inspect/viz_helpers.py`
- Reference examples in `viz_examples/scripts/inspect_viz_examples.ipynb`

## Module Organization

| Module | Purpose |
|--------|---------|
| parsing.md | Experiment location and YAML parsing |
| data_loading.md | Helper functions for loading logs |
| generation.md | Plot creation workflow |
| logging.md | JSONL audit trail specification |

## Implementation Reference

Working examples exist in `viz_examples/scripts/inspect_viz_examples.ipynb` demonstrating all supported pre-built views.

Helper functions are implemented in `tools/inspect/viz_helpers.py`.

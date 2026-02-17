# Generation: Creating Visualizations

This module describes how to generate visualizations using inspect-viz pre-built views.

## Overview

The generation workflow:

1. Wrap dataframe in `Data.from_dataframe()`
2. Call the appropriate view function with parameters
3. Save HTML with `write_html()`

## Setup

```python
import os
from inspect_viz import Data
from inspect_viz.plot import write_html
from inspect_viz.view.beta import (
    scores_by_task,
    scores_heatmap,
    scores_radar_by_task,
    scores_radar_by_task_df,
    scores_by_model,
    scores_by_factor,
)
from tools.inspect.viz_helpers import sanitize_columns_for_viz

# Create output directory
output_dir = os.path.join(experiment_dir, "analysis")
os.makedirs(output_dir, exist_ok=True)

# IMPORTANT: sanitize column names before passing to inspect-viz.
# Metric names with "/" (e.g. risk_scorer_cruijff_kit/auc_score) cause
# DuckDB parse errors in inspect-viz. Use the sanitized df for all
# Data.from_dataframe() calls; use the original df for report generation.
viz_df = sanitize_columns_for_viz(logs_df)
data = Data.from_dataframe(viz_df)
```

## View Function Signatures

**Before using these functions**, run `help()` to verify the API hasn't changed:

```python
from inspect_viz.view.beta import scores_by_task
help(scores_by_task)
```

The signatures below are reference examples — the actual function may have additional or renamed parameters.

### scores_by_task

Compare scores across multiple tasks or conditions.

```python
plot = scores_by_task(
    data,                           # Data object
    task_name='task_name',          # Column with task names
    score_value="score_match_accuracy",  # Score column
    score_stderr="score_match_stderr",   # Standard error column (optional)
    score_label="Match Accuracy",   # Label for y-axis
    ci=0.95                         # Confidence interval
)
```

### scores_heatmap

Model × task matrix visualization.

**When to skip:** If each model appears in only one task (1:1 mapping between
model and vis_label), the heatmap is just a diagonal — skip it. Check before
generating:

```python
# Only generate heatmap when multiple models share tasks
models_per_task = logs_df.groupby('task_name')['model'].nunique()
if models_per_task.max() > 1:
    plot = scores_heatmap(
        data,
        task_name='task_name',
        model_name="model_display_name",
        model_label="Model",
        score_value="score_match_accuracy",
        tip=True,
        title="",
        orientation="vertical"
    )
else:
    print("Skipping heatmap: each model maps to a single task (diagonal matrix)")
```

### scores_radar_by_task

Radar plot for multiple metrics comparison.

```python
# First prepare the data
radar_df = scores_radar_by_task_df(
    logs_df,                        # Original dataframe (not Data object)
    invert=["score_match_accuracy", "score_includes_accuracy"],  # Scores to include
    normalization="min_max",        # "min_max", "percentile", or "absolute"
    domain=(0, 1)                   # Domain for normalization
)

# Then create the plot
plot = scores_radar_by_task(Data.from_dataframe(radar_df))
```

### scores_by_factor

Compare binary factor effects.

```python
plot = scores_by_factor(
    data,
    factor="prompt_type",           # Boolean column
    factor_labels=("No Prompt", "Prompt"),  # Labels for False, True
    score_value="score_match_accuracy",
    score_stderr="score_match_stderr",
    score_label="Match Accuracy",
    model="model",                  # Column for grouping by model
    model_label="Model",
    ci=0.95
)
```

### scores_by_model

Compare models directly. **Requires single-task experiments only.**

```python
plot = scores_by_model(
    data,
    score_value="score_match_accuracy",
    score_stderr="score_match_stderr",
    score_label="Match Accuracy",
    ci=0.95,
    height=200                      # Plot height in pixels
)
```

**Note:** This view will fail if the data contains multiple tasks. For multi-task experiments, use `scores_heatmap` or `scores_by_task` instead.

## Output File Naming

Use descriptive names that indicate the view type and content:

```python
# Pattern: {view_type}_{experiment_type}_{metric}.html

# Examples:
"scores_by_task_sample_size_match.html"
"scores_by_task_balanced_match.html"
"scores_heatmap_sample_size_match.html"
"scores_by_factor_balanced_match.html"
"scores_by_model_match.html"
"scores_radar_sample_size.html"
```

## Dynamic Metric Detection

Instead of hardcoding metrics, detect them from the dataframe:

```python
from tools.inspect.viz_helpers import detect_metrics, display_name

# Automatically detect available metrics
detected = detect_metrics(logs_df)
# detected.accuracy -> e.g., ['match', 'includes']
# detected.supplementary -> e.g., ['risk_scorer_cruijff_kit/ece', ...]

# Generate plots for accuracy metrics
for metric in detected.accuracy:
    score_col = f"score_{metric}_accuracy"
    # ... create accuracy plot

# Generate plots for supplementary metrics (calibration, risk)
# NOTE: use sanitized column names (/ -> __) for inspect-viz score_value,
# but the original names for display_name() lookups.
for metric in detected.supplementary:
    score_col = f"score_{metric}".replace("/", "__")  # sanitized for viz
    label = display_name(metric)  # "ECE", "Brier Score", "AUC", etc.
    # These work with any view that accepts score_value parameter
    plot = scores_by_task(data, task_name='task_name', score_value=score_col, score_label=label)
```

## PNG Export

In addition to HTML, export PNG snapshots for reports and presentations:

```python
from inspect_viz.plot import write_html, write_png

# Always save HTML
write_html(html_path, plot)
print(f"  HTML saved: {html_path}")

# Auto-detect playwright and export PNG if available
try:
    from playwright.sync_api import sync_playwright
    write_png(png_path, plot)
    print(f"  PNG saved: {png_path}")
except ImportError:
    print("  PNG skipped: playwright not installed (pip install playwright && playwright install chromium)")
```

**Note:** PNG export requires playwright with chromium. If not installed, PNG generation is skipped with a warning.

## Error Handling

**If view function fails:**
- Log error with traceback
- Continue with other views
- Report failed views in summary

**If output directory is not writable:**
- Log error
- Suggest alternative location
- Do not proceed with generation

**If data is missing required columns:**
- Log which columns are missing
- Skip views that require those columns
- Report in summary

## Per-Sample Risk Plots (ROC & Calibration)

When the experiment used `risk_scorer`, generate overlay plots from per-sample data. These are **matplotlib PNGs** (not inspect-viz HTML), gated on `detected.has_risk_scorer`:

```python
from tools.inspect.viz_helpers import (
    extract_per_sample_risk_data, generate_roc_overlay,
    generate_calibration_overlay, generate_prediction_histogram,
)

if detected.has_risk_scorer:
    risk_data = extract_per_sample_risk_data(kept)  # kept = deduplicated .eval paths
    if risk_data:
        roc_path = generate_roc_overlay(risk_data, f"{output_dir}/roc_curves.png")
        cal_path = generate_calibration_overlay(risk_data, f"{output_dir}/calibration_curves.png")
        hist_path = generate_prediction_histogram(risk_data, f"{output_dir}/prediction_histogram.png")
        if roc_path:
            generated_pngs.append(str(roc_path))
        if cal_path:
            generated_pngs.append(str(cal_path))
        if hist_path:
            generated_pngs.append(str(hist_path))
```

**Performance note:** `extract_per_sample_risk_data()` reads full eval logs (all samples), so it's slower than the aggregate `evals_df_prep()` path. For a 5000-sample eval this typically takes a few seconds per file.

**Skipped models:** Models with <2 valid samples or only one class (e.g., zeroshot models that always predict the same thing) are silently skipped. The function logs which models were skipped at INFO level.

## Analysis & Interpretation (Required)

**This step is required by default.** After generating visualizations and before calling `generate_report()`, write a `future_directions` string that interprets results and suggests next steps.

The analysis should cover:

1. **Key findings**: What do the metrics mean for the research question in experiment_summary.yaml? Reference the hypothesis if one exists.
2. **Surprises or anomalies**: Anything unexpected — large gaps between models, metrics that disagree, base rates that explain accuracy, etc.
3. **Concrete next steps**: Suggest 2-4 specific follow-up experiments or analyses. Be actionable, not vague.

Example:
```python
future_directions = """
### Key Findings

Fine-tuning dramatically improved both models from 0% zero-shot to 75-78% accuracy,
confirming the hypothesis. The instruct model's edge (78.5% vs 0.2%) appears to come
from its ability to format text output, not from better discrimination — the base
model actually achieves higher AUC (0.883 vs 0.876).

### Anomalies

The base fine-tuned model has near-zero text accuracy despite excellent AUC, suggesting
it learned the probability distribution but not the output format. This is expected
for base models without chat template training.

### Suggested Next Steps

1. **Compare with 1B/3B results**: The 8B instruct model (78.5%) barely outperforms
   the 3B (76.6%) — check if scaling to 8B is worth the 4x compute cost.
2. **Calibration deep-dive**: ECE of 0.019 for instruct is surprisingly good —
   verify this holds with the corrected ECE metric on fresh evals.
3. **Increase training data**: All models trained on 40K samples. Try 100K to see
   if the accuracy plateau breaks.
"""
```

Only omit this section if the user explicitly asks to skip it (e.g., "just generate the plots, no analysis").

## Report Generation

After generating visualizations and writing the analysis, create a markdown report:

```python
from pathlib import Path
from tools.inspect.report_generator import generate_report

# Track PNGs generated during this run
generated_pngs = []

# After each successful PNG export:
# generated_pngs.append(png_path)

# Generate report after visualizations
report = generate_report(
    df=logs_df,                              # DataFrame from evals_df_prep + parse_eval_metadata
    experiment_name=experiment_name,          # e.g., "acs_income_2026-01-29"
    output_path=Path(output_dir) / "report.md",
    config=experiment_config,                 # Optional: experiment_summary.yaml dict
    generated_pngs=generated_pngs,            # Only embed PNGs from this run
    eval_log_paths=eval_log_paths,            # List of .eval file Paths used
    generated_by="Claude Opus 4.6",           # Attribution (use actual model name)
    future_directions=future_directions,      # Analysis from previous step
)
```

**Important:** Pass `generated_pngs` to ensure the report only embeds visualizations created in this session, not stale outputs from previous runs.

### Report Contents

The generated `report.md` includes:

| Section | Description |
|---------|-------------|
| Executive Summary | Best performer and model count |
| Model Comparison | Table with accuracy, 95% CI, sample size |
| Calibration & Risk Metrics | ECE, Brier, AUC, Mean Risk Score (if supplementary metrics detected) |
| Per-Task Breakdown | Best model per task (if multiple tasks) |
| Analysis & Interpretation | Key findings, anomalies, and suggested next steps |
| Provenance | Attribution and source eval log paths (collapsible) |

### Confidence Intervals

Uses Wilson score intervals (preferred over normal approximation):
- Never produces intervals outside [0, 1]
- Works well with small samples and extreme proportions

### Error Handling

**If metrics extraction fails:**
- Log error
- Report generation skipped with warning
- Visualizations still generated

## Compute Utilization Analysis (Optional)

After generating visualizations and before the report, optionally add compute metrics. This requires run logs from `run-experiment`.

### Extracting Job IDs

```python
import re
import json
from pathlib import Path
from tools.slurm.compute_metrics import parse_seff_output, summarize_gpu_metrics, format_compute_table

# Extract job IDs from run logs
job_ids = {}
for log_name in ["run-torchtune.log", "run-inspect.log"]:
    log_path = Path(experiment_dir) / log_name
    if log_path.exists():
        text = log_path.read_text()
        # Match patterns like "Result: Job ID 12345678"
        for match in re.finditer(r"(?:SUBMIT_JOB|SUBMIT_EVAL):\s*(\S+).*?Result:\s*Job ID\s*(\d+)", text, re.DOTALL):
            run_name = match.group(1)
            job_id = match.group(2)
            job_type = "finetune" if "torchtune" in log_name else "eval"
            job_ids[job_id] = {"run_name": run_name, "job_type": job_type}
```

### Running seff and Parsing

```python
import subprocess

jobs = []
for job_id, info in job_ids.items():
    try:
        result = subprocess.run(["seff", job_id], capture_output=True, text=True, timeout=30)
        seff_data = parse_seff_output(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        seff_data = {}

    job_entry = {
        "run_name": info["run_name"],
        "job_type": info["job_type"],
        "wall_time": seff_data.get("wall_time", "-"),
        "time_limit": seff_data.get("time_limit", "-"),
    }
    jobs.append(job_entry)
```

### Adding GPU Metrics from gpu_metrics.csv

```python
for job_entry in jobs:
    # Look for gpu_metrics.csv in the run's output directory
    gpu_csv = find_gpu_metrics_csv(job_entry["run_name"])  # implementation depends on dir structure
    if gpu_csv and gpu_csv.exists():
        gpu_data = summarize_gpu_metrics(gpu_csv)
        job_entry.update(gpu_data)
```

### Generating the Compute Section

```python
# Format as markdown table
compute_table = format_compute_table(jobs)

# Claude writes narrative recommendations based on the data
# (e.g., "GPU utilization averaged 80% during training. Consider increasing
# batch size to improve throughput." or "Eval jobs used <1 minute of a
# 10-minute allocation — eval_time is well-sized.")

compute_section = f"""## Compute Utilization

{compute_table}

### Recommendations

{recommendations}  # Claude-generated narrative based on structured data
"""

# Save raw metrics for reproducibility
metrics_path = Path(output_dir) / "compute_metrics.json"
metrics_path.write_text(json.dumps(jobs, indent=2, default=str))
```

### Passing to Report Generator

```python
report = generate_report(
    df=logs_df,
    experiment_name=experiment_name,
    output_path=Path(output_dir) / "report.md",
    compute_section=compute_section,  # NEW: inserted after Analysis & Interpretation
    # ... other args
)
```

### Error Handling

- **Missing seff**: If `seff` command not found or fails, skip seff data (GPU metrics from CSV may still be available)
- **Missing gpu_metrics.csv**: Show `-` for GPU columns in the table
- **Missing run logs**: Skip compute analysis entirely (log a note)
- **Partial data**: Generate table with whatever data is available

## Logging

Log generation actions using `GENERATE_PLOT`, `GENERATE_REPORT`, and `COMPUTE_METRICS` action types. See `logging.md` for format specification.

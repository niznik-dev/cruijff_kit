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

## Compute Utilization Analysis

After generating visualizations and before the report, optionally add compute metrics. This requires run logs from `run-experiment`.

**Workflow:**

1. Extract job IDs from `run-torchtune.log` and `run-inspect.log` (regex: `SUBMIT_JOB|SUBMIT_EVAL` → Job ID)
2. Check if jobstats is available with `check_jobstats_available()`
3. For each job:
   a. Run `seff {job_id}` and parse with `parse_seff_output()`. If `time_limit` is None (some clusters omit it), run `sacct -j {job_id} --format=Timelimit -P -n` and parse with `parse_sacct_time_limit()`.
   b. Read `gpu_metrics.csv` with `summarize_gpu_metrics()`. **Paths differ by job type:**
      - Fine-tuning: `{output_dir}/ck-out-{run}/gpu_metrics.csv`
      - Evaluation: `{output_dir}/ck-out-{run}/epoch_{N}/gpu_metrics.csv`
   c. If jobstats available: run `run_jobstats(job_id)` for CPU metrics (JSON) and `run_jobstats(job_id, json_mode=False)` for notes. Parse with `parse_jobstats_json()` and `extract_jobstats_notes()`.
4. Build job dicts combining all sources:
   - **CPU**: from jobstats (`cpu_efficiency_pct`, `cpu_mem_used_gb`, `cpu_mem_allocated_gb`), or seff `cpu_efficiency` as fallback
   - **GPU utilization**: dual-source — set `gpu_util_jobstats_pct` from `parse_jobstats_json()["gpu_util_pct"]` (Prometheus average), and `gpu_util_min`/`gpu_util_max` from `summarize_gpu_metrics()` (nvidia-smi range). `format_compute_table` renders this as `avg% (min–max%)` when both are present.
   - **GPU memory / power**: from nvidia-smi CSV (`gpu_mem_used_mean_gb`, `gpu_mem_total_gb`, `power_mean_w`)
5. Format with `format_compute_table(jobs, recommendations=recs)` → markdown table with optional recommendations
6. Save raw metrics to `{output_dir}/compute_metrics.json`
7. Pass `compute_section=` to `generate_report()` (inserted after Analysis & Interpretation)

**Key functions** from `tools.slurm.compute_metrics`:
- `check_jobstats_available() -> bool` — auto-detect jobstats on PATH
- `run_jobstats(job_id, json_mode=True) -> dict | str | None` — run jobstats (JSON or text mode), 30s timeout
- `parse_jobstats_json(js_data: dict) -> dict` — CPU metrics (cores, efficiency, memory) and GPU metrics (`gpu_util_pct`, `gpu_mem_used_gb`, `gpu_mem_total_gb`) from jobstats JSON
- `extract_jobstats_notes(text: str) -> list[str]` — actionable recommendations from jobstats text output
- `parse_seff_output(stdout: str) -> dict` — wall time, time limit, memory, CPU efficiency (fallback)
- `parse_sacct_time_limit(stdout: str) -> str | None` — fallback for time limit when seff omits it
- `summarize_gpu_metrics(csv_path: Path) -> dict` — GPU util (mean, min, max), memory, power from nvidia-smi CSV
- `generate_gpu_recommendations(jobs) -> dict[str, list[str]]` — GPU memory optimization suggestions for finetune jobs
- `format_compute_table(jobs, recommendations=None) -> str` — markdown table with CPU + GPU columns. GPU Util shows `avg% (min–max%)` when `gpu_util_jobstats_pct` and `gpu_util_min`/`gpu_util_max` are both present in the job dict.

**Data sources:** nvidia-smi CSV for GPU memory/power and utilization range (min/max), jobstats for CPU metrics and GPU utilization average (Prometheus-sampled, more reliable than nvidia-smi's 30s polling), seff for job metadata + CPU fallback.

**Error handling:** Missing seff → skip seff columns; missing gpu_metrics.csv → show `-` for GPU columns; jobstats unavailable or fails → fall back to seff for CPU data; missing run logs → skip compute analysis entirely; partial data → generate table with whatever is available.

## Logging

Log generation actions using `GENERATE_PLOT`, `GENERATE_REPORT`, and `COMPUTE_METRICS` action types. See `logging.md` for format specification.

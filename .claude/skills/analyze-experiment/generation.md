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

# Create output directory
output_dir = os.path.join(experiment_dir, "analysis")
os.makedirs(output_dir, exist_ok=True)
```

## View Function Signatures

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

```python
plot = scores_heatmap(
    data,
    task_name='task_name',          # Column with task names
    model_name="model_display_name", # Column with model names
    model_label="Model",            # Label for model axis
    score_value="score_match_accuracy",
    tip=True,                       # Show tooltips
    title="",                       # Plot title (empty for no title)
    orientation="vertical"          # or "horizontal"
)
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
from tools.inspect.viz_helpers import detect_metrics

# Automatically detect available metrics
metrics = detect_metrics(logs_df)
# Returns e.g., ['match', 'includes'] or ['match'] depending on data
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

## Generating Multiple Plots

For experiments with multiple metrics, generate separate plots using dynamic detection:

```python
from tools.inspect.viz_helpers import detect_metrics

def generate_all_plots(data, logs_df, output_dir, views):
    """Generate all inferred visualizations."""

    generated = []
    metrics = detect_metrics(logs_df)  # Dynamic detection

    for view_spec in views:
        view_type = view_spec['view']

        for metric in metrics:
            score_col = f"score_{metric}_accuracy"
            stderr_col = f"score_{metric}_stderr"

            if score_col not in logs_df.columns:
                continue

            try:
                if view_type == 'scores_by_task':
                    plot = scores_by_task(
                        data,
                        task_name='task_name',
                        score_value=score_col,
                        score_stderr=stderr_col,
                        score_label=f"{metric.title()} Accuracy",
                        ci=0.95
                    )
                    filename = f"scores_by_task_{metric}.html"

                elif view_type == 'scores_heatmap':
                    plot = scores_heatmap(
                        data,
                        task_name='task_name',
                        model_name="model_display_name",
                        model_label="Model",
                        score_value=score_col,
                        tip=True,
                        title=""
                    )
                    filename = f"scores_heatmap_{metric}.html"

                elif view_type == 'scores_by_model':
                    plot = scores_by_model(
                        data,
                        score_value=score_col,
                        score_stderr=stderr_col,
                        score_label=f"{metric.title()} Accuracy",
                        ci=0.95,
                        height=200
                    )
                    filename = f"scores_by_model_{metric}.html"

                elif view_type == 'scores_by_factor':
                    factor = view_spec.get('factor', 'factor')
                    plot = scores_by_factor(
                        data,
                        factor=factor,
                        factor_labels=("False", "True"),
                        score_value=score_col,
                        score_stderr=stderr_col,
                        score_label=f"{metric.title()} Accuracy",
                        model="model",
                        model_label="Model",
                        ci=0.95
                    )
                    filename = f"scores_by_factor_{metric}.html"

                # Save HTML
                html_path = os.path.join(output_dir, filename)
                write_html(html_path, plot)
                generated.append(filename)

                # Save PNG (auto-detect playwright)
                png_filename = filename.replace('.html', '.png')
                png_path = os.path.join(output_dir, png_filename)
                try:
                    from playwright.sync_api import sync_playwright
                    write_png(png_path, plot)
                    generated.append(png_filename)
                except ImportError:
                    pass  # PNG skipped silently in batch generation

            except Exception as e:
                print(f"Error generating {view_type} for {metric}: {e}")

    # Special case: radar plot (combines multiple metrics)
    if any(v['view'] == 'scores_radar_by_task' for v in views):
        try:
            radar_df = scores_radar_by_task_df(
                logs_df,
                invert=["score_match_accuracy", "score_includes_accuracy"],
                normalization="min_max",
                domain=(0, 1)
            )
            plot = scores_radar_by_task(Data.from_dataframe(radar_df))
            filepath = os.path.join(output_dir, "scores_radar.html")
            write_html(filepath, plot)
            generated.append("scores_radar.html")
        except Exception as e:
            print(f"Error generating radar plot: {e}")

    return generated
```

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

## Logging

Log generation actions:

```json
{"action": "GENERATE", "timestamp": "...", "view": "scores_by_task", "metric": "match", "output": "scores_by_task_match.html", "status": "success"}
```

```json
{"action": "GENERATE", "timestamp": "...", "view": "scores_heatmap", "metric": "includes", "output": null, "status": "error", "error": "Missing task_name column"}
```

See `logging.md` for complete format specification.

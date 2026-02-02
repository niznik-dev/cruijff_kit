# Data Loading: Loading Evaluation Logs

This module describes how to load evaluation logs and prepare data for visualization using the helper functions in `tools/inspect/viz_helpers.py`.

## Automatic Deduplication

When multiple evaluations exist for the same model+epoch (e.g., from re-runs), use `deduplicate_eval_files()` to keep only the most recent:

```python
from tools.inspect.viz_helpers import deduplicate_eval_files

# Collect all .eval files
eval_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.eval')]

# Deduplicate (keeps most recent per model+epoch)
kept, skipped = deduplicate_eval_files(eval_files)
print(f"Kept {len(kept)} files, skipped {len(skipped)} duplicates")

# Use kept files for further processing
logs_df = evals_df_prep(kept)
```

**How deduplication works:**
1. Reads model and epoch from each eval file
2. Extracts timestamp from filename (format: `YYYYMMDDTHHMMSS_...`)
3. Groups by (model, epoch) combination
4. Keeps only the most recent timestamp per group
5. Returns tuple of (kept_files, skipped_files) for logging

## Metadata Extraction

Use `parse_eval_metadata()` to extract structured metadata from eval files:

```python
from tools.inspect.viz_helpers import evals_df_prep, parse_eval_metadata

# Load eval files
logs_df = evals_df_prep(eval_files)

# Parse metadata into columns
logs_df = parse_eval_metadata(logs_df)

# Now available:
# - logs_df['task_name']    (from task_arg_vis_label)
# - logs_df['epoch']        (from JSON metadata)
# - logs_df['finetuned']    (from JSON metadata)
# - logs_df['source_model'] (from JSON metadata)
```

## Helper Functions

### evals_df_prep()

Prepare eval-level dataframes from a list of .eval file paths:

```python
from tools.inspect.viz_helpers import evals_df_prep

logs_df = evals_df_prep(kept)  # kept is list of .eval file paths
```

### detect_metrics()

Dynamically detect available score columns:

```python
from tools.inspect.viz_helpers import detect_metrics

metrics = detect_metrics(logs_df)
# Returns e.g., ['match', 'includes'] depending on what's in the data
```

## Constructing Subdirs from Config

Build the subdirs list from experiment_summary.yaml:

```python
def get_subdirs_from_config(config):
    """Extract run directory names from parsed config."""
    runs = config.get('runs', [])
    return [run['name'] for run in runs]

# Usage
subdirs = get_subdirs_from_config(config)
```

## Required Columns by View Type

| View | Required Columns |
|------|------------------|
| scores_by_task | `task_name` |
| scores_by_model | (single task only) |
| scores_heatmap | `task_name`, model column |
| scores_by_factor | boolean factor column |
| scores_radar_by_task | multiple `score_*_accuracy` columns |

All score columns (`score_match_accuracy`, `score_includes_accuracy`, etc.) are automatically included via `EvalScores` in `evals_df_prep()`.

## Log Viewer URL (Optional)

The `log_viewer_url` enables clickable links in visualizations for drill-down analysis.

**To start log viewer:**
```bash
inspect view --port=8000
```

## Error Handling

**If eval/logs directory doesn't exist:**
```python
for subdir in subdirs:
    log_path = os.path.join(experiment_path, subdir, "eval", "logs")
    if not os.path.exists(log_path):
        print(f"Warning: No eval logs found in {subdir}")
```

**If no .eval files found:**
- Log error
- Report which runs are missing evaluations
- Suggest running run-experiment skill

**If metadata extraction fails:**
- Log warning
- Use default values where possible
- Continue with other columns

## Logging

Log data loading actions using `LOAD_EVALS` and `DEDUPLICATE` action types. See `logging.md` for format specification.

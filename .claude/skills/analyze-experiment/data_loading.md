# Data Loading: Loading Evaluation Logs

This module describes how to load evaluation logs and prepare data for visualization using the helper functions in `tools/inspect/viz_helpers.py`.

## Helper Functions

### load_experiment_logs()

The main function for loading experiment data:

```python
from tools.inspect.viz_helpers import load_experiment_logs

logs_df = load_experiment_logs(
    experiment_path="/path/to/experiment",
    subdirs=["run1", "run2"],
    log_viewer_url="http://localhost:8000/logs/",
    metadata_extractors={
        "column_name": lambda df: df['source_col'].str.extract(r'pattern', expand=False)
    }
)
```

**Parameters:**
- `experiment_path`: Path to main experiment directory
- `subdirs`: List of subdirectory names containing `eval/logs/`
- `log_viewer_url`: URL for inspect log viewer (enables clickable links in plots)
- `metadata_extractors`: Dict of column names to lambda functions for custom metadata

**Returns:** Prepared DataFrame ready for inspect-viz visualization

### evals_df_prep()

Lower-level function for preparing eval-level dataframes:

```python
from tools.inspect.viz_helpers import evals_df_prep

logs_df = evals_df_prep(logs)  # logs is list of .eval file paths
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

## Metadata Extractors for Capitalization Experiments

### Word Length Experiment

Extract model size and word length:

```python
metadata_extractors = {
    # Extract model size (e.g., "1B", "3B")
    "model": lambda df: (
        df['model'].str.extract(r'Llama-3\.2-(?P<size>\d+B)', expand=False)
        + '_epoch'
        + df['model'].str.extract(r'epoch_(?P<epoch>\d+)', expand=False)
    ),
    # Extract word length from task data path
    "task_name": lambda df: df['task_arg_data_path'].str.extract(
        r'words_(?P<task_name>\d+L)', expand=False
    )
}
```

### Model × Prompt Experiment

Extract model and prompt type:

```python
metadata_extractors = {
    # Extract model name
    "model": lambda df: df['model_path'].str.extract(
        r'(Llama-3\.2-\d+B)', expand=False
    ),
    # Extract prompt type from config path
    "prompt_type": lambda df: df['task_arg_config_path'].str.extract(
        r'Instruct_(?P<prompt_type>with_prompt|no_prompt)', expand=False
    )
}

# After loading, convert to boolean for scores_by_factor
logs_df['prompt_type'] = logs_df['prompt_type'] == 'with_prompt'
```

### Cross-Organization Experiment

Extract organization and model:

```python
metadata_extractors = {
    "model": lambda df: df['model_path'].str.extract(
        r'(Google-Gemma-2B|Meta-Llama-3\.2-1B-Instruct)', expand=False
    )
}
```

## Log Viewer URL

The `log_viewer_url` enables clickable links in visualizations that open the inspect log viewer for drill-down analysis.

**Format:** `http://localhost:{port}/{path}/`

**Example:**
```python
log_viewer_url = "http://localhost:8000/cap_wordlen_logs_viewer/"
```

**To start log viewer:**
```bash
inspect view --port=8000
```

## Data Preparation for Each View Type

### For scores_by_task

Requires `task_name` column:
```python
# Ensure task_name is extracted
metadata_extractors = {
    "task_name": lambda df: df['task_arg_data_path'].str.extract(r'pattern', expand=False)
}
```

### For scores_by_factor

Requires boolean factor column:
```python
# Extract factor then convert to boolean
logs_df['factor'] = logs_df['factor'] == 'condition_true'
```

### For scores_heatmap

Requires both model and task columns:
```python
metadata_extractors = {
    "model": lambda df: ...,
    "task_name": lambda df: ...
}
```

### For scores_radar_by_task

Requires multiple score columns (automatically included via EvalScores):
- `score_match_accuracy`
- `score_includes_accuracy`
- etc.

### For scores_by_model

Uses `model_display_name` column (automatically added by prepare()):
```python
# No special extraction needed - uses default model column
```

## Error Handling

**If eval/logs directory doesn't exist:**
```python
import os

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
- Log warning with extraction pattern
- Use default values where possible
- Continue with other columns

## Logging

Log data loading actions:

```json
{"action": "LOAD", "timestamp": "...", "experiment_path": "/path/to/exp", "subdirs": ["run1", "run2"], "logs_found": 12, "status": "success"}
```

See `logging.md` for complete format specification.

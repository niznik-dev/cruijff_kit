# Inference: Selecting Appropriate Visualizations

This module describes how to infer which inspect-viz pre-built views are appropriate for a given experiment based on its variables and structure.

## View Selection Logic

Map experimental characteristics to appropriate views:

| Characteristic | View | Rationale |
|----------------|------|-----------|
| Multiple models + single task | `scores_by_model` | Bar chart comparing model performance (single task only) |
| Binary factor (yes/no, with/without) | `scores_by_factor` | Paired comparison across factor levels |
| Multiple tasks or conditions | `scores_by_task` | Compare scores across different tasks |
| Model × task combinations | `scores_heatmap` | Matrix visualization of all combinations |
| Multiple metrics to compare | `scores_radar_by_task` | Radar plot showing metric trade-offs |

## Inference Algorithm

```python
def infer_visualizations(config, logs_df):
    """
    Determine which visualizations to generate based on experiment structure.

    Returns list of view specifications.
    """
    views = []
    variables = extract_variable_info(config)

    # Count unique values in key columns
    n_models = logs_df['model'].nunique() if 'model' in logs_df.columns else 0
    n_tasks = logs_df['task_name'].nunique() if 'task_name' in logs_df.columns else 0

    # Check for binary factors
    binary_factors = [v for v in variables if v['type'] == 'binary']

    # Rule 1: Multiple models + single task → scores_by_model
    # Note: scores_by_model requires single-task experiments
    if n_models > 1 and n_tasks == 1:
        views.append({
            'view': 'scores_by_model',
            'reason': f'Found {n_models} models with single task'
        })

    # Rule 2: Binary factor → scores_by_factor
    if binary_factors:
        for factor in binary_factors:
            views.append({
                'view': 'scores_by_factor',
                'factor': factor['name'],
                'reason': f'Binary factor: {factor["name"]}'
            })

    # Rule 3: Multiple tasks → scores_by_task
    if n_tasks > 1:
        views.append({
            'view': 'scores_by_task',
            'reason': f'Found {n_tasks} tasks/conditions'
        })

    # Rule 4: Model × task matrix → scores_heatmap
    if n_models > 1 and n_tasks > 1:
        views.append({
            'view': 'scores_heatmap',
            'reason': f'Model × task matrix ({n_models} × {n_tasks})'
        })

    # Rule 5: Multiple score columns → scores_radar_by_task
    score_cols = [c for c in logs_df.columns if c.startswith('score_') and c.endswith('_accuracy')]
    if len(score_cols) > 1:
        views.append({
            'view': 'scores_radar_by_task',
            'scores': score_cols,
            'reason': f'Multiple metrics: {score_cols}'
        })

    return views
```

## Variable Type Detection

Detect variable types from experiment_summary.yaml:

```python
def detect_variable_type(values):
    """Detect if variable is binary, continuous, or categorical."""

    # Binary: exactly 2 values that look boolean-ish
    if len(values) == 2:
        bool_patterns = [True, False, 0, 1, 'yes', 'no', 'true', 'false',
                        'with_prompt', 'no_prompt', 'enabled', 'disabled',
                        'balanced', 'imbalanced', 'train', 'test']
        if all(str(v).lower() in [str(p).lower() for p in bool_patterns] for v in values):
            return 'binary'

    # Continuous: all numeric
    if all(isinstance(v, (int, float)) for v in values):
        return 'continuous'

    # Default: categorical
    return 'categorical'
```

## Column Detection

Check what columns are available in the loaded dataframe:

```python
def detect_available_columns(logs_df):
    """Detect which columns are available for visualization."""

    available = {
        'model': 'model' in logs_df.columns,
        'task_name': 'task_name' in logs_df.columns,
        'model_display_name': 'model_display_name' in logs_df.columns,
    }

    # Find score columns
    available['score_columns'] = [
        c for c in logs_df.columns
        if c.startswith('score_') and c.endswith('_accuracy')
    ]

    # Find potential factor columns (boolean)
    available['factor_columns'] = [
        c for c in logs_df.columns
        if logs_df[c].dtype == bool
    ]

    return available
```

## vis_label View Selection

When experiments use `vis_label` metadata (from scaffold-inspect), multiple conditions may exist. In this case, **always ask the user** which visualization to generate:

```
Found {N} conditions via vis_label: {list of vis_labels}

Which visualization would you like?

1. scores_by_task - Compare conditions side-by-side (Recommended)
2. scores_heatmap - Model × condition matrix
3. Both
```

**Detection logic:**
```python
# Check for vis_label variants
if 'task_name' in logs_df.columns:
    vis_labels = logs_df['task_name'].unique().tolist()
    if len(vis_labels) > 1:
        # Ask user which view to generate
        pass
```

## Handling Ambiguous Cases

If inference cannot determine the appropriate view:

1. **Log the ambiguity:**
   ```json
   {"action": "INFER", "status": "ambiguous", "message": "Could not determine view type", "available_columns": [...]}
   ```

2. **Ask user for guidance:**
   ```
   I found {N} models and {M} tasks but I'm not sure which visualization would be most useful.

   Options:
   1. scores_by_model - Compare models directly
   2. scores_by_task - Compare across tasks
   3. scores_heatmap - Show model × task matrix
   4. All of the above

   Which would you prefer?
   ```

3. **Default to safe options:**
   - If multiple models: generate `scores_by_model`
   - If multiple tasks: generate `scores_by_task`
   - Always safe to generate multiple views

## Logging

Log inference decisions:

```json
{"action": "INFER", "timestamp": "...", "views_selected": ["scores_by_task", "scores_heatmap"], "reasons": ["Found 3 tasks", "2 models × 3 tasks matrix"], "status": "success"}
```

See `logging.md` for complete format specification.

# Inspect_Viz Workflow

## 1. Load experiment data
- Define experiment paths (experiment directory and subfolders for each run variation).
- Create LOG_DIRS mapping (for log_viewers)
- Collect eval logs.

## 2. Identify relevant variables from experimental design and infer suitable visualizations.
- Use the experiment summary to infer which factors vary in the model (e.g., task, factor, model, etc.). In practice, this will need to be done manually (by user input) or using LLM assistance.

### Pre-Built Views in Inspect_Viz - When to Use Each

#### scores_by_task()
- **Displays**: Bar plot comparing eval scores across models and tasks (with confidence intervals)
- **Use when**: You have multiple models evaluated on different task variations
- **Example**: Experiment 1 (model x wordlength) - comparing how 1B and 3B models perform on 5L, 6L, 7L word tasks
- **Data needs**: `model` column, `task_name` column, `score` values

#### scores_by_model()
- **Displays**: Bar plot comparing model scores on a single eval (with optional baselines)
- **Use when**: Benchmarking multiple models against each other on the same evaluation
- **Example use case**: Comparing different model sizes or organizations on the same capitalization task
- **Data needs**: `model` column, `score` values

#### scores_by_factor()
- **Displays**: Bar plot comparing scores by model and a **boolean factor** (e.g., no hint vs. hint)
- **Use when**: You have a binary experimental condition (True/False, with/without)
- **Example**: Experiment 2 (model x system prompt) - comparing performance with_prompt vs. no_prompt
- **Data needs**: `model` column, boolean factor column (e.g., `prompt_type`), `score` values
- **Note**: Factor labels can be customized with tuple like `("No Prompt", "Prompt")`

#### scores_heatmap()
- **Displays**: 2D heatmap showing scores across model and task dimensions
- **Use when**: You want to quickly spot patterns/weak spots across the model×task grid
- **Example**: Experiment 1 - visualizing which models struggle with which word lengths
- **Data needs**: Two categorical dimensions (typically `model` and `task_name`)

#### scores_radar_by_task()
- **Displays**: Radar plot comparing models across headline metrics from different evals
- **Use when**: You want multi-dimensional comparison showing model performance across multiple task types
- **Example**: Experiment 1 - showing how models perform across all word lengths simultaneously
- **Data needs**: Requires special preparation with `scores_radar_by_task_df()` for normalization

#### scores_radar_by_metric()
- **Displays**: Radar plot for comparing model scores across multiple metrics from a **single eval**
- **Use when**: You have multiple performance metrics (accuracy, F1, precision, etc.) from one evaluation
- **Example use case**: Not yet used in capitalization experiments but useful for multi-metric evaluations

#### scores_by_limit()
- **Displays**: Line plot showing success rate by token limit
- **Use when**: Analyzing how computational constraints (token/time limits) affect performance
- **Example use case**: Testing how models perform with varying max_tokens settings

#### scores_timeline()
- **Displays**: Scatter plot with scores by model, organization, and release date
- **Use when**: Tracking model improvements over time or comparing models by release date
- **Example use case**: Longitudinal studies of model capabilities across releases

#### sample_tool_calls()
- **Displays**: Heatmap visualizing tool calls over evaluation turns
- **Use when**: Analyzing tool usage patterns in multi-turn agent evaluations
- **Example use case**: Not applicable to capitalization task (no tool use)

### Experimental Design Patterns

**For factorial experiments** (multiple independent variables):
- Use **scores_by_task** + **scores_heatmap** + **scores_radar_by_task** together for comprehensive views

**For binary condition experiments**:
- Use **scores_by_factor** specifically (designed for True/False variables)
- Can also use **scores_by_task** (treating each condition as a task)

## 3. Prepare data for visualization

### General Data Preparation Pattern

1. **Load logs with `read_eval_log()`**
   ```python
   from inspect_ai.log import read_eval_log

   logs = [path1, path2, ...]  # List of .eval file paths
   model_paths = [read_eval_log(log).eval.model_args['model_path'] for log in logs]
   ```

2. **Create dataframe with `evals_df()`**
   ```python
   from inspect_ai.analysis import evals_df, EvalInfo, EvalTask, EvalModel, EvalResults, EvalScores

   logs_df = evals_df(
       logs=logs,
       columns=(EvalInfo + EvalTask + EvalModel + EvalResults + EvalScores)
   )
   ```

3. **ONLY IF local model is used: Extract variables from paths using regex**
   - Model name: `logs_df['model_name'] = logs_df['model_path'].str.extract(r'(Llama-3\.2-\d+B)', expand=False)`
   - Epoch: `logs_df['epoch'] = logs_df['model_path'].str.extract(r'epoch_(\d+)', expand=False)`
   - Task name: `logs_df['task_name'] = logs_df['task_arg_config_path'].str.extract(r'eval_config_(?P<task_name>\d+L)', expand=False)`
   - Combine: `logs_df['model'] = logs_df['model_name'] + '_epoch' + logs_df['epoch']`

4. **Apply `prepare()` with `log_viewer()` for clickable links**
   ```python
   from inspect_ai.analysis import log_viewer, prepare

   LOG_DIRS = {
       "/path/to/logs/": "http://localhost:8000/bundled_viewer/",
   }

   logs_df = prepare(logs_df, [log_viewer("eval", LOG_DIRS)])
   ```

5. **Convert to `Data.from_dataframe()` for inspect_viz**
   ```python
   from inspect_viz import Data

   evals = Data.from_dataframe(logs_df)
   ```

### Specific Instructions for Each View

#### For scores_by_task()
- **Required columns**: `model`, `task_name`, score values
- **Preparation**:
  ```python
  evals = Data.from_dataframe(logs_df)
  scores_by_task(evals, task_name='task_name')
  ```

#### For scores_by_model()
- **Required columns**: `model`, score values
- **Preparation**: Same as scores_by_task but focuses on model comparison for single eval

#### For scores_by_factor()
- **Required columns**: `model`, boolean factor column (e.g., `prompt_type`), score values
- **Preparation**:
  ```python
  # Convert factor to boolean
  logs_df['prompt_type'] = logs_df['prompt_type'] == 'with_prompt'

  evals_factor = Data.from_dataframe(logs_df)
  scores_by_factor(evals_factor, "prompt_type", ("No Prompt", "Prompt"))
  ```

#### For scores_heatmap()
- **Required columns**: `model`, `task_name`, score values
- **Preparation**:
  ```python
  evals = Data.from_dataframe(logs_df)
  scores_heatmap(evals, task_name='task_name')
  ```

#### For scores_radar_by_task()
- **Required columns**: Normalized scores across tasks
- **Special preparation (for normalization)**:
  ```python
  from inspect_viz.view.beta import scores_radar_by_task_df

  # Create normalized scores dataframe
  evals_scores = scores_radar_by_task_df(
      logs_df,
      normalization="min_max",
      domain=(0, 1),
  )

  # Apply log viewer
  evals_scores = prepare(evals_scores, [log_viewer("eval", LOG_DIRS)])

  # Save and reload via parquet (not ne workaround)
  evals_radar = Data.from_dataframe(evals_scores)
  scores_radar_by_task(evals_radar)
  ```

## 4. Create appropriate visualizations using Inspect_Viz.

### Example 1: Model x Task (Factorial Design)

```python
# After data preparation from section 3
from inspect_viz.view.beta import scores_by_task, scores_heatmap, scores_radar_by_task

# Create multiple complementary views
evals = Data.from_dataframe(logs_df)

# Bar chart for detailed comparisons
scores_by_task(evals, task_name='task_name')

# Heatmap for pattern identification
scores_heatmap(evals, task_name='task_name')

# Radar chart for holistic view (after special preparation)
scores_radar_by_task(evals_radar)
```

### Example 2: Model x Binary Factor

```python
# After data preparation from section 3
from inspect_viz.view.beta import scores_by_factor

evals_factor = Data.from_dataframe(logs_df)
scores_by_factor(evals_factor, "prompt_type", ("No Prompt", "Prompt"))
```

### Saving Visualizations to HTML

```python
from inspect_viz.plot import plot, write_html

# Create plot
p = plot(scores_by_task(evals, task_name='task_name'))

# Save to HTML file
write_html('../html/my_visualization.html', p)
```

### Adding Clickable Log Viewer Links

For visualizations with clickable links to detailed eval logs:

1. **Create bundled log viewer**:
   ```bash
   inspect view bundle \
     --log-dir /path/to/run1/eval/logs \
     --log-dir /path/to/run2/eval/logs \
     --output-dir viz_examples/html/experiment_logs_viewer
   ```

2. **Map in LOG_DIRS**:
   ```python
   LOG_DIRS = {
       "/path/to/run1/eval/logs": "http://localhost:8000/experiment_logs_viewer/",
       "/path/to/run2/eval/logs": "http://localhost:8000/experiment_logs_viewer/",
   }
   ```

3. **Serve via HTTP**:
   ```bash
   cd viz_examples/html
   python -m http.server 8000
   ```

4. **Open visualization**: Navigate to `http://localhost:8000/my_visualization.html`

See `CLICKABLE_LINKS_SETUP.md` for detailed instructions on setting up clickable log viewer links.
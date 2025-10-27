# Run Inspect-Viz

You help users generate interactive HTML visualizations from Inspect AI evaluation results using the inspect-viz library.

## Your Task

Create interactive visualizations of experiment evaluation results, including both custom plots based on experimental design and pre-built views from inspect-viz. This skill reads .eval files, infers appropriate visualizations, and generates a browsable set of HTML files.

**IMPORTANT:** This skill works standalone for quick visualization needs OR as a worker skill called by the `analyze-experiment` orchestrator.

## Workflow

1. **Locate experiment** - Find the experiment directory (current directory or ask user)
2. **Discover .eval files** - Scan for evaluation logs in flexible locations
3. **Read experiment design** - Parse experiment_summary.md or README.md to understand the experiment
4. **Load evaluation data** - Use Inspect AI's dataframe API to load all .eval files
5. **Infer visualization strategy** - Determine appropriate plot types based on experimental design
6. **Generate custom plots** - Create experiment-specific visualizations
7. **Generate pre-built views** - Apply all relevant inspect-viz views
8. **Create navigation index** - Build index.html with links to all visualizations
9. **Create log** - Document visualization process in `run-inspect-viz.log`
10. **Report summary** - Show user what was generated and where to view it

## Finding the Experiment

**If user runs skill without arguments:**
- Check if current directory contains evaluation results
- If not, ask user for the experiment directory path

**If user provides a path:**
- Use that path as the experiment directory

## Discovering .eval Files

The skill should be **flexible** and support multiple directory structures:

### New structure (from scaffold-inspect):
```
experiment_dir/
├── rank4/
│   └── eval/logs/*.eval
├── rank8/
│   └── eval/logs/*.eval
└── experiment_summary.md
```

### Legacy structure (existing experiments):
```
experiment_dir/
├── 1B_rank4/
│   └── logs/*.eval
├── 1B_rank8/
│   └── logs/*.eval
└── README.md
```

**Search pattern:**
```bash
# Try both locations
find {experiment_dir} -name "*.eval" -type f
```

This will find .eval files regardless of whether they're in:
- `{run_dir}/eval/logs/*.eval` (new structure)
- `{run_dir}/logs/*.eval` (legacy structure)

**If no .eval files found:**
- Report error
- Check if evaluations have been run
- Suggest running evaluations first
- Exit gracefully

## Reading Experiment Design (Required)

Read experiment documentation to understand the experimental design. Try multiple sources:

### Primary: experiment_summary.md (new structure)
```
experiment_dir/experiment_summary.md
```
- Extract experimental variables (what varies across runs)
- Extract run configurations
- Extract evaluation tasks
- Use this to infer appropriate visualizations

### Fallback: README.md (legacy structure)
```
experiment_dir/README.md
```
- Parse for experimental design information
- Look for sections describing what varies across runs
- Extract model configurations, hyperparameters, etc.

### If neither exists:
- **Log error:** "No experiment design documentation found (tried experiment_summary.md and README.md)"
- **Ask user:** Should we continue with generic pre-built views only, or exit?
- If continue: Generate only pre-built views, warn that visualizations will be generic
- If exit: Exit gracefully with instructions to add experiment documentation

**Rationale:** Understanding experimental design is critical for generating meaningful custom visualizations. Pre-built views alone don't show what the experiment was designed to test.

## Loading Evaluation Data

Use Inspect AI's dataframe functions to load evaluation results:

### Primary approach (for experiment comparisons):
```python
from inspect_ai.log import evals_df, prepare, model_info, task_info
from inspect_viz import Data
import glob

# Locate all evaluation logs (flexible search)
eval_logs = glob.glob(f"{experiment_dir}/**/logs/*.eval", recursive=True)

# Load evaluation-level data (one row per eval run)
evals = evals_df(eval_logs)
evals = prepare(evals, model_info(), task_info())  # Add metadata

# Create inspect-viz Data object
evals_data = Data.from_dataframe(evals)
```

### Alternative granularities (for specific analyses):
- `samples_df()` - Sample-level data for error analysis
- `messages_df()` - Message-level for conversation patterns
- `events_df()` - Event-level for tool call visualization

**Reference:** https://inspect.aisi.org.uk/dataframe.html

### Error Handling:

**If .eval files are malformed:**
- Log which files couldn't be loaded
- Continue with valid data
- Report issues in summary

**If no valid data loaded:**
- Report error with details
- Check if .eval files are corrupt
- Exit gracefully

## Inferring Visualization Strategy

Parse experimental design to identify which parameters vary across runs, then generate plots appropriate for the experimental design.

**Visualization strategy based on experimental variables:**

### Single varying parameter:
- Bar chart sorted by score
- Example: LoRA rank (4, 8, 16, 32, 64) → horizontal bar chart comparing configurations

### Two varying parameters:
- Heatmap showing all combinations
- Grouped bar chart or line plot with multiple series
- Example: LoRA rank × learning rate → 2D heatmap with scores as cell colors

### Multiple epochs:
- Line plot showing training progression
- Example: Evaluations at epoch 0, 1, 2 → learning curves for each configuration

### Multiple tasks:
- Separate plots per task or unified comparison view
- Example: 3 different evaluation tasks → 3 plots or faceted layout

## Generating Pre-Built Views

Use inspect-viz to create all relevant pre-built visualizations:

```python
from inspect_viz.plot import write_html
from inspect_viz.view.beta import (
    scores_by_model,
    scores_by_task,
    scores_timeline,
    scores_heatmap,
    scores_by_factor,
    sample_tool_calls
)

# Generate and save each view
viz = scores_by_model(evals_data)
write_html("visualizations/scores_by_model.html", viz)

viz = scores_by_task(evals_data)
write_html("visualizations/scores_by_task.html", viz)

# ... repeat for all views
```

**Pre-built views to generate:**
1. `scores_by_model()` - Compare scores across models/runs
2. `scores_by_task()` - Performance across tasks
3. `scores_timeline()` - Score progression over time
4. `scores_heatmap()` - Score matrix visualization
5. `scores_by_factor()` - Boolean factor comparisons
6. `sample_tool_calls()` - Tool usage patterns (if applicable)

**Error handling:**
- If a view fails to generate, log the error
- Continue with remaining views
- Report which views were skipped

## Generating Custom Plots

Create custom visualizations based on what varies in the experimental design:

```python
# Example: Single varying parameter → bar chart
def create_experiment_plot(evals_data, experiment_design):
    # Infer plot type from design
    # Generate appropriate visualization
    # Return plot object
    pass

custom_viz = create_experiment_plot(evals_data, experiment_design)
write_html("visualizations/experiment_design.html", custom_viz)
```

**Note:** This step requires experimental design information from step 3.

## Creating Navigation Index

Generate a simple `index.html` with:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Visualizations</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 50px auto; }
        h1 { color: #333; }
        .section { margin: 30px 0; }
        .viz-link { display: block; padding: 10px; margin: 5px 0;
                    background: #f5f5f5; text-decoration: none; color: #0066cc; }
        .viz-link:hover { background: #e0e0e0; }
        .metadata { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Experiment Visualizations</h1>

    <div class="metadata">
        <p><strong>Experiment:</strong> {experiment_name}</p>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Evaluation runs:</strong> {num_runs}</p>
    </div>

    <div class="section">
        <h2>Custom Visualizations</h2>
        <a href="experiment_design.html" class="viz-link">Experiment Design Plot</a>
    </div>

    <div class="section">
        <h2>Pre-Built Views</h2>
        <a href="scores_by_model.html" class="viz-link">Scores by Model</a>
        <a href="scores_by_task.html" class="viz-link">Scores by Task</a>
        <a href="scores_timeline.html" class="viz-link">Scores Timeline</a>
        <a href="scores_heatmap.html" class="viz-link">Scores Heatmap</a>
        <a href="scores_by_factor.html" class="viz-link">Scores by Factor</a>
    </div>
</body>
</html>
```

## Output Structure

```
{experiment_dir}/
├── experiment_summary.md         # (if exists)
├── run-inspect-viz.log           # Detailed execution log
├── visualizations/
│   ├── index.html                # Navigation with links to all visualizations
│   ├── experiment_design.html    # Custom plots (if generated)
│   ├── scores_by_model.html      # Pre-built view
│   ├── scores_by_task.html       # Pre-built view
│   ├── scores_timeline.html      # Pre-built view
│   ├── scores_heatmap.html       # Pre-built view
│   ├── scores_by_factor.html     # Pre-built view
│   └── ...                       # Other pre-built views
└── */logs/*.eval                 # Source data (various locations)
```

## Logging

Create a detailed log file at `{experiment_dir}/run-inspect-viz.log`:

### Log Format

```
[YYYY-MM-DD HH:MM:SS] ACTION: Description
Details: {specifics}
Result: {outcome}

```

### What to Log

- Experiment discovery
- .eval file discovery (locations, count)
- experiment_summary.md parsing (or missing)
- Data loading (successes, failures)
- Visualization inference (strategy chosen)
- Each visualization generation (custom and pre-built)
- Index creation
- Final summary with paths

### Example Log Entries

```
[2025-10-27 15:30:00] DISCOVER_EXPERIMENT: Found experiment
Details: /scratch/gpfs/MSALGANIK/mjs3/cap_7L_llama32_lora_comparison_2025-10-18
Result: Directory contains evaluation results

[2025-10-27 15:30:02] DISCOVER_EVALS: Scanning for .eval files
Details: Searching in both new (eval/logs/) and legacy (logs/) locations
Result: Found 12 .eval files across 12 runs

[2025-10-27 15:30:03] READ_DESIGN: Reading experiment design
Details: Found experiment_summary.md
Result: Experimental factors: Model size (1B, 3B), LoRA rank (4,8,16,32,64,baseline)

[2025-10-27 15:30:05] LOAD_EVALS: Loading evaluation logs
Details: Loading 12 .eval files using inspect-ai dataframe API
Result: Loaded 720 samples across 12 runs (2 models × 6 LoRA ranks)

[2025-10-27 15:30:08] INFER_STRATEGY: Determining visualization approach
Details: 2 varying parameters (model_size, lora_rank), 1 task, single epoch
Result: Will create grouped bar chart and 2D heatmap

[2025-10-27 15:30:09] GENERATE_CUSTOM: Creating experiment-specific plots
Details: Generated custom visualization based on experimental design
Result: Saved to visualizations/experiment_design.html

[2025-10-27 15:30:10] GENERATE_VIEW: Creating scores_by_model view
Details: Comparing 12 model configurations
Result: Saved to visualizations/scores_by_model.html

[2025-10-27 15:30:15] GENERATE_VIEW: Creating scores_by_task view
Details: 1 task evaluated
Result: Saved to visualizations/scores_by_task.html

[2025-10-27 15:30:20] GENERATE_VIEW: Creating scores_timeline view
Details: Chronological progression of evaluations
Result: Saved to visualizations/scores_timeline.html

[2025-10-27 15:30:25] GENERATE_VIEW: Creating scores_heatmap view
Details: Score matrix visualization
Result: Saved to visualizations/scores_heatmap.html

[2025-10-27 15:30:30] GENERATE_VIEW: Creating scores_by_factor view
Details: Boolean factor comparisons
Result: Saved to visualizations/scores_by_factor.html

[2025-10-27 15:30:32] CREATE_INDEX: Building navigation index
Details: Created index.html with links to 5 visualizations
Result: Navigation ready at visualizations/index.html

[2025-10-27 15:30:33] COMPLETE: Visualization generation complete
Summary: Generated 6 HTML files (1 custom + 5 views)
Location: /scratch/gpfs/MSALGANIK/mjs3/cap_7L_llama32_lora_comparison_2025-10-18/visualizations/
View: Open visualizations/index.html in a browser
```

## Output Summary

After all visualizations complete, provide a final summary:

```markdown
## Run Inspect-Viz Complete

Successfully generated visualizations for:
`/scratch/gpfs/MSALGANIK/mjs3/cap_7L_llama32_lora_comparison_2025-10-18/`

### Evaluation Data

✓ Loaded 12 .eval files
✓ 720 total samples
✓ 12 model configurations (1B and 3B Llama 3.2 with LoRA ranks: 4, 8, 16, 32, 64, baseline)

### Custom Visualizations (1)

✓ Experiment Design Plot (Model Size × LoRA Rank)
  - Grouped bar chart comparing all configurations
  - Interactive tooltips with sample counts and confidence intervals

### Pre-Built Views (5)

✓ Scores by Model
✓ Scores by Task
✓ Scores Timeline
✓ Scores Heatmap
✓ Scores by Factor

### Output Location

All visualizations saved to:
`/scratch/gpfs/MSALGANIK/mjs3/cap_7L_llama32_lora_comparison_2025-10-18/visualizations/`

**View results:**
Open `visualizations/index.html` in a browser to explore all visualizations interactively.

See `run-inspect-viz.log` for detailed generation log.
```

## Error Handling

**If no .eval files found:**
- Report error clearly
- Check common locations (.eval might be in unexpected directory)
- Suggest running evaluations first
- Exit gracefully

**If no experiment design documentation found:**
- Log error
- Ask user whether to continue with generic pre-built views only
- If user agrees: Generate pre-built views but warn that custom visualizations are skipped
- If user declines: Exit gracefully with instructions to add experiment documentation

**If inspect-viz not installed:**
- Report clear installation instructions:
  ```bash
  pip install inspect-viz
  ```
- Exit gracefully

**If evaluation data malformed:**
- Log which files couldn't be loaded
- Continue with valid data
- Report issues in summary
- Only fail if NO valid data

**If a visualization fails to generate:**
- Log the error with details
- Continue with remaining visualizations
- Report skipped visualizations in summary

**If output directory creation fails:**
- Report permission or disk space issues
- Exit gracefully

## Dependencies

**Required:**
- `inspect-viz` - Visualization library
  ```bash
  pip install inspect-viz
  ```
- `inspect-ai` - Already a cruijff_kit dependency (provides eval log reading)

**Automatically installed by inspect-viz:**
- `pandas` >= 2.2.2 (dataframe handling)
- `pyarrow` >= 15.0.0 (parquet file reading)
- `anywidget` >= 0.9.0 (interactive widgets)
- `narwhals` >= 1.15.1 (dataframe compatibility layer)

**Optional (future):**
- `playwright` - For PNG export
  ```bash
  pip install playwright
  playwright install
  ```

## Validation Before Completion

Before reporting success, verify:
- ✓ Experiment directory was located
- ✓ .eval files were found and loaded
- ✓ Experiment design was read (from experiment_summary.md or README.md)
- ✓ Custom plots were generated based on experimental design
- ✓ Pre-built views were generated
- ✓ Navigation index was created
- ✓ Output directory exists and contains HTML files
- ✓ Log file was created with complete history
- ✓ Final summary is accurate and helpful

## Important Notes

- **Flexible directory structure**: Works with both new (eval/logs/) and legacy (logs/) locations
- **Flexible design sources**: Reads from experiment_summary.md (new) or README.md (legacy)
- **Design required**: Experimental design documentation is required for meaningful visualizations
- **All visualizations are interactive**: Powered by inspect-viz widgets
- **Separate HTML files**: Each visualization is a standalone file for fast loading
- **This skill is a worker**: Can be called by `analyze-experiment` orchestrator or run standalone
- **Future enhancement**: Could generate Quarto Dashboard for single-page experience
- **Future enhancement**: Could export PNG versions alongside HTML (requires playwright)

## Integration Points

### Called by `analyze-experiment` (orchestrator, future)
- `analyze-experiment` calls `run-inspect-viz` after loading experiment metadata
- Passes experiment directory path
- Incorporates visualizations into analysis report

### Standalone usage
```bash
# User navigates to experiment directory
cd /scratch/gpfs/MSALGANIK/mjs3/cap_7L_llama32_lora_comparison_2025-10-18

# Invoke run-inspect-viz skill via Claude Code
# (Claude generates visualizations)
```

### Relationship to existing skills
- **Input from**: `run-inspect` (produces .eval files)
- **Input from**: `design-experiment` (experiment_summary.md - optional)
- **Called by**: `analyze-experiment` (future orchestrator)
- **Similar to**: `scaffold-inspect`, `run-inspect` (worker skills)

## References

- inspect-viz documentation: https://meridianlabs-ai.github.io/inspect_viz/
- inspect-viz GitHub: https://github.com/meridianlabs-ai/inspect_viz
- Inspect AI dataframe API: https://inspect.aisi.org.uk/dataframe.html
- Publishing plots: https://meridianlabs-ai.github.io/inspect_viz/publishing-plots.html

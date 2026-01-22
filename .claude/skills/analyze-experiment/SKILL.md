---
name: analyze-experiment
description: Generate visualizations from completed experiment evaluations using inspect-viz. Use after run-experiment to create interactive HTML plots from inspect-ai evaluation logs.
---

# Analyze Experiment

**STATUS: Currently only supports capitalization experiments.** Future work will extend support to all cruijff_kit experiments.

You help users visualize and analyze results from completed experiments by generating interactive HTML plots using inspect-viz pre-built views.

## Your Task

Generate visualizations from evaluation results:

1. Read experiment_summary.yaml to understand experimental design
2. Load evaluation logs from run directories
3. Infer appropriate visualization types based on experimental variables
4. Generate interactive HTML plots using inspect-viz
5. Log the process in analyze-experiment.jsonl

## Prerequisites

- experiment_summary.yaml exists (from design-experiment)
- Evaluations complete (from run-experiment)
- inspect-viz installed (`pip install inspect-viz`)
- Conda environment activated (from claude.local.md)

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
- `load_experiment_logs()` - Load logs with metadata extraction
- `evals_df_prep()` - Prepare eval-level dataframes

**See `data_loading.md` for:**
- How to construct subdirs list from experiment_summary.yaml
- Metadata extractor patterns for common experiments
- Preparing data for each view type

### 3. Infer Visualizations → `inference.md`

Map experimental variables to appropriate pre-built views:

| Variable Type | View |
|---------------|------|
| Multiple models | `scores_by_model` |
| Binary factor | `scores_by_factor` |
| Multiple tasks/conditions | `scores_by_task` |
| Model × task matrix | `scores_heatmap` |
| Multiple metrics | `scores_radar_by_task` |

**See `inference.md` for:**
- Complete inference logic
- How to detect variable types from experiment_summary.yaml
- Selecting appropriate views

### 4. Generate Plots → `generation.md`

Create visualizations using inspect-viz:
1. Wrap dataframes with `Data.from_dataframe()`
2. Call appropriate pre-built view functions
3. Save HTML with `write_html()`

**See `generation.md` for:**
- View function signatures and parameters
- Output file naming conventions
- Creating multiple plots per experiment

### 5. Logging → `logging.md`

Document process in `{experiment_dir}/analyze-experiment.jsonl`

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
| `scores_by_model` | Cross-model comparison | model column |

## Output Structure

After running, the experiment directory will contain:

```
{experiment_dir}/
├── analysis/
│   ├── scores_by_model.html
│   ├── scores_by_task.html
│   ├── scores_heatmap.html
│   └── ...
├── analyze-experiment.jsonl
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
- Log error details in analyze-experiment.jsonl
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
- ✓ Log file created (analyze-experiment.jsonl)

## Output Summary

After completing analysis, provide a summary:

```markdown
## Analyze Experiment Complete

Experiment: `{experiment_dir}`

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

Details recorded in `analyze-experiment.jsonl`
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

- **Currently only supports capitalization experiments** - metadata extractors and inference logic are specific to capitalization experiment structure
- **Must run after evaluations complete** - requires .eval files to exist
- Visualizations are independent - one failing doesn't stop others
- Output directory (`analysis/`) is created if it doesn't exist
- HTML files are standalone (no external dependencies to view)
- Uses helper functions from `tools/inspect/viz_helpers.py`
- Reference examples in `viz_examples/scripts/inspect_viz_examples.ipynb`

## Module Organization

| Module | Purpose |
|--------|---------|
| parsing.md | Experiment location and YAML parsing |
| data_loading.md | Helper functions for loading logs |
| inference.md | View selection logic |
| generation.md | Plot creation workflow |
| logging.md | JSONL audit trail specification |

## Implementation Reference

Working examples exist in `viz_examples/scripts/inspect_viz_examples.ipynb` demonstrating all supported pre-built views with capitalization experiments.

Helper functions are implemented in `tools/inspect/viz_helpers.py`.

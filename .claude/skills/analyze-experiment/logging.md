# Logging - analyze-experiment

**See [shared/logging_spec.md](../../shared/logging_spec.md) for complete format specification and general logging guidelines.**

This document covers analyze-experiment-specific logging practices.

---

## Log File Location

```
{experiment_dir}/analyze-experiment.log
```

Created during analysis to record data loading, visualization selection, and generation.

---

## Action Types

| Action Type | Purpose |
|-------------|---------|
| `LOCATE_EXPERIMENT` | Find and validate experiment directory |
| `PARSE_CONFIG` | Parse experiment_summary.yaml for run info |
| `LOAD_EVALS` | Load evaluation logs from run directories |
| `DEDUPLICATE` | Remove duplicate evals (keep most recent per model+epoch) |
| `SELECT_VIEW` | Record user's visualization choice |
| `GENERATE_PLOT` | Create individual visualization |
| `COMPLETE` | Mark analysis finished with summary |

---

## When to Log

### During Setup
- Which experiment directory is being used
- Config parsing results (runs found, variables)

### During Data Loading
- How many eval files found per run
- Any duplicates removed (and why)
- Final dataframe shape

### During Visualization
- Which view type user selected
- Each plot generated (with output path)
- Any generation errors

### On Completion
- Total plots generated
- Output directory location
- Duration

---

## Example Log Entries

```
[2026-01-29 14:00:00] LOCATE_EXPERIMENT
Details: Found experiment at /scratch/experiments/acs_income_balanced_2026-01-29
Result: experiment_summary.yaml exists

[2026-01-29 14:00:01] PARSE_CONFIG
Details: Parsed experiment_summary.yaml
Result: 2 runs (1B_balanced, 3B_balanced), 1 task (acs_income)

[2026-01-29 14:00:02] LOAD_EVALS
Details: Loading from 2 run directories
Result: 6 eval files loaded

[2026-01-29 14:00:03] DEDUPLICATE
Details: Checking for duplicate model+epoch combinations
Result: Kept 6 files, skipped 0 duplicates

[2026-01-29 14:00:10] SELECT_VIEW
Details: User selected visualization type
Result: scores_by_task

[2026-01-29 14:00:15] GENERATE_PLOT: scores_by_task
Details: Creating bar chart with match metric
Result: analysis/scores_by_task.html

[2026-01-29 14:00:20] COMPLETE
Details: All visualizations generated
Result: 2 plots created in analysis/
Duration: 20 seconds
```

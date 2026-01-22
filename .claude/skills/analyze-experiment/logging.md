# Logging: JSONL Audit Trail

This module specifies the logging format for the analyze-experiment skill.

## Log File Location

```
{experiment_dir}/analyze-experiment.jsonl
```

## Format

**JSON Lines (.jsonl)** - One JSON object per line.

Each entry contains:
- `action`: Action type (LOCATE, PARSE, LOAD, INFER, GENERATE, COMPLETE, ERROR)
- `timestamp`: ISO 8601 timestamp
- `status`: "success", "warning", or "error"
- Action-specific fields

## Action Types

### LOCATE

Logged when finding the experiment directory.

```json
{"action": "LOCATE", "timestamp": "2026-01-21T15:30:00Z", "experiment_dir": "/path/to/experiment", "method": "current_directory", "status": "success"}
```

```json
{"action": "LOCATE", "timestamp": "2026-01-21T15:30:00Z", "experiment_dir": "/path/to/experiment", "method": "user_provided", "status": "success"}
```

**Fields:**
- `experiment_dir`: Path to experiment directory
- `method`: How it was found ("current_directory", "user_provided")

### PARSE

Logged when parsing experiment_summary.yaml.

```json
{"action": "PARSE", "timestamp": "2026-01-21T15:30:01Z", "file": "experiment_summary.yaml", "runs_found": 4, "variables": ["model_size", "word_length"], "status": "success"}
```

```json
{"action": "PARSE", "timestamp": "2026-01-21T15:30:01Z", "file": "experiment_summary.yaml", "status": "error", "error": "File not found"}
```

**Fields:**
- `file`: File being parsed
- `runs_found`: Number of runs discovered
- `variables`: List of experimental variables found
- `error`: Error message (if status is "error")

### LOAD

Logged when loading evaluation logs.

```json
{"action": "LOAD", "timestamp": "2026-01-21T15:30:02Z", "experiment_path": "/path/to/experiment", "subdirs": ["run1", "run2", "run3", "run4"], "logs_found": 12, "status": "success"}
```

```json
{"action": "LOAD", "timestamp": "2026-01-21T15:30:02Z", "experiment_path": "/path/to/experiment", "subdirs": ["run1"], "logs_found": 0, "status": "warning", "warning": "No .eval files found in run1/eval/logs"}
```

**Fields:**
- `experiment_path`: Path to experiment
- `subdirs`: List of subdirectories searched
- `logs_found`: Total number of .eval files loaded
- `warning`: Warning message (if status is "warning")

### INFER

Logged when inferring visualization types.

```json
{"action": "INFER", "timestamp": "2026-01-21T15:30:03Z", "views_selected": ["scores_by_task", "scores_heatmap", "scores_radar_by_task"], "reasons": ["Found 3 tasks", "2 models × 3 tasks matrix", "Multiple metrics available"], "status": "success"}
```

```json
{"action": "INFER", "timestamp": "2026-01-21T15:30:03Z", "views_selected": [], "status": "warning", "warning": "Could not determine appropriate views", "available_columns": ["model", "score_match_accuracy"]}
```

**Fields:**
- `views_selected`: List of view types to generate
- `reasons`: Explanation for each selection
- `available_columns`: Columns found in dataframe (for debugging)

### GENERATE

Logged for each visualization generated (or attempted).

```json
{"action": "GENERATE", "timestamp": "2026-01-21T15:30:04Z", "view": "scores_by_task", "metric": "match", "output": "analysis/scores_by_task_match.html", "status": "success"}
```

```json
{"action": "GENERATE", "timestamp": "2026-01-21T15:30:05Z", "view": "scores_heatmap", "metric": "match", "output": "analysis/scores_heatmap_match.html", "status": "success"}
```

```json
{"action": "GENERATE", "timestamp": "2026-01-21T15:30:06Z", "view": "scores_by_factor", "metric": "includes", "output": null, "status": "error", "error": "Missing required column: factor"}
```

**Fields:**
- `view`: View type being generated
- `metric`: Metric being plotted (match, includes, etc.)
- `output`: Output file path (relative to experiment_dir), null if failed
- `error`: Error message (if status is "error")

### COMPLETE

Logged when skill completes successfully.

```json
{"action": "COMPLETE", "timestamp": "2026-01-21T15:30:10Z", "visualizations_generated": 6, "visualizations_failed": 1, "output_dir": "analysis/", "status": "success"}
```

**Fields:**
- `visualizations_generated`: Count of successful generations
- `visualizations_failed`: Count of failed generations
- `output_dir`: Directory containing output files

### ERROR

Logged for fatal errors that prevent completion.

```json
{"action": "ERROR", "timestamp": "2026-01-21T15:30:00Z", "stage": "PARSE", "error": "experiment_summary.yaml not found", "status": "error"}
```

**Fields:**
- `stage`: Which stage the error occurred in
- `error`: Error message

## Example Complete Log

```jsonl
{"action": "LOCATE", "timestamp": "2026-01-21T15:30:00Z", "experiment_dir": "/scratch/experiments/cap_wordlen_2026-01-12", "method": "current_directory", "status": "success"}
{"action": "PARSE", "timestamp": "2026-01-21T15:30:01Z", "file": "experiment_summary.yaml", "runs_found": 2, "variables": ["model_size", "word_length"], "status": "success"}
{"action": "LOAD", "timestamp": "2026-01-21T15:30:02Z", "experiment_path": "/scratch/experiments/cap_wordlen_2026-01-12", "subdirs": ["Llama-3.2-1B-Instruct_5L", "Llama-3.2-3B-Instruct_5L"], "logs_found": 6, "status": "success"}
{"action": "INFER", "timestamp": "2026-01-21T15:30:03Z", "views_selected": ["scores_by_task", "scores_heatmap", "scores_radar_by_task"], "reasons": ["Found 3 tasks", "2 models × 3 tasks matrix", "Multiple metrics"], "status": "success"}
{"action": "GENERATE", "timestamp": "2026-01-21T15:30:04Z", "view": "scores_by_task", "metric": "match", "output": "analysis/scores_by_task_match.html", "status": "success"}
{"action": "GENERATE", "timestamp": "2026-01-21T15:30:05Z", "view": "scores_by_task", "metric": "includes", "output": "analysis/scores_by_task_includes.html", "status": "success"}
{"action": "GENERATE", "timestamp": "2026-01-21T15:30:06Z", "view": "scores_heatmap", "metric": "match", "output": "analysis/scores_heatmap_match.html", "status": "success"}
{"action": "GENERATE", "timestamp": "2026-01-21T15:30:07Z", "view": "scores_heatmap", "metric": "includes", "output": "analysis/scores_heatmap_includes.html", "status": "success"}
{"action": "GENERATE", "timestamp": "2026-01-21T15:30:08Z", "view": "scores_radar_by_task", "metric": "combined", "output": "analysis/scores_radar.html", "status": "success"}
{"action": "COMPLETE", "timestamp": "2026-01-21T15:30:09Z", "visualizations_generated": 5, "visualizations_failed": 0, "output_dir": "analysis/", "status": "success"}
```

## Writing Logs

Use Python's json module to write log entries:

```python
import json
from datetime import datetime

def log_action(log_file, action, **kwargs):
    """Append a log entry to the JSONL file."""
    entry = {
        "action": action,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **kwargs
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

# Usage
log_file = os.path.join(experiment_dir, "analyze-experiment.jsonl")
log_action(log_file, "LOCATE", experiment_dir=experiment_dir, method="current_directory", status="success")
log_action(log_file, "PARSE", file="experiment_summary.yaml", runs_found=4, variables=["model", "task"], status="success")
```

## Reading Logs

Parse the JSONL file for reporting or debugging:

```python
def read_log(log_file):
    """Read all entries from a JSONL log file."""
    entries = []
    with open(log_file) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries

# Filter by action type
def get_actions(entries, action_type):
    return [e for e in entries if e["action"] == action_type]
```

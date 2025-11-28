# Logging - design-experiment

This document covers design-experiment-specific logging practices.

---

## Log File Location

`{experiment_dir}/design-experiment.jsonl`

Created during the planning workflow to record all verification steps, calculations, and decisions.

---

## File Format: JSON Lines (.jsonl)

Each line is a complete JSON object representing one log entry. This format is:
- **Machine-readable**: Easy to parse with standard JSON libraries
- **Streamable**: Can be written incrementally during skill execution
- **Queryable**: Easy to filter/analyze with tools like `jq`, `grep`, or pandas

---

## Log Entry Schema

Every log entry must follow this base schema:

```json
{
  "timestamp": "ISO8601 string",      // REQUIRED: When action occurred
  "action": "ACTION_TYPE",            // REQUIRED: What happened (see Action Types)
  "result": "success|failure|warning", // REQUIRED: Outcome
  "duration_ms": 123,                 // OPTIONAL: How long action took
  "...": "..."                        // ACTION-SPECIFIC: Additional fields
}
```

---

## Action Types

### START_DESIGN

Marks the beginning of experiment design.

```json
{
  "timestamp": "2025-10-22T14:30:00.000Z",
  "action": "START_DESIGN",
  "result": "success",
  "experiment_name": "cap_4L_lora_rank_comparison",
  "experiment_type": "sanity_check",
  "experiment_dir": "/scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_rank_comparison"
}
```

---

### VERIFY_MODEL

Check that a model directory exists and get its size.

```json
{
  "timestamp": "2025-10-22T14:30:01.123Z",
  "action": "VERIFY_MODEL",
  "result": "success",
  "resource_type": "model",
  "resource_name": "Llama-3.2-1B-Instruct",
  "path": "/scratch/gpfs/MSALGANIK/niznik/llms/Meta-Llama-3.2-1B-Instruct",
  "command": "ls -lh /scratch/gpfs/MSALGANIK/niznik/llms/Meta-Llama-3.2-1B-Instruct",
  "size_gb": 4.0,
  "duration_ms": 45
}
```

---

### VERIFY_DATASET

Check that a dataset file exists and get its size.

```json
{
  "timestamp": "2025-10-22T14:30:01.456Z",
  "action": "VERIFY_DATASET",
  "result": "success",
  "resource_type": "dataset",
  "path": "/home/sarahep/cruijff_kit/data/green/words_5L_80P_1000.json",
  "command": "ls -lh /home/sarahep/cruijff_kit/data/green/words_5L_80P_1000.json",
  "size_kb": 84,
  "duration_ms": 23
}
```

---

### COUNT_DATASET_SAMPLES

Count the number of samples in each dataset split.

```json
{
  "timestamp": "2025-10-22T14:30:01.789Z",
  "action": "COUNT_DATASET_SAMPLES",
  "result": "success",
  "dataset": "words_5L_80P_1000.json",
  "command": "jq '.train | length' /home/sarahep/cruijff_kit/data/green/words_5L_80P_1000.json",
  "train_samples": 1000,
  "validation_samples": 200,
  "test_samples": 200,
  "duration_ms": 156
}
```

---

### VERIFY_EVAL_TASK

Check that an evaluation task script exists.

```json
{
  "timestamp": "2025-10-22T14:30:02.012Z",
  "action": "VERIFY_EVAL_TASK",
  "result": "success",
  "task_name": "capitalization",
  "script_path": "/home/sarahep/cruijff_kit/experiments/capitalization/cap_task.py",
  "command": "ls -lh /home/sarahep/cruijff_kit/experiments/capitalization/cap_task.py",
  "size_kb": 12,
  "duration_ms": 34
}
```

---

### SEARCH_PRIOR_RUNS

Search for prior runs to extract training speed estimates.

```json
{
  "timestamp": "2025-10-22T14:30:02.234Z",
  "action": "SEARCH_PRIOR_RUNS",
  "result": "success",
  "search_pattern": "find /scratch/gpfs/MSALGANIK/niznik -name 'slurm-*.out' -path '*/ck-out-*'",
  "command": "find /scratch/gpfs/MSALGANIK/niznik -name 'slurm-*.out' -path '*/ck-out-*' | head -10",
  "found_count": 3,
  "paths": [
    "/scratch/gpfs/MSALGANIK/niznik/prior_exp/run1/slurm-123.out",
    "/scratch/gpfs/MSALGANIK/niznik/prior_exp/run2/slurm-456.out"
  ],
  "duration_ms": 890
}
```

---

### EXTRACT_TRAINING_SPEED

Extract training speed from prior run logs.

```json
{
  "timestamp": "2025-10-22T14:30:03.567Z",
  "action": "EXTRACT_TRAINING_SPEED",
  "result": "success",
  "prior_run": "/scratch/gpfs/MSALGANIK/niznik/prior_exp/run1/slurm-123.out",
  "command": "grep -E '[0-9.]+it/s' /scratch/.../slurm-123.out | tail -20",
  "iterations_per_sec": 2.5,
  "estimated_seconds_per_epoch": 120,
  "duration_ms": 234
}
```

---

### CALCULATE_TRAINING_TIME

Compute estimated training time for all runs.

```json
{
  "timestamp": "2025-10-22T14:30:04.890Z",
  "action": "CALCULATE_TRAINING_TIME",
  "result": "success",
  "basis": "prior_run_average",
  "per_epoch_seconds": 120,
  "epochs": 1,
  "num_runs": 2,
  "total_seconds": 240,
  "total_minutes": 4
}
```

---

### CHECK_DISK_SPACE

Check available disk space on the target filesystem.

```json
{
  "timestamp": "2025-10-22T14:30:05.123Z",
  "action": "CHECK_DISK_SPACE",
  "result": "success",
  "command": "df -h /scratch/gpfs/MSALGANIK/niznik",
  "available_gb": 5120,
  "used_gb": 2048,
  "total_gb": 7168,
  "duration_ms": 67
}
```

---

### CALCULATE_DISK_USAGE

Estimate disk space required for all checkpoints.

```json
{
  "timestamp": "2025-10-22T14:30:05.456Z",
  "action": "CALCULATE_DISK_USAGE",
  "result": "success",
  "per_checkpoint_gb": 2.5,
  "epochs": 1,
  "num_runs": 2,
  "total_checkpoints": 2,
  "total_gb": 5.0
}
```

---

### GENERATE_RUNS

Generate the list of runs from variables and controls.

```json
{
  "timestamp": "2025-10-22T14:30:06.789Z",
  "action": "GENERATE_RUNS",
  "result": "success",
  "method": "cartesian_product",
  "variables": {
    "lora_rank": [4, 8]
  },
  "generated_count": 2,
  "control_runs": 1,
  "total_runs": 3
}
```

---

### GENERATE_EVAL_MATRIX

Generate the evaluation matrix (which runs × tasks × epochs).

```json
{
  "timestamp": "2025-10-22T14:30:07.012Z",
  "action": "GENERATE_EVAL_MATRIX",
  "result": "success",
  "num_runs": 3,
  "num_tasks": 1,
  "total_evaluations": 3
}
```

---

### CREATE_YAML

Write the experiment_summary.yaml file.

```json
{
  "timestamp": "2025-10-22T14:30:08.234Z",
  "action": "CREATE_YAML",
  "result": "success",
  "file_path": "/scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_rank_comparison/experiment_summary.yaml",
  "size_bytes": 1456
}
```

---

### CREATE_LOG

Write the design-experiment.jsonl log file (this entry).

```json
{
  "timestamp": "2025-10-22T14:30:08.567Z",
  "action": "CREATE_LOG",
  "result": "success",
  "file_path": "/scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_rank_comparison/design-experiment.jsonl",
  "size_bytes": 2134,
  "num_entries": 14
}
```

---

### COMPLETE_DESIGN

Mark the end of experiment design.

```json
{
  "timestamp": "2025-10-22T14:30:08.890Z",
  "action": "COMPLETE_DESIGN",
  "result": "success",
  "experiment_name": "cap_4L_lora_rank_comparison",
  "total_duration_seconds": 8.89,
  "files_created": [
    "experiment_summary.yaml",
    "design-experiment.jsonl"
  ]
}
```

---

## Error Handling

When `result: "failure"` or `result: "warning"`, include additional error fields:

```json
{
  "timestamp": "2025-10-22T14:30:01.123Z",
  "action": "VERIFY_MODEL",
  "result": "failure",
  "resource_type": "model",
  "resource_name": "Llama-3.2-1B-Instruct",
  "path": "/scratch/gpfs/MSALGANIK/niznik/llms/Meta-Llama-3.2-1B-Instruct",
  "command": "ls -lh /scratch/gpfs/MSALGANIK/niznik/llms/Meta-Llama-3.2-1B-Instruct",
  "error_type": "FileNotFoundError",
  "error_message": "Model directory does not exist",
  "duration_ms": 45
}
```

**Required error fields:**
- `error_type`: Classification of the error
- `error_message`: Human-readable description

---

## Writing Logs

### Incrementally Append Entries

Write log entries as soon as actions complete. Do not wait until the end to write all logs at once.

### Use ISO 8601 Timestamps

```python
from datetime import datetime
timestamp = datetime.utcnow().isoformat() + 'Z'
```

### One Entry Per Line

Each JSON object must be on a single line (no pretty-printing):

```jsonl
{"timestamp":"2025-10-22T14:30:00.000Z","action":"START_DESIGN","result":"success"}
{"timestamp":"2025-10-22T14:30:01.123Z","action":"VERIFY_MODEL","result":"success","resource_name":"Llama-3.2-1B-Instruct"}
```

### Valid JSON

Each line must parse as valid JSON independently.

---

## Example Complete Log

Here's a complete example log from an experiment design session:

```jsonl
{"timestamp":"2025-10-22T14:30:00.000Z","action":"START_DESIGN","result":"success","experiment_name":"cap_4L_lora_rank_comparison","experiment_type":"sanity_check","experiment_dir":"/scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_rank_comparison"}
{"timestamp":"2025-10-22T14:30:01.123Z","action":"VERIFY_MODEL","result":"success","resource_type":"model","resource_name":"Llama-3.2-1B-Instruct","path":"/scratch/gpfs/MSALGANIK/niznik/llms/Meta-Llama-3.2-1B-Instruct","command":"ls -lh /scratch/gpfs/MSALGANIK/niznik/llms/Meta-Llama-3.2-1B-Instruct","size_gb":4.0,"duration_ms":45}
{"timestamp":"2025-10-22T14:30:01.456Z","action":"VERIFY_DATASET","result":"success","resource_type":"dataset","path":"/home/sarahep/cruijff_kit/data/green/words_5L_80P_1000.json","command":"ls -lh /home/sarahep/cruijff_kit/data/green/words_5L_80P_1000.json","size_kb":84,"duration_ms":23}
{"timestamp":"2025-10-22T14:30:01.789Z","action":"COUNT_DATASET_SAMPLES","result":"success","dataset":"words_5L_80P_1000.json","command":"jq '.train | length' /home/sarahep/cruijff_kit/data/green/words_5L_80P_1000.json","train_samples":1000,"validation_samples":200,"test_samples":200,"duration_ms":156}
{"timestamp":"2025-10-22T14:30:02.012Z","action":"VERIFY_EVAL_TASK","result":"success","task_name":"capitalization","script_path":"/home/sarahep/cruijff_kit/experiments/capitalization/cap_task.py","command":"ls -lh /home/sarahep/cruijff_kit/experiments/capitalization/cap_task.py","size_kb":12,"duration_ms":34}
{"timestamp":"2025-10-22T14:30:02.234Z","action":"SEARCH_PRIOR_RUNS","result":"success","search_pattern":"find /scratch/gpfs/MSALGANIK/niznik -name 'slurm-*.out' -path '*/ck-out-*'","command":"find /scratch/gpfs/MSALGANIK/niznik -name 'slurm-*.out' -path '*/ck-out-*' | head -10","found_count":3,"paths":["/scratch/gpfs/MSALGANIK/niznik/prior_exp/run1/slurm-123.out"],"duration_ms":890}
{"timestamp":"2025-10-22T14:30:03.567Z","action":"EXTRACT_TRAINING_SPEED","result":"success","prior_run":"/scratch/gpfs/MSALGANIK/niznik/prior_exp/run1/slurm-123.out","command":"grep -E '[0-9.]+it/s' /scratch/.../slurm-123.out | tail -20","iterations_per_sec":2.5,"estimated_seconds_per_epoch":120,"duration_ms":234}
{"timestamp":"2025-10-22T14:30:04.890Z","action":"CALCULATE_TRAINING_TIME","result":"success","basis":"prior_run_average","per_epoch_seconds":120,"epochs":1,"num_runs":2,"total_seconds":240,"total_minutes":4}
{"timestamp":"2025-10-22T14:30:05.123Z","action":"CHECK_DISK_SPACE","result":"success","command":"df -h /scratch/gpfs/MSALGANIK/niznik","available_gb":5120,"used_gb":2048,"total_gb":7168,"duration_ms":67}
{"timestamp":"2025-10-22T14:30:05.456Z","action":"CALCULATE_DISK_USAGE","result":"success","per_checkpoint_gb":2.5,"epochs":1,"num_runs":2,"total_checkpoints":2,"total_gb":5.0}
{"timestamp":"2025-10-22T14:30:06.789Z","action":"GENERATE_RUNS","result":"success","method":"cartesian_product","variables":{"lora_rank":[4,8]},"generated_count":2,"control_runs":1,"total_runs":3}
{"timestamp":"2025-10-22T14:30:07.012Z","action":"GENERATE_EVAL_MATRIX","result":"success","num_runs":3,"num_tasks":1,"total_evaluations":3}
{"timestamp":"2025-10-22T14:30:08.234Z","action":"CREATE_YAML","result":"success","file_path":"/scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_rank_comparison/experiment_summary.yaml","size_bytes":1456}
{"timestamp":"2025-10-22T14:30:08.567Z","action":"CREATE_LOG","result":"success","file_path":"/scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_rank_comparison/design-experiment.jsonl","size_bytes":2134,"num_entries":14}
{"timestamp":"2025-10-22T14:30:08.890Z","action":"COMPLETE_DESIGN","result":"success","experiment_name":"cap_4L_lora_rank_comparison","total_duration_seconds":8.89,"files_created":["experiment_summary.yaml","design-experiment.jsonl"]}
```

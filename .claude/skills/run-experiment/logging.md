# Logging - run-experiment

**See [shared/logging_spec.md](../../shared/logging_spec.md) for complete format specification and general logging guidelines.**

This document covers run-experiment-specific logging practices.

---

## Log File Locations

Run-experiment creates tool-specific logs:

- **Torchtune execution:** `{experiment_dir}/logs/run-torchtune.log`
- **Inspect-ai execution:** `{experiment_dir}/logs/run-inspect.log`

Both created during job execution to record job submissions, status changes, and completion.

---

## Action Types

### Torchtune Actions

| Action Type | Purpose |
|-------------|---------|
| `DISCOVER_EXPERIMENT` | Find and validate experiment directory |
| `IDENTIFY_RUNS` | List runs with finetune.slurm scripts |
| `SUBMIT_JOB` | Submit individual fine-tuning job to SLURM |
| `STATUS_CHECK` | Poll SLURM for job statuses |
| `STATE_CHANGE` | Record job state transition |
| `COMPUTE_METRICS` | Record seff compute metrics when job reaches terminal state |
| `RESOURCE_STATUS` | Report GPU utilization from gpu_metrics.csv during monitoring |
| `ALL_COMPLETE` | Mark all fine-tuning jobs finished |

### Inspect-ai Actions

| Action Type | Purpose |
|-------------|---------|
| `DISCOVER_EXPERIMENT` | Find and validate experiment directory |
| `VERIFY_FINETUNING` | Check that fine-tuning completed |
| `VERIFY_CHECKPOINTS` | Check that model checkpoints exist |
| `IDENTIFY_EVALS` | List evaluations to execute |
| `SUBMIT_EVAL` | Submit individual evaluation job to SLURM |
| `STATUS_CHECK` | Poll SLURM for job statuses |
| `STATE_CHANGE` | Record job state transition |
| `COMPUTE_METRICS` | Record seff compute metrics when eval job reaches terminal state |
| `RESOURCE_STATUS` | Report GPU utilization from gpu_metrics.csv during monitoring |
| `ALL_COMPLETE` | Mark all evaluations finished |

---

## When to Log

### During Parsing
- Which experiment_summary.yaml is being read
- Extracted run/evaluation information

### During Selection
- Which runs/evaluations will be submitted
- Any runs/evaluations being skipped (and why)

### During Job Submission
- Each job submitted (with job ID and timestamp)
- 5-second stagger delays between submissions
- Any submission errors

### During Monitoring
- Status checks (every 60 seconds)
- State changes: PENDING → RUNNING → COMPLETED/FAILED
- Any monitoring errors

### During Validation
- Checkpoint verification (inspect-ai only)
- Completion verification
- Final summary with duration and success/failure counts

---

## Important Notes

**Job ID tracking:** Always log job IDs when jobs are submitted - critical for monitoring and debugging

**State changes:** Log all state transitions with timestamps to track execution time

**Staggering:** Note 5-second delays between submissions to explain timeline

**Errors:** Log SLURM error messages when jobs fail

---

## Example Log Entries

### Torchtune Execution

```
[2025-11-11 11:00:00] DISCOVER_EXPERIMENT
Details: Found experiment at /path/to/experiment
Result: 4 runs configured

[2025-11-11 11:00:05] SUBMIT_JOB: r8_lr1e-5
Details: sbatch finetune.slurm
Result: Job ID 1234567

[2025-11-11 11:00:10] SUBMIT_JOB: r8_lr5e-5
Details: sbatch finetune.slurm (5-second stagger)
Result: Job ID 1234568

[2025-11-11 11:02:00] STATE_CHANGE: r8_lr1e-5
Details: PENDING → RUNNING
Result: Training started

[2025-11-11 11:45:00] ALL_COMPLETE
Details: All 4 jobs reached terminal states
Result: 4 COMPLETED, 0 FAILED
Duration: 45 minutes
```

### Inspect-ai Execution

```
[2025-11-11 12:00:01] VERIFY_FINETUNING
Details: Checking torchtune jobs via sacct
Result: All fine-tuning jobs COMPLETED

[2025-11-11 12:00:02] VERIFY_CHECKPOINTS
Details: Checking epoch_0, epoch_1 for 4 runs
Result: All 8 checkpoints exist

[2025-11-11 12:00:05] SUBMIT_EVAL: r8_lr1e-5/capitalization/epoch0
Details: sbatch capitalization_epoch0.slurm
Result: Job ID 2345678

[2025-11-11 12:05:00] STATE_CHANGE: r8_lr1e-5/capitalization/epoch0
Details: RUNNING → COMPLETED
Result: Evaluation finished

[2025-11-11 12:15:00] ALL_COMPLETE
Details: All 16 evaluations reached terminal states
Result: 16 COMPLETED, 0 FAILED
Duration: 15 minutes
```

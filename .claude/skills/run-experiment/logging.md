# Logging - run-experiment

**See [shared/logging_spec.md](../shared/logging_spec.md) for complete format specification and general logging guidelines.**

This document covers run-experiment-specific logging practices.

---

## Log File Locations

Run-experiment creates three logs:

- **Orchestration:** `{experiment_directory}/logs/run-experiment.log` — high-level flow (which stages ran, when), kept short
- **Torchtune execution:** `{experiment_directory}/logs/run-torchtune.log` — detailed per-job records (submissions, status changes, completion)
- **Inspect-ai execution:** `{experiment_directory}/logs/run-inspect.log` — detailed per-job records (submissions, status changes, completion)

Detailed sub-logs are created during job execution. The orchestration log records skill-level transitions.

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
| `MONITOR_DETACHED` | Watcher exited via SIGINT / SIGTERM / sentinel; jobs continue running |
| `MONITOR_CONFIG` | Watcher applied a live edit from `logs/monitor.json` (poll/stagger/max_submit) |
| `OOM_RETRY` | A fine-tune was detected as OOM (SLURM `OUT_OF_MEMORY`, or `FAILED` with `torch.OutOfMemoryError` / "CUDA out of memory" in its `slurm-<jid>.out`) and was resubmitted with `batch_size` halved in its `finetune.yaml`. Up to `max_oom_retries` (default 3) per run. State entry records `oom_detected_via` as either `"slurm_state"` or `"cuda_log"`. Disable per-invocation with `--no-retry`. |
| `OOM_RETRY_SUBMIT_FAILED` | An OOM retry tried to resubmit but `sbatch` itself errored; the run is left in its prior failure state (`OUT_OF_MEMORY` or `FAILED`) and the original cause should be investigated before further retries. |
| `OOM_EXHAUSTED` | A fine-tune hit `max_oom_retries` and remains in its last OOM-equivalent state. No active escalation in this release; manual intervention required. |

### Inspect-ai Actions

| Action Type | Purpose |
|-------------|---------|
| `DISCOVER_EXPERIMENT` | Find and validate experiment directory |
| `VERIFY_FINETUNING` | Check that fine-tuning completed |
| `VERIFY_CHECKPOINTS` | Check that model checkpoints exist |
| `CHECKPOINT_MISSING` | Model checkpoint not found for a run |
| `CACHE_PREBUILD` | Start pre-building HF datasets cache |
| `CACHE_BUILT` | Single dataset cache built successfully |
| `CACHE_PREBUILD_COMPLETE` | All dataset caches ready |
| `IDENTIFY_EVALS` | List evaluations to execute |
| `SUBMIT_EVAL` | Submit individual evaluation job to SLURM |
| `ALL_SUBMITTED` | All evaluation jobs submitted |
| `STATUS_CHECK` | Poll SLURM for job statuses |
| `STATE_CHANGE` | Record job state transition |
| `COMPUTE_METRICS` | Record seff compute metrics when eval job reaches terminal state |
| `RESOURCE_STATUS` | Report GPU utilization from gpu_metrics.csv during monitoring |
| `ALL_COMPLETE` | Mark all evaluations finished |
| `MONITOR_DETACHED` | Watcher exited via SIGINT / SIGTERM / sentinel; jobs continue running |
| `MONITOR_CONFIG` | Watcher applied a live edit from `logs/monitor.json` (poll/stagger/max_submit) |

---

## When to Log

### During Parsing
- Which experiment_summary.yaml is being read
- Extracted run/evaluation information

### During Selection
- Which runs/evaluations will be submitted
- Any runs/evaluations being skipped (and why)

### During Cache Pre-building (inspect-ai only)
- Which datasets need cache pre-building
- Each dataset cache built
- Any cache build failures

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

**Staggering:** Torchtune uses 5-second stagger between submissions to prevent cache collision; inspect submits all at once (cache is pre-built)

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
Job ID: 1234567
Result: success

[2025-11-11 11:00:10] SUBMIT_JOB: r8_lr5e-5
Details: sbatch finetune.slurm (5-second stagger)
Job ID: 1234568
Result: success

[2025-11-11 11:02:00] STATE_CHANGE: r8_lr1e-5
Details: PENDING → RUNNING
Result: Training started

[2025-11-11 11:45:00] ALL_COMPLETE
Details: All 4 jobs reached terminal states
Result: 4 COMPLETED, 0 FAILED
Duration: 45 minutes
```

### Live Monitor Settings (MONITOR_CONFIG)

The watcher re-reads `<experiment_directory>/logs/monitor.json` on every poll
iteration. When at least one knob changes, a `MONITOR_CONFIG` block lands
in the per-tool log. Unchanged knobs are still shown so the active
picture is complete. The block contains no `Job ID:` line, so the
`SUBMIT_JOB:` / `SUBMIT_EVAL:` harvest regex used by `explore-experiment`
does not match it.

```
[2025-11-11 11:30:00] MONITOR_CONFIG: applied
Details: poll_sec=30 (was 60); stagger_sec=5 (unchanged); max_submit=25 (unchanged)
Result: settings updated from logs/monitor.json
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
Job ID: 2345678
Result: success

[2025-11-11 12:05:00] STATE_CHANGE: r8_lr1e-5/capitalization/epoch0
Details: RUNNING → COMPLETED
Result: Evaluation finished

[2025-11-11 12:15:00] ALL_COMPLETE
Details: All 16 evaluations reached terminal states
Result: 16 COMPLETED, 0 FAILED
Duration: 15 minutes
```

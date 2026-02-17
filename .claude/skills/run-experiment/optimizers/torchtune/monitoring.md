# Monitoring - Execution Tracking

Monitor SLURM jobs until all reach terminal states.

## Polling Strategy

**Interval:** 60 seconds (1 minute)

**Stop condition:** All jobs reach terminal states (COMPLETED, FAILED, CANCELLED, TIMEOUT)

## Query Commands

### For Running Jobs

```bash
squeue -u $USER -o "%.18i %.9P %.50j %.8T %.10M %.6D"
```

Returns:
- JOBID
- PARTITION
- NAME (job name)
- STATE (PENDING, RUNNING, etc.)
- TIME (elapsed time)
- NODES

### For Completed Jobs

Jobs disappear from squeue when complete. Use sacct:

```bash
sacct -j {job_id} --format=JobID,State,Start,End,Elapsed
```

## Status Update Process

On each poll (every 60 seconds):

### 1. Query All Jobs at Once

**Good:** `squeue -u $USER` (single call)
**Avoid:** Querying jobs one by one

### 2. Match Job IDs

Cross-reference squeue results with tracked job IDs to identify which belong to this experiment.

### 3. Detect State Changes

Compare current state to previous state for each job.

**If state changed:**
- Log the change with timestamp
- Update run state tracking
- Record new status

### 4. Update Run State

**Note:** Status tracking is maintained in a separate state file (e.g., `run-experiment-state.json`), not in experiment_summary.yaml which is static configuration.

Track state transitions:

**PENDING → RUNNING:**
```markdown
| r8_lr1e-5 | RUNNING | 12345678 | 2025-11-11 14:35:22 | - | - |
```
Add "Started" timestamp

**RUNNING → COMPLETED:**
```markdown
| r8_lr1e-5 | COMPLETED | 12345678 | 2025-11-11 14:35:22 | 2025-11-11 14:43:25 | 8m 3s |
```
Add "Completed" timestamp and elapsed time

**RUNNING → FAILED:**
```markdown
| r8_lr1e-5 | FAILED | 12345678 | 2025-11-11 14:35:22 | 2025-11-11 14:40:00 | Check slurm-12345678.out |
```
Add note to check SLURM log

## Terminal States

Stop monitoring when job reaches:
- **COMPLETED** - Finished successfully
- **FAILED** - Error occurred
- **CANCELLED** - User canceled job
- **TIMEOUT** - Exceeded time limit

## Logging

### Status Checks (Brief)

```
[YYYY-MM-DD HH:MM:SS] STATUS_CHECK: Polling SLURM
Result: 8 jobs found - 6 PENDING, 2 RUNNING, 0 COMPLETED
```

### State Changes (Detailed)

```
[YYYY-MM-DD HH:MM:SS] STATE_CHANGE: r8_lr1e-5
Previous: PENDING
Current: RUNNING
Started: 2025-11-11 14:35:22
Action: Updated run state
```

```
[YYYY-MM-DD HH:MM:SS] STATE_CHANGE: r8_lr1e-5
Previous: RUNNING
Current: COMPLETED
Completed: 2025-11-11 14:43:25
Elapsed: 8m 3s
Action: Updated run state
```

## Post-Completion Metrics

When a job transitions to COMPLETED or FAILED, capture compute metrics with `seff`. SLURM accounting has a lag — if seff returns no data, wait 30 seconds and retry once.

```bash
# First attempt
seff {job_id}
# If output shows "not available" or empty, wait and retry:
sleep 30
seff {job_id}
```

Log the metrics:

```
[YYYY-MM-DD HH:MM:SS] COMPUTE_METRICS: {run_name}
Job ID: {job_id}
GPU Time: {wall_time} / {time_limit} ({time_eff}%)
Memory: {mem_used} / {mem_requested}
CPU Efficiency: {cpu_eff}%
```

If the wall time exceeds the time limit (job state TIMEOUT), flag it:

```
[YYYY-MM-DD HH:MM:SS] COMPUTE_METRICS: {run_name}
Job ID: {job_id}
GPU Time: {wall_time} / {time_limit} (**EXCEEDED**)
Memory: {mem_used} / {mem_requested}
CPU Efficiency: {cpu_eff}%
⚠️ Time limit exceeded — consider increasing training time for this model
```

## Completion Detection

When all jobs reach terminal states:

```
[YYYY-MM-DD HH:MM:SS] ALL_COMPLETE: Monitoring finished
Summary: 8 jobs completed - 7 COMPLETED, 1 FAILED
Total time: 25 minutes
```

## GPU Status (Real-Time Feedback)

Every 5th poll (~5 minutes) for RUNNING jobs, read the latest GPU metrics from the shared filesystem:

```bash
tail -1 {output_dir}/gpu_metrics.csv
```

Log as:

```
[YYYY-MM-DD HH:MM:SS] RESOURCE_STATUS: {run_name}
GPU Utilization: {gpu_util}% | GPU Memory: {gpu_mem_used}/{gpu_mem_total} GB
```

If `gpu_metrics.csv` doesn't exist yet (job just started), skip silently.

## Error Handling

**If monitoring query fails:**
- Log the error
- Wait until next interval
- Retry query
- If repeated failures, alert user

**If job fails:**
- Record failure in status
- Continue monitoring other jobs
- Include in final summary

## Best Practices

**Efficient:**
- Single squeue call for all jobs
- Poll every 60 seconds (not faster)
- Stop when all terminal

**Avoid:**
- Polling faster than 60 seconds (wastes resources)
- Querying jobs individually (slow)
- Continuing after all complete (unnecessary)

## Next Stage

Pass completion status to validation.md for final checks.

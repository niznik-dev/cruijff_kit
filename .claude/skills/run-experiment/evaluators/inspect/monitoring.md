# Monitoring - Execution Tracking

Monitor SLURM evaluation jobs until all reach terminal states.

## Polling Strategy

**Interval:** 60 seconds (1 minute)

**Stop condition:** All jobs reach terminal states (COMPLETED, FAILED, CANCELLED, TIMEOUT)

**Note:** Evaluation jobs are typically faster than fine-tuning (5-10 minutes vs hours)

## Query Commands

### For Running Jobs

```bash
squeue -u $USER -o "%.18i %.9P %.50j %.8T %.10M %.6D"
```

Returns:
- JOBID
- PARTITION
- NAME (job name - includes "eval" for identification)
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

Cross-reference squeue results with tracked evaluation job IDs.

### 3. Detect State Changes

Compare current state to previous state for each evaluation.

**If state changed:**
- Log the change with timestamp
- Update experiment_summary.yaml
- Record new status

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
Result: 8 jobs found - 4 PENDING, 4 RUNNING, 0 COMPLETED
```

### State Changes (Detailed)

```
[YYYY-MM-DD HH:MM:SS] STATE_CHANGE: r8_lr1e-5/capitalization/epoch0
Previous: PENDING
Current: RUNNING
Job ID: 12345690
Action: Updated experiment_summary.yaml
```

```
[YYYY-MM-DD HH:MM:SS] STATE_CHANGE: r8_lr1e-5/capitalization/epoch0
Previous: RUNNING
Current: COMPLETED
Completed: 2025-11-11 14:50:12
Elapsed: 4m 52s
Action: Updated experiment_summary.yaml
Result log: r8_lr1e-5/eval/logs/
```

## Completion Detection

When all evaluation jobs reach terminal states:

```
[YYYY-MM-DD HH:MM:SS] ALL_COMPLETE: Monitoring finished
Summary: 8 evaluations completed - 8 COMPLETED, 0 FAILED
Total time: 15 minutes
```

## Error Handling

**If monitoring query fails:**
- Log the error
- Wait until next interval
- Retry query
- If repeated failures, alert user

**If evaluation fails:**
- Record failure in status
- Check SLURM log and inspect-ai logs for details
- Continue monitoring other jobs
- Include in final summary

## Best Practices

**Efficient:**
- Single squeue call for all jobs
- Poll every 60 seconds (not faster)
- Stop when all terminal

**Avoid:**
- Polling faster than 60 seconds
- Querying jobs individually
- Continuing after all complete

## Next Stage

Pass completion status to validation.md for final checks.

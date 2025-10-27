# Run Torchtune

You help users submit all fine-tuning jobs for a scaffolded experiment and monitor their progress until completion.

## Your Task

Submit SLURM jobs for all fine-tuning runs in an experiment directory (created by `scaffold-torchtune`) and monitor their status with efficient polling until all jobs complete.

## Workflow

1. **Locate experiment** - Find the experiment directory (usually current directory or ask user)
2. **Read experiment_summary.md** - Parse to identify all fine-tuning runs
3. **Identify runs to submit** - Find directories with `finetune.slurm` files
4. **Submit jobs** - Execute `sbatch finetune.slurm` for each run and capture job IDs
5. **Track submissions** - Record job IDs with their corresponding runs
6. **Monitor jobs** - Poll `squeue` every minute to check status
7. **Update status** - Keep experiment_summary.md status tables current
8. **Create log** - Document all submissions and status checks in `run-torchtune.log`
9. **Report completion** - Notify user when all jobs reach terminal states

## Finding the Experiment

**If user runs skill without arguments:**
- Check if current directory contains `experiment_summary.md`
- If not, ask user for the experiment directory path

**If user provides a path:**
- Use that path as the experiment directory

## Parsing experiment_summary.md

Extract the following information:

### Required Information

1. **Experiment name** - From the title (line 1)
2. **All runs** - From "All Runs" table
3. **Run directories** - Infer from scaffolding pattern (e.g., `rank8_lr1e-5/`)

## Identifying Runs to Submit

Scan the experiment directory for subdirectories containing `finetune.slurm`:

```bash
for dir in */; do
  if [ -f "$dir/finetune.slurm" ]; then
    echo "Found run: $dir"
  fi
done
```

**Skip runs that are:**
- Already submitted (check if job ID exists in status tracking)
- Already completed (check SLURM output files)
- Control runs (no fine-tuning needed)

## Submitting Jobs

For each run to submit:

1. **Navigate to run directory:**
   ```bash
   cd {experiment_dir}/{run_directory}
   ```

2. **Submit job and capture ID:**
   ```bash
   job_id=$(sbatch finetune.slurm | awk '{print $4}')
   ```
   The output format is: "Submitted batch job 12345678"
   Extract the job ID (last field)

3. **Record submission:**
   - Job ID
   - Run name
   - Submission timestamp
   - Initial status (PENDING)

4. **Update experiment_summary.md:**
   - Fill in Job ID column in Fine-tuning status table
   - Update Status to PENDING
   - Add submission timestamp to Started column

5. **Stagger submissions:**
   - Wait 5 seconds before submitting the next job
   - This prevents race conditions in HuggingFace datasets cache initialization
   - The delay is small enough to not significantly impact total experiment time
   - Skip the delay after the last job submission

## Monitoring Jobs

### Polling Strategy

**Interval:** Poll every 60 seconds (1 minute)

**Query command:**
```bash
squeue -u $USER -o "%.18i %.9P %.50j %.8T %.10M %.6D"
```

This returns:
- JOBID
- PARTITION
- NAME (job name)
- STATE (PENDING, RUNNING, COMPLETED, FAILED, etc.)
- TIME (elapsed time)
- NODES

**For completed jobs, use sacct:**
```bash
sacct -j {job_id} --format=JobID,State,Start,End,Elapsed
```

### Status Updates

On each poll:

1. **Query all user jobs at once** (not one by one)
2. **Match job IDs** to runs in the experiment
3. **Update status** for each run:
   - If state changed, update experiment_summary.md
   - Log the state change with timestamp
4. **Detect completion:**
   - COMPLETED: Job finished successfully
   - FAILED: Job failed
   - CANCELLED: Job was canceled
   - TIMEOUT: Job exceeded time limit

### Updating experiment_summary.md

Update the "Fine-tuning" status table:

```markdown
| Run Name | Status | Job ID | Started | Completed | Notes |
|----------|--------|--------|---------|-----------|-------|
| rank8_lr1e-5 | RUNNING | 12345678 | 2025-10-24 00:05:32 | - | - |
```

**Status values:**
- `pending` → PENDING (queued, waiting for resources)
- `running` → RUNNING (actively executing)
- `completed` → COMPLETED (finished successfully)
- `failed` → FAILED (error occurred)

**Timestamps:**
- Started: When job transitioned from PENDING to RUNNING
- Completed: When job reached terminal state

## Logging

Create a detailed log file at `{experiment_dir}/run-torchtune.log`:

### Log Format

```
[YYYY-MM-DD HH:MM:SS] ACTION: Description
Details: {specifics}
Result: {outcome}

```

### What to Log

- Experiment discovery
- Runs identified for submission
- Each job submission (run name, job ID, timestamp)
- Status checks (every poll - brief)
- State changes (detailed - when status changes)
- Completion detection
- Final summary

### Example Log Entries

```
[2025-10-24 00:05:00] DISCOVER_EXPERIMENT: Found experiment
Details: /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/experiment_summary.md
Result: Experiment has 8 fine-tuning runs

[2025-10-24 00:05:05] IDENTIFY_RUNS: Scanning for runs to submit
Details: Found 8 directories with finetune.slurm files
Result: 8 runs ready to submit (0 already running, 0 completed)

[2025-10-24 00:05:10] SUBMIT_JOB: rank8_lr1e-5
Command: cd rank8_lr1e-5 && sbatch finetune.slurm
Result: Job ID 12345678 submitted at 2025-10-24 00:05:10

[2025-10-24 00:05:15] SUBMIT_JOB: rank8_lr5e-5
Command: cd rank8_lr5e-5 && sbatch finetune.slurm
Result: Job ID 12345679 submitted at 2025-10-24 00:05:15
Note: 5 second stagger delay to prevent cache collision

[2025-10-24 00:05:20] ALL_SUBMITTED: Job submission complete
Summary: 8 jobs submitted successfully (0 failures)
Job IDs: 12345678-12345685

[2025-10-24 00:06:20] STATUS_CHECK: Polling SLURM
Command: squeue -u niznik
Result: 8 jobs found - 8 PENDING, 0 RUNNING, 0 COMPLETED

[2025-10-24 00:07:20] STATUS_CHECK: Polling SLURM
Result: 8 jobs found - 6 PENDING, 2 RUNNING, 0 COMPLETED

[2025-10-24 00:07:20] STATE_CHANGE: rank8_lr1e-5
Previous: PENDING
Current: RUNNING
Started: 2025-10-24 00:07:15
Action: Updated experiment_summary.md

[2025-10-24 00:07:20] STATE_CHANGE: rank8_lr5e-5
Previous: PENDING
Current: RUNNING
Started: 2025-10-24 00:07:18
Action: Updated experiment_summary.md

[2025-10-24 00:15:20] STATE_CHANGE: rank8_lr1e-5
Previous: RUNNING
Current: COMPLETED
Completed: 2025-10-24 00:15:17
Elapsed: 8m 2s
Action: Updated experiment_summary.md

[2025-10-24 00:25:00] ALL_COMPLETE: Monitoring finished
Summary: 8 jobs completed - 8 COMPLETED, 0 FAILED
Total time: 20 minutes
Next: See experiment_summary.md for workflow next steps
```

## Output Summary

After all jobs complete, provide a final summary:

```markdown
## Run Torchtune Complete

All 8 fine-tuning jobs have finished.

### Final Status

✓ COMPLETED: 8 runs
  - rank8_lr1e-5 (Job 12345678) - 8m 2s
  - rank8_lr5e-5 (Job 12345679) - 8m 15s
  - rank16_lr1e-5 (Job 12345680) - 9m 1s
  - rank16_lr5e-5 (Job 12345681) - 9m 12s
  - rank32_lr1e-5 (Job 12345682) - 10m 45s
  - rank32_lr5e-5 (Job 12345683) - 10m 58s
  - rank64_lr1e-5 (Job 12345684) - 12m 3s
  - rank64_lr5e-5 (Job 12345685) - 12m 18s

✗ FAILED: 0 runs

### Outputs

Model checkpoints saved to:
- `/scratch/gpfs/MSALGANIK/niznik/ck-outputs/ck-out-rank8_lr1e-5/epoch_0/`
- `/scratch/gpfs/MSALGANIK/niznik/ck-outputs/ck-out-rank8_lr5e-5/epoch_0/`
... (etc)

SLURM logs available in each run directory:
- `rank8_lr1e-5/slurm-12345678.out`
- `rank8_lr5e-5/slurm-12345679.out`
... (etc)

### Next Steps

**Refer to experiment_summary.md** for the complete workflow plan, including:
- Whether evaluation is part of this experiment
- How to execute evaluation jobs
- Analysis and result interpretation steps

**Validation** (recommended before proceeding):
1. **Check SLURM logs for any warnings or errors:**
   ```bash
   grep -i "error\|warning" rank*/slurm-*.out
   ```

2. **Verify model checkpoints were created:**
   ```bash
   ls -lh /scratch/gpfs/MSALGANIK/niznik/ck-outputs/ck-out-*/epoch_0/
   ```

**Typical next step** (see experiment_summary.md for specifics):
- If evaluation is configured: Execute evaluation jobs (via orchestrator or manually)

See `run-torchtune.log` for detailed execution history.
```

## Error Handling

**If job submission fails:**
- Log the error
- Continue with remaining jobs
- Report all failures at the end

**If monitoring fails:**
- Log the error
- Retry the query after next interval
- If repeated failures, ask user to check SLURM status manually

**If a job fails:**
- Record the failure in status
- Check SLURM output for error messages
- Continue monitoring other jobs
- Include failed jobs in final summary

## State Management

Track job states in memory during monitoring. Optionally create a `jobs.json` file for persistence:

```json
{
  "experiment": "cap_4L_lora_lr_sweep_2025-10-22",
  "monitoring_started": "2025-10-24 00:05:20",
  "jobs": {
    "rank8_lr1e-5": {
      "job_id": "12345678",
      "status": "COMPLETED",
      "submitted": "2025-10-24 00:05:10",
      "started": "2025-10-24 00:07:15",
      "completed": "2025-10-24 00:15:17",
      "elapsed": "8m 2s"
    }
  }
}
```

## Efficient Polling

**Good practices:**
- Query all jobs with one `squeue -u $USER` call
- Poll every 60 seconds (not more frequently)
- Use `sacct` for completed jobs (they disappear from `squeue`)
- Stop polling when all jobs reach terminal states

**Avoid:**
- Querying jobs one by one
- Polling faster than 60 seconds
- Continuing to poll after all jobs complete

## Validation Before Completion

Before reporting success, verify:
- ✓ All jobs were submitted successfully
- ✓ Job IDs were captured for all runs
- ✓ experiment_summary.md was updated with all job info
- ✓ All jobs reached terminal states (COMPLETED, FAILED, CANCELLED, TIMEOUT)
- ✓ Final status summary is accurate
- ✓ Log file was created

## Important Notes

- Only submit "Fine-tuned" runs, not "Control" runs (controls don't need SLURM jobs)
- Jobs run independently - one failing doesn't stop others
- Users can safely stop monitoring without canceling jobs (jobs keep running)
- Monitoring can be resumed later by re-running the skill
- Update experiment_summary.md atomically to avoid corruption
- Always use `squeue -u $USER` to filter to just user's jobs
- Poll interval: 60 seconds (1 minute)
- This skill is typically called by `run-experiment` orchestrator but can be run standalone
- After completion, model checkpoints should exist in `output_dir_base` directories

# Run Inspect

You help users submit all evaluation jobs for a scaffolded experiment and monitor their progress until completion.

## Your Task

Submit SLURM jobs for all inspect-ai evaluations in an experiment directory (created by `scaffold-inspect`) and monitor their status with efficient polling until all jobs complete.

**IMPORTANT:** This skill should only run AFTER fine-tuning completes, as evaluations require the fine-tuned model checkpoints to exist.

## Workflow

1. **Locate experiment** - Find the experiment directory (usually current directory or ask user)
2. **Read experiment_summary.md** - Parse to identify all evaluations
3. **Verify fine-tuning complete** - Check that model checkpoints exist
4. **Identify evaluations to submit** - Find `eval/` directories with `.slurm` files
5. **Submit jobs** - Execute `sbatch {task}_epoch{N}.slurm` for each evaluation
6. **Track submissions** - Record job IDs with their corresponding evaluations
7. **Monitor jobs** - Poll `squeue` every minute to check status
8. **Update status** - Keep experiment_summary.md evaluation status table current
9. **Create log** - Document all submissions and status checks in `run-inspect.log`
10. **Report completion** - Notify user when all jobs reach terminal states

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
3. **Evaluation plan** - From "Evaluations" status table:
   - Which run/task/epoch combinations to evaluate
   - Which evaluations are already completed
4. **Output directories** - Where fine-tuned models were saved

## Verifying Fine-Tuning Complete

Before submitting evaluation jobs, verify that fine-tuning has completed:

### Check 1: SLURM Job Status

Query if fine-tuning jobs are still running:
```bash
squeue -u $USER | grep finetune
```

If jobs are running:
- Report to user that fine-tuning is still in progress
- Ask: "Fine-tuning jobs are still running. Wait for completion or proceed anyway?"
- If wait: Poll every minute until complete, then proceed
- If proceed anyway: Continue (jobs will fail if models don't exist yet)

### Check 2: Model Checkpoints Exist

Verify that expected model checkpoints exist:
```bash
# For each run that needs evaluation
ls {output_dir_base}/ck-out-{run_name}/epoch_{N}/
```

If checkpoints missing:
- Log warning for that run
- Skip that run's evaluations
- Continue with runs that have checkpoints
- Report skipped evaluations in summary

## Identifying Evaluations to Submit

Scan for evaluation SLURM scripts:

```bash
for run_dir in */; do
  if [ -d "$run_dir/eval" ]; then
    for eval_script in "$run_dir/eval"/*.slurm; do
      if [ -f "$eval_script" ]; then
        echo "Found evaluation: $eval_script"
      fi
    done
  fi
done
```

**Skip evaluations that are:**
- Already submitted (check if job ID exists in status tracking)
- Already completed (check for inspect-ai log files in `eval/logs/`)
- For runs with missing model checkpoints

## Submitting Jobs

For each evaluation to submit:

1. **Navigate to eval directory:**
   ```bash
   cd {experiment_dir}/{run_directory}/eval
   ```

2. **Submit job and capture ID:**
   ```bash
   job_id=$(sbatch {task}_epoch{N}.slurm | awk '{print $4}')
   ```
   The output format is: "Submitted batch job 12345678"
   Extract the job ID (last field)

3. **Record submission:**
   - Job ID
   - Run name
   - Task name
   - Epoch number
   - Submission timestamp
   - Initial status (PENDING)

4. **Update experiment_summary.md:**
   - Fill in Job ID column in Evaluations status table
   - Update Status to PENDING
   - Add submission timestamp

5. **No stagger delay needed:**
   - Unlike fine-tuning, evaluations don't have cache race conditions
   - Can submit all evaluations rapidly
   - Still good to have brief delay (1 second) for rate limiting

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
- NAME (job name - includes "eval" for identification)
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
2. **Match job IDs** to evaluations in the experiment
3. **Update status** for each evaluation:
   - If state changed, update experiment_summary.md
   - Log the state change with timestamp
4. **Detect completion:**
   - COMPLETED: Evaluation finished successfully
   - FAILED: Evaluation failed
   - CANCELLED: Job was canceled
   - TIMEOUT: Job exceeded time limit

### Updating experiment_summary.md

Update the "Evaluations" status table:

```markdown
| Run Name | Task | Epoch | Status | Job ID | Completed | Notes |
|----------|------|-------|--------|--------|-----------|-------|
| rank8_lr1e-5 | capitalization | 0 | RUNNING | 12345690 | - | - |
```

**Status values:**
- `pending` → PENDING (queued, waiting for resources)
- `running` → RUNNING (actively executing)
- `completed` → COMPLETED (finished successfully)
- `failed` → FAILED (error occurred)

**Timestamps:**
- Completed: When job reached terminal state

## Logging

Create a detailed log file at `{experiment_dir}/run-inspect.log`:

### Log Format

```
[YYYY-MM-DD HH:MM:SS] ACTION: Description
Details: {specifics}
Result: {outcome}

```

### What to Log

- Experiment discovery
- Fine-tuning completion verification
- Model checkpoint verification
- Evaluations identified for submission
- Each job submission (run, task, epoch, job ID, timestamp)
- Status checks (every poll - brief)
- State changes (detailed - when status changes)
- Completion detection
- Final summary with paths to results

### Example Log Entries

```
[2025-10-24 00:30:00] DISCOVER_EXPERIMENT: Found experiment
Details: /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/experiment_summary.md
Result: Experiment has 8 runs with evaluations

[2025-10-24 00:30:05] VERIFY_FINETUNING: Checking if fine-tuning complete
Command: squeue -u niznik | grep finetune
Result: No fine-tuning jobs running - training complete

[2025-10-24 00:30:10] VERIFY_CHECKPOINTS: Checking model checkpoints
Details: Verifying 8 model checkpoint directories
Result: All 8 checkpoints exist and accessible

[2025-10-24 00:30:15] IDENTIFY_EVALS: Scanning for evaluation scripts
Details: Found 8 evaluation scripts in */eval/ directories
Result: 8 evaluations ready to submit (0 already completed)

[2025-10-24 00:30:20] SUBMIT_EVAL: rank8_lr1e-5/capitalization/epoch0
Command: cd rank8_lr1e-5/eval && sbatch capitalization_epoch0.slurm
Result: Job ID 12345690 submitted at 2025-10-24 00:30:20

[2025-10-24 00:30:21] SUBMIT_EVAL: rank8_lr5e-5/capitalization/epoch0
Command: cd rank8_lr5e-5/eval && sbatch capitalization_epoch0.slurm
Result: Job ID 12345691 submitted at 2025-10-24 00:30:21

[2025-10-24 00:30:28] ALL_SUBMITTED: Evaluation submission complete
Summary: 8 evaluations submitted successfully (0 failures)
Job IDs: 12345690-12345697

[2025-10-24 00:31:28] STATUS_CHECK: Polling SLURM
Command: squeue -u niznik
Result: 8 jobs found - 8 PENDING, 0 RUNNING, 0 COMPLETED

[2025-10-24 00:32:28] STATUS_CHECK: Polling SLURM
Result: 8 jobs found - 4 PENDING, 4 RUNNING, 0 COMPLETED

[2025-10-24 00:32:28] STATE_CHANGE: rank8_lr1e-5/capitalization/epoch0
Previous: PENDING
Current: RUNNING
Job ID: 12345690
Action: Updated experiment_summary.md

[2025-10-24 00:35:15] STATE_CHANGE: rank8_lr1e-5/capitalization/epoch0
Previous: RUNNING
Current: COMPLETED
Completed: 2025-10-24 00:35:12
Elapsed: 4m 52s
Action: Updated experiment_summary.md
Result log: rank8_lr1e-5/eval/logs/

[2025-10-24 00:45:00] ALL_COMPLETE: Monitoring finished
Summary: 8 evaluations completed - 8 COMPLETED, 0 FAILED
Total time: 15 minutes
Next: See experiment_summary.md for result analysis steps
```

## Output Summary

After all evaluations complete, provide a final summary:

```markdown
## Run Inspect Complete

All 8 evaluations have finished.

### Final Status

✓ COMPLETED: 8 evaluations
  - rank8_lr1e-5/capitalization/epoch0 (Job 12345690) - 4m 52s
  - rank8_lr5e-5/capitalization/epoch0 (Job 12345691) - 5m 1s
  - rank16_lr1e-5/capitalization/epoch0 (Job 12345692) - 5m 15s
  - rank16_lr5e-5/capitalization/epoch0 (Job 12345693) - 5m 22s
  - rank32_lr1e-5/capitalization/epoch0 (Job 12345694) - 5m 45s
  - rank32_lr5e-5/capitalization/epoch0 (Job 12345695) - 5m 51s
  - rank64_lr1e-5/capitalization/epoch0 (Job 12345696) - 6m 8s
  - rank64_lr5e-5/capitalization/epoch0 (Job 12345697) - 6m 15s

✗ FAILED: 0 evaluations

### Results

Evaluation logs saved to:
- `rank8_lr1e-5/eval/logs/*.eval`
- `rank8_lr5e-5/eval/logs/*.eval`
... (etc)

SLURM logs available:
- `rank8_lr1e-5/eval/slurm-12345690.out`
- `rank8_lr5e-5/eval/slurm-12345691.out`
... (etc)

### Viewing Results

**Interactive viewer (recommended):**
```bash
cd /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22
inspect view --port=$(get_free_port)
```
Then open the provided URL in your browser to explore results.

**Command-line summary:**
```bash
# View accuracy for all evaluations
for dir in rank*/eval/logs; do
  echo "=== $(dirname $(dirname $dir)) ==="
  inspect log ls $dir/*.eval | grep -i accuracy
done
```

### Next Steps

**Refer to experiment_summary.md** for the complete workflow plan, including:
- How to view and analyze results
- Next analysis steps
- Experiment conclusions and documentation

**Viewing results** (inspect-ai specific commands):
1. **Interactive viewer:**
   ```bash
   inspect view --port=$(get_free_port)
   ```

2. **Export for analysis:**
   ```bash
   inspect log export rank8_lr1e-5/eval/logs/*.eval --format csv > results.csv
   ```

**Typical next step** (see experiment_summary.md for specifics):
- Analyze and compare results across runs

See `run-inspect.log` for detailed execution history.
```

## Error Handling

**If fine-tuning jobs still running:**
- Warn user
- Offer to wait or proceed anyway
- If proceed, jobs may fail when models don't exist

**If model checkpoints missing:**
- Log warning for each missing checkpoint
- Skip evaluations for those runs
- Continue with runs that have checkpoints
- Report skipped evaluations in summary

**If job submission fails:**
- Log the error
- Continue with remaining evaluations
- Report all failures at the end

**If monitoring fails:**
- Log the error
- Retry the query after next interval
- If repeated failures, ask user to check SLURM status manually

**If an evaluation fails:**
- Record the failure in status
- Check SLURM output and inspect-ai logs for error messages
- Continue monitoring other jobs
- Include failed evaluations in final summary

## State Management

Track evaluation job states in memory during monitoring. Optionally create an `eval_jobs.json` file for persistence:

```json
{
  "experiment": "cap_4L_lora_lr_sweep_2025-10-22",
  "monitoring_started": "2025-10-24 00:30:28",
  "evaluations": {
    "rank8_lr1e-5_capitalization_epoch0": {
      "run": "rank8_lr1e-5",
      "task": "capitalization",
      "epoch": 0,
      "job_id": "12345690",
      "status": "COMPLETED",
      "submitted": "2025-10-24 00:30:20",
      "completed": "2025-10-24 00:35:12",
      "elapsed": "4m 52s",
      "log_path": "rank8_lr1e-5/eval/logs/"
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
- Evaluation jobs are typically faster than fine-tuning (5-10 minutes vs hours)

**Avoid:**
- Querying jobs one by one
- Polling faster than 60 seconds
- Continuing to poll after all jobs complete

## Validation Before Completion

Before reporting success, verify:
- ✓ Fine-tuning completion was checked
- ✓ Model checkpoints were verified
- ✓ All evaluations were submitted successfully
- ✓ Job IDs were captured for all evaluations
- ✓ experiment_summary.md was updated with all job info
- ✓ All jobs reached terminal states
- ✓ Final status summary is accurate
- ✓ Paths to results are included in summary
- ✓ Log file was created

## Important Notes

- **Must run after fine-tuning completes** - evaluations require model checkpoints
- Evaluation jobs are independent - one failing doesn't stop others
- Users can safely stop monitoring without canceling jobs (jobs keep running)
- Monitoring can be resumed later by re-running the skill
- Update experiment_summary.md atomically to avoid corruption
- Always use `squeue -u $USER` to filter to just user's jobs
- Poll interval: 60 seconds (1 minute)
- This skill is typically called by `run-experiment` orchestrator but can be run standalone
- inspect-ai evaluation logs (.eval files) are the primary output
- Use `inspect view` for interactive result browsing
- Use `inspect log export` for programmatic result extraction

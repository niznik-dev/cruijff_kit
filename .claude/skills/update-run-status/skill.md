# Update Run Status

You are helping the user track the progress of their fine-tuning and evaluation runs by querying SLURM and updating the status tracker.

## Your Task

Read the run plan, query SLURM for current job states, and update the status tracker (`runs_status.yaml`) to reflect the current progress of all runs.

## Workflow Overview

Follow these steps to update run status:

1. **Locate the run directory**
   - Ask user for the run group directory path (or detect from context)
   - Verify `runs_plan.md` exists
   - Check if `runs_status.yaml` exists

2. **Initialize status file (if needed)**
   - If `runs_status.yaml` doesn't exist, create it from `runs_plan.md`
   - Extract all run names from the plan
   - Initialize all runs with `pending` status

3. **Query SLURM for job states**
   - Use `squeue` to check running/pending jobs
   - Use `sacct` to check completed/failed jobs
   - Match job names to run names

4. **Update status tracker**
   - Update job statuses based on SLURM data
   - Update job IDs if found
   - Update output paths if jobs completed
   - Set timestamps

5. **Report to user**
   - Summarize current state (how many pending/running/completed/failed)
   - Highlight any issues or failures
   - Suggest next actions if needed

## Status File Format

The `runs_status.yaml` file tracks each run through the pipeline:

```yaml
# Run Status Tracker
# Auto-generated and updated by update-run-status skill
# Last updated: 2025-10-19 18:30:00

run_group: cap_8L_llama32_lora_comparison_2025-10-18
runs:
  1B_rank4:
    finetune_status: completed
    finetune_job_id: "1234567"
    finetune_output: /scratch/gpfs/MSALGANIK/mjs3/ck-out-bright-horizon/epoch_2
    eval_status: running
    eval_job_id: "1234568"
    eval_output: null
    last_updated: "2025-10-19 18:30:00"
  1B_rank64:
    finetune_status: running
    finetune_job_id: "1234569"
    finetune_output: null
    eval_status: pending
    eval_job_id: null
    eval_output: null
    last_updated: "2025-10-19 18:25:00"
```

### Status Values

- **pending**: Not yet submitted to SLURM
- **submitted**: Job submitted but not yet detected in queue (transitional state)
- **running**: Job is currently running (in `squeue`)
- **completed**: Job finished successfully
- **failed**: Job failed or was cancelled

## Querying SLURM

### Check Running Jobs

```bash
# Get all jobs for user with formatted output
squeue -u $USER --format="%.10i %.30j %.8T %.10M %.10l %.8N"

# Filter for specific job name pattern
squeue -u $USER --name=finetune_1B_rank4
```

### Check Completed Jobs

```bash
# Get recent job history (today)
sacct -u $USER -S today --format=JobID,JobName,State,ExitCode,Elapsed,End -X

# Check specific job by ID
sacct -j 1234567 --format=JobID,JobName,State,ExitCode,Elapsed,End
```

### Matching Jobs to Runs

Jobs are matched by job name. The job name should contain or match the run name. Common patterns:
- Job name = run name (e.g., `1B_rank4`)
- Job name includes run name (e.g., `finetune_1B_rank4`)
- Job name from wandb run name (check against run directory)

**If job matching is ambiguous:**
- Ask user to clarify job naming convention
- Look for job output files in run directories
- Check SLURM output logs for correlation

## Determining Output Paths

For completed jobs, determine the output path:

### For Fine-tuning Jobs

1. **Check the run's `setup_finetune.yaml`** for `output_dir_base` and `my_wandb_run_name`
2. **Construct path**: `{output_dir_base}/ck-out-{run_name}/epoch_{N}`
3. **Verify path exists** using `ls`
4. **Find latest epoch** if multiple epochs saved

Example:
```bash
# Find the output directory
ls /scratch/gpfs/MSALGANIK/mjs3/ck-out-*/

# Find epochs
ls /scratch/gpfs/MSALGANIK/mjs3/ck-out-bright-horizon/
```

### For Evaluation Jobs

1. **Check for inspect logs** in the run directory
2. **Look for inspect database** (usually in `.inspect` or task directory)
3. **Path is typically**: Task directory or specified eval output location

## Status Update Logic

For each run, determine status based on SLURM queries:

### Fine-tuning Status

```
If job found in squeue:
  → status = "running"
  → capture job_id
Else if job found in sacct with State=COMPLETED:
  → status = "completed"
  → capture job_id
  → determine output path
Else if job found in sacct with State=FAILED/CANCELLED:
  → status = "failed"
  → capture job_id
Else:
  → status remains "pending" (or "submitted" if recently added)
```

### Evaluation Status

Same logic as fine-tuning, but for eval jobs.

## Error Handling

**If runs_plan.md not found:**
- Verify the directory path with user
- Ensure they're in the correct run group directory
- Cannot proceed without the plan file

**If SLURM commands fail:**
- Check if user is on the HPC cluster
- Verify SLURM is available (`which squeue`)
- Ask user to run commands manually and provide output

**If job names don't match runs:**
- Ask user about their job naming convention
- Look for patterns in actual SLURM job names
- May need to manually map jobs to runs

**If output paths not found:**
- Check if jobs actually completed successfully
- Look for errors in SLURM logs
- Ask user if output location is non-standard

## Output Summary

After updating status, provide a clear summary:

```markdown
## Run Status Update - 2025-10-19 18:30:00

### Overall Progress
- **Pending**: 2 runs
- **Running**: 1 run
- **Completed**: 3 runs
- **Failed**: 0 runs

### Details by Run
| Run Name | Finetune | Eval | Notes |
|----------|----------|------|-------|
| 1B_rank4 | ✓ completed | → running | Eval job 1234568 |
| 1B_rank64 | → running | pending | Finetune job 1234569 |
| 3B_rank4 | ✓ completed | ✓ completed | All done! |
| 3B_rank64 | pending | pending | Not started |

### Next Actions
- Monitor running jobs: `squeue -u $USER`
- 1 eval job in progress
- 2 runs not yet started
```

## Workflow Integration

This skill is part of the complete run pipeline:

1. **`plan-runs`** → Creates `runs_plan.md` and initializes `runs_status.yaml`
2. **`setup-experiment-dirs`** → Creates directory structure for each run
3. **`create-torchtune-config`** → Generates `setup_finetune.yaml` configs for each run
4. **`generate-slurm-script`** → Generates `finetune.slurm` scripts from configs
5. **User submits jobs** → `sbatch` or `submit_all.sh`
6. **`update-run-status`** → Track job progress (run periodically)
7. **Evaluation phase** → After fine-tuning completes
8. **`update-run-status`** → Track evaluation progress

## Usage Patterns

This skill can be used in several ways:

1. **Periodic checks**: User invokes periodically to check progress
2. **After job submission**: Immediately after submitting jobs to capture job IDs
3. **Before evaluation**: Check that fine-tuning is complete before starting eval
4. **Debugging**: Identify failed jobs and investigate

## Important Notes

- **Job IDs are strings** in YAML (wrap in quotes) to preserve leading zeros
- **Timestamps** should be in ISO format or readable format (YYYY-MM-DD HH:MM:SS)
- **Output paths** should be absolute paths
- **Status updates are idempotent**: Running this skill multiple times is safe
- **SLURM caching**: `sacct` may have slight delays in showing very recent job completions
- **Null values**: Use `null` (not empty string) for unset fields in YAML

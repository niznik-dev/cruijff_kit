# Run Selection - Planning

Decide which fine-tuning runs need to be submitted.

## Selection Criteria

**Submit if:**
- Run has `finetune.slurm` file
- Run type is "Fine-tuned" (not "Control")
- Run is NOT already running (check squeue)
- Run is NOT already completed (check SLURM logs or status)

**Skip if:**
- Run is "Control" type (no fine-tuning needed)
- Job ID already exists in run state tracking and job is running
- SLURM output file exists and shows COMPLETED status
- Model checkpoint already exists (resumability)

## Check Current Status

### Query Running Jobs

```bash
squeue -u $USER -o "%.18i %.50j %.8T"
```

Match job names to run directories to identify already-running jobs.

### Check for Completion

Look for SLURM output files:
```bash
ls {run_dir}/slurm-*.out 2>/dev/null
```

If exists, check final status in file.

### Check for Existing Checkpoints

```bash
ls {output_dir_base}/ck-out-{run_name}/epoch_{N}/ 2>/dev/null
```

If checkpoint exists, job completed successfully (skip resubmission).

## Output

**Runs to submit:** List of run directories that need job submission

**Runs to skip:** List of run directories with reason for skipping
- Already running
- Already completed
- Control run (no job needed)

## Next Stage

Pass "runs to submit" list to job_submission.md for execution.

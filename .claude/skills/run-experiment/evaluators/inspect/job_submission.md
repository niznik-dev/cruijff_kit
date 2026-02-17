# Job Submission - Execution

Submit SLURM jobs for selected evaluations.

## Submission Process

For each evaluation in "evaluations to submit" list:

### 1. Navigate to Eval Directory

```bash
cd {experiment_dir}/{run_directory}/eval
```

### 2. Submit Job and Capture ID

```bash
job_id=$(sbatch {task}_epoch{N}.slurm | awk '{print $4}')
```

**Output format:** "Submitted batch job 12345678"
**Extract:** Job ID from last field

### 3. Record Submission

Track:
- Job ID
- Run name
- Task name
- Epoch number
- Submission timestamp
- Initial status (PENDING)

### 4. Submission Timing

Brief 1-second delay between submissions:
```bash
sleep 1
```

We don't need a longer delay because HF datasets cache is pre-built before submission (see cache_prebuilding.md)

## Error Handling

**If submission fails:**
- Log the error with details
- Continue with remaining evaluations
- Record failure in run state
- Report all failures in final summary

## Logging

Log each submission:

```
[YYYY-MM-DD HH:MM:SS] SUBMIT_EVAL: {run_name}/{task}/epoch{N}
Command: cd {run_dir}/eval && sbatch {task}_epoch{N}.slurm
Result: Job ID {job_id} submitted at {timestamp}
```

Example:
```
[2025-11-11 14:45:20] SUBMIT_EVAL: r8_lr1e-5/capitalization/epoch0
Command: cd r8_lr1e-5/eval && sbatch capitalization_epoch0.slurm
Result: Job ID 12345690 submitted at 2025-11-11 14:45:20
```

## Batch Summary

After all submissions:

```
[YYYY-MM-DD HH:MM:SS] ALL_SUBMITTED: Evaluation submission complete
Summary: 8 evaluations submitted successfully (0 failures)
Job IDs: 12345690-12345697
```

## Output

- All evaluations submitted with captured job IDs
- Detailed log of submissions

## Next Stage

Pass job IDs to monitoring.md for status tracking.

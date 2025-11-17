# Job Submission - Execution

Submit SLURM jobs for selected fine-tuning runs.

## Submission Process

For each run in "runs to submit" list:

### 1. Navigate to Run Directory

```bash
cd {experiment_dir}/{run_directory}
```

### 2. Submit Job and Capture ID

```bash
job_id=$(sbatch finetune.slurm | awk '{print $4}')
```

**Output format:** "Submitted batch job 12345678"
**Extract:** Job ID from last field

### 3. Record Submission

Track:
- Job ID
- Run name
- Submission timestamp
- Initial status (PENDING)

### 4. Update experiment_summary.md

Update Fine-tuning status table:

```markdown
| Run Name | Status | Job ID | Started | Completed | Notes |
|----------|--------|--------|---------|-----------|-------|
| r8_lr1e-5 | PENDING | 12345678 | 2025-11-11 14:30:00 | - | - |
```

### 5. Stagger Submissions

**CRITICAL:** Wait 5 seconds before submitting next job.

**Why?** Prevents race conditions in HuggingFace datasets cache initialization when multiple jobs start simultaneously.

**Implementation:**
```bash
sleep 5
```

**Exception:** Skip delay after final job submission.

## Error Handling

**If submission fails:**
- Log the error with details
- Continue with remaining jobs
- Record failure in status table
- Report all failures in final summary

## Logging

Log each submission:

```
[YYYY-MM-DD HH:MM:SS] SUBMIT_JOB: {run_name}
Command: cd {run_dir} && sbatch finetune.slurm
Result: Job ID {job_id} submitted at {timestamp}
Note: 5 second stagger delay to prevent cache collision
```

## Output

- All jobs submitted with captured job IDs
- experiment_summary.md updated with job metadata
- Detailed log of submissions

## Next Stage

Pass job IDs to monitoring.md for status tracking.

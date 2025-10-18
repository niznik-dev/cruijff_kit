# Run Experiments

You are helping the user submit and monitor SLURM jobs for their fine-tuning experiments.

## Your Task

Submit all SLURM scripts for the experiment set and help monitor their progress.

## Steps

### 1. Review Experiments

Before submitting, show the user:
- List of all experiments to be run
- Expected resource usage (time, GPUs, memory)
- Total number of jobs

Get confirmation before proceeding.

### 2. Submit Jobs

For each experiment directory:
1. Navigate to the experiment directory
2. Submit the SLURM script using `sbatch`:
   ```bash
   cd /path/to/experiment
   sbatch finetune.slurm
   ```
3. Record the job ID returned by sbatch

Keep track of all submitted job IDs.

### 3. Monitor Jobs

After submission, provide the user with commands to monitor progress:

**Check job status:**
```bash
squeue -u mjs3
```

**Check specific job:**
```bash
squeue -j <JOB_ID>
```

**View job output (while running):**
```bash
tail -f slurm-<JOB_ID>.out
```

**Check completed job info:**
```bash
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS
```

### 4. Job Management

If needed, provide commands for:

**Cancel a job:**
```bash
scancel <JOB_ID>
```

**Cancel all user jobs:**
```bash
scancel -u mjs3
```

**Cancel specific job array:**
```bash
scancel <JOB_ID>_<ARRAY_INDEX>
```

## Monitoring Script

You can create a simple monitoring script to check all jobs:

```bash
#!/bin/bash
# Monitor all jobs from this experiment
for job_id in <JOB_ID_1> <JOB_ID_2> <JOB_ID_3>; do
    echo "Job $job_id:"
    squeue -j $job_id 2>/dev/null || echo "  Completed or not found"
done
```

## Summary Report

After submitting all jobs, provide a summary:
- Total jobs submitted: X
- Job IDs: [list]
- Estimated completion time: [based on time limits]
- How to check status: `squeue -u mjs3`

## Next Steps

Once jobs complete, suggest using the `create-evaluation` skill to set up evaluation scripts for the fine-tuned models.

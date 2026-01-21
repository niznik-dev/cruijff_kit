# Torchtune Execution Workflow

This document describes the detailed step-by-step process for executing torchtune fine-tuning jobs.

## Prerequisites

- experiment_summary.yaml exists
- Fine-tuning scaffolding complete (finetune.slurm files exist)
- SLURM cluster access

## Step-by-Step Process

### 1. Parse Experiment Configuration

**Read experiment_summary.yaml:**
- Extract experiment name from `experiment.name`
- Parse `runs:` section for run names and types
- Identify run directories

**Scan for job scripts:**
```bash
for dir in */; do
  if [ -f "$dir/finetune.slurm" ]; then
    echo "Found run: $dir"
  fi
done
```

**Build run list:**
For each directory with finetune.slurm, collect:
- Run directory name (e.g., `r8_lr1e-5`)
- Path to SLURM script

**Technical details:** See [optimizers/torchtune/parsing.md](../optimizers/torchtune/parsing.md)

### 2. Select Runs to Submit

**Check for already running jobs:**
```bash
squeue -u $USER -o "%.18i %.50j %.8T"
```

**Check for completed runs:**
- Look for SLURM output files (`slurm-*.out`)
- Check for existing model checkpoints

**Decision criteria:**
- ✓ Submit: Run has finetune.slurm, is "Fine-tuned" type, not running, not completed
- ✗ Skip: Already running, already completed, or "Control" type

**Technical details:** See [optimizers/torchtune/run_selection.md](../optimizers/torchtune/run_selection.md)

### 3. Submit SLURM Jobs

For each run in "submit" list:

**Navigate and submit:**
```bash
cd {experiment_dir}/{run_directory}
job_id=$(sbatch finetune.slurm | awk '{print $4}')
```

**Record submission:**
- Capture job ID
- Record timestamp

**Stagger submissions:**
```bash
sleep 5  # CRITICAL: Prevents HuggingFace cache race conditions
```

Wait 5 seconds between each submission (except after last job).

**Why stagger?** Multiple jobs starting simultaneously can race to initialize HuggingFace datasets cache, causing failures. 5-second delay prevents this while having minimal impact on total experiment time.

**Technical details:** See [optimizers/torchtune/job_submission.md](../optimizers/torchtune/job_submission.md)

### 4. Monitor Job Progress

**Poll SLURM every 60 seconds:**
```bash
squeue -u $USER -o "%.18i %.9P %.50j %.8T %.10M %.6D"
```

**For completed jobs (disappeared from squeue):**
```bash
sacct -j {job_id} --format=JobID,State,Start,End,Elapsed
```

**Track state changes:**
- PENDING → RUNNING: Record start timestamp
- RUNNING → COMPLETED: Record completion timestamp and elapsed time
- RUNNING → FAILED: Record failure and note to check logs

**Continue until all terminal:**
Stop monitoring when all jobs reach: COMPLETED, FAILED, CANCELLED, or TIMEOUT

**Technical details:** See [optimizers/torchtune/monitoring.md](../optimizers/torchtune/monitoring.md)

### 5. Validate Completion

**Check all jobs submitted:**
Verify job IDs captured for all runs.

**Check all jobs terminal:**
No jobs still PENDING or RUNNING.

**Verify model checkpoints:**
```bash
ls {output_dir_base}/ck-out-{run_name}/epoch_{N}/
```

For each COMPLETED job, verify checkpoint directory exists with:
- Weight files (adapter_model.bin or similar)
- Config files (adapter_config.json)

**Check log file created:**
Verify detailed execution log exists.

**Technical details:** See [optimizers/torchtune/validation.md](../optimizers/torchtune/validation.md)

## Logging

Create detailed log at `{experiment_dir}/run-torchtune.log` (or similar name based on module invocation).

**Log entries:**
```
[YYYY-MM-DD HH:MM:SS] DISCOVER_EXPERIMENT
[YYYY-MM-DD HH:MM:SS] IDENTIFY_RUNS
[YYYY-MM-DD HH:MM:SS] SUBMIT_JOB: {run_name}
[YYYY-MM-DD HH:MM:SS] STATUS_CHECK
[YYYY-MM-DD HH:MM:SS] STATE_CHANGE: {run_name}
[YYYY-MM-DD HH:MM:SS] ALL_COMPLETE
```

## Error Handling

**If job submission fails:**
- Log error
- Continue with remaining jobs
- Report failures in summary

**If a job fails:**
- Record FAILED state
- Continue monitoring other jobs
- Include in final summary with note to check SLURM log

**If monitoring query fails:**
- Log error
- Retry at next interval
- Alert user if repeated failures

## Success Criteria

- ✓ All jobs submitted successfully
- ✓ All jobs reached terminal states
- ✓ Model checkpoints exist for COMPLETED jobs
- ✓ Log file complete

## Important Notes

**Job staggering is critical:**
The 5-second delay between submissions prevents HuggingFace cache race conditions. Don't skip this!

**Polling efficiency:**
- Query all user jobs at once (not one by one)
- Poll every 60 seconds (not faster)
- Use sacct for completed jobs (they disappear from squeue)

**Resumability:**
Safe to re-run. Selection stage checks for already-completed runs and skips resubmission.

**Control runs:**
Only submit "Fine-tuned" runs, not "Control" runs (controls don't need training).

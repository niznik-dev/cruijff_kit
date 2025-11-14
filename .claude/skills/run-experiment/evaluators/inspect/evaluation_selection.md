# Evaluation Selection - Planning

Decide which evaluations need to be submitted.

## Selection Criteria

**Submit if:**
- Evaluation has `.slurm` file in `eval/` directory
- Model checkpoint exists (from dependency_checking.md)
- Evaluation is NOT already running (check squeue)
- Evaluation is NOT already completed (check for .eval logs)

**Skip if:**
- Model checkpoint missing (from dependency_checking.md skip list)
- Job ID already exists in experiment_summary.md and job is running
- Evaluation log file exists in `eval/logs/*.eval` (already completed)

## Check Current Status

### Query Running Evaluation Jobs

```bash
squeue -u $USER -o "%.18i %.50j %.8T" | grep eval
```

Match job names to evaluation scripts to identify already-running evaluations.

### Check for Completed Evaluations

Look for inspect-ai log files:
```bash
ls {run_dir}/eval/logs/*.eval 2>/dev/null
```

**If .eval file exists:**
- Evaluation already completed
- Skip resubmission (resumability)
- Can optionally allow re-run if user requests

### Parse File Names

Extract evaluation details from SLURM script filenames:
- `capitalization_epoch0.slurm` → task: capitalization, epoch: 0
- `generalization_epoch1.slurm` → task: generalization, epoch: 1

## Output

**Evaluations to submit:** List of evaluations that need job submission
- Run directory
- Task name
- Epoch number
- SLURM script path

**Evaluations to skip:** List of evaluations with reason for skipping
- Already running (job ID: ...)
- Already completed (.eval log exists)
- Model checkpoint missing (fine-tuning failed or incomplete)

## Logging

```
[YYYY-MM-DD HH:MM:SS] IDENTIFY_EVALS: Scanning for evaluation scripts
Details: Found 8 evaluation scripts in */eval/ directories
Result: 6 evaluations ready to submit (2 already completed)

Skipped evaluations:
- r8_lr1e-5/capitalization/epoch0: Already completed
- r32_lr5e-5/capitalization/epoch0: Model checkpoint missing
```

## Next Stage

Pass "evaluations to submit" list to job_submission.md for execution.

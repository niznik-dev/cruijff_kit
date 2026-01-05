# Inspect Execution Workflow

This document describes the detailed step-by-step process for executing inspect-ai evaluation jobs.

**CRITICAL:** This workflow MUST run AFTER fine-tuning completes. Evaluations require fine-tuned model checkpoints.

## Prerequisites

- experiment_summary.yaml exists
- Evaluation scaffolding complete (eval/*.slurm files exist)
- Fine-tuning complete (model checkpoints exist)
- SLURM cluster access

## Step-by-Step Process

### 1. Parse Experiment Configuration

**Read experiment_summary.yaml:**
- Extract experiment name from `experiment.name`
- Parse `runs:` section for run names and types
- Parse `evaluation.matrix:` section for which runs/epochs to evaluate
- Identify output directories from `output.base_directory`

**Scan for evaluation scripts:**
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

**Build evaluation list:**
For each eval/*.slurm file, collect:
- Run directory name
- Task name (from filename, e.g., capitalization)
- Epoch number (from filename, e.g., 0)
- Path to SLURM script
- Expected checkpoint path
- Current status (if exists)

**Technical details:** See [evaluators/inspect/parsing.md](../evaluators/inspect/parsing.md)

### 2. Verify Fine-Tuning Complete (CRITICAL)

**⚠️ DEPENDENCY CHECK - MANDATORY**

This stage ensures evaluations won't fail due to missing model checkpoints.

#### Check 2a: Fine-Tuning Jobs Status

```bash
squeue -u $USER | grep finetune
```

**If fine-tuning jobs still running:**
- Warn user: "Fine-tuning jobs are still in progress"
- Ask: "Wait for completion or proceed anyway?"
- If wait: Poll every minute until all complete, then proceed
- If proceed: Continue (evaluations may fail if checkpoints don't exist yet)

#### Check 2b: Model Checkpoints Exist

For each evaluation, verify required checkpoint exists:

```bash
ls {output_dir_base}/ck-out-{run_name}/epoch_{N}/
```

**For each evaluation:**
- ✓ Checkpoint exists: Ready for evaluation
- ✗ Checkpoint missing: Add to skip list, log warning

**If checkpoint missing:**
```
[YYYY-MM-DD HH:MM:SS] CHECKPOINT_MISSING: r32_lr5e-5
Details: Expected path: /scratch/.../ck-out-r32_lr5e-5/epoch_0/
Result: Checkpoint not found - skipping evaluations for this run
Recommendation: Check if fine-tuning job failed
```

**Technical details:** See [evaluators/inspect/dependency_checking.md](../evaluators/inspect/dependency_checking.md)

### 3. Select Evaluations to Submit

**Check for already running evaluations:**
```bash
squeue -u $USER -o "%.18i %.50j %.8T" | grep eval
```

**Check for completed evaluations:**
```bash
ls {run_dir}/eval/logs/*.eval 2>/dev/null
```

**Decision criteria:**
- ✓ Submit: Has .slurm file, checkpoint exists, not running, not completed
- ✗ Skip: Already running, already completed, or checkpoint missing

**Technical details:** See [evaluators/inspect/evaluation_selection.md](../evaluators/inspect/evaluation_selection.md)

### 4. Submit SLURM Jobs

For each evaluation in "submit" list:

**Navigate and submit:**
```bash
cd {experiment_dir}/{run_directory}/eval
job_id=$(sbatch {task}_epoch{N}.slurm | awk '{print $4}')
```

**Record submission:**
- Capture job ID
- Record timestamp

**No stagger delay needed:**
Unlike fine-tuning, evaluations don't have cache race conditions. Can submit rapidly (optional 1-second delay for rate limiting).

**Technical details:** See [evaluators/inspect/job_submission.md](../evaluators/inspect/job_submission.md)

### 5. Monitor Job Progress

**Poll SLURM every 60 seconds:**
```bash
squeue -u $USER -o "%.18i %.9P %.50j %.8T %.10M %.6D"
```

**For completed jobs (disappeared from squeue):**
```bash
sacct -j {job_id} --format=JobID,State,Start,End,Elapsed
```

**Track state changes:**
- PENDING → RUNNING: Record transition
- RUNNING → COMPLETED: Record completion timestamp and elapsed time
- RUNNING → FAILED: Record failure and note to check logs

**Continue until all terminal:**
Stop monitoring when all jobs reach: COMPLETED, FAILED, CANCELLED, or TIMEOUT

**Note:** Evaluation jobs are typically faster than fine-tuning (5-10 minutes vs hours).

**Technical details:** See [evaluators/inspect/monitoring.md](../evaluators/inspect/monitoring.md)

### 6. Validate Completion

**Check dependency verification performed:**
Verify fine-tuning completion and checkpoints were checked before submission.

**Check all selected evaluations submitted:**
Verify job IDs captured for all submissions.

**Check all jobs terminal:**
No jobs still PENDING or RUNNING.

**Verify evaluation logs exist:**
```bash
ls {run_dir}/eval/logs/*.eval
```

For each COMPLETED evaluation, verify inspect-ai log file exists.

**Check log file created:**
Verify detailed execution log exists.

**Technical details:** See [evaluators/inspect/validation.md](../evaluators/inspect/validation.md)

## Logging

Create detailed log at `{experiment_dir}/run-inspect.log` (or similar name based on module invocation).

**Log entries:**
```
[YYYY-MM-DD HH:MM:SS] DISCOVER_EXPERIMENT
[YYYY-MM-DD HH:MM:SS] VERIFY_FINETUNING
[YYYY-MM-DD HH:MM:SS] VERIFY_CHECKPOINTS
[YYYY-MM-DD HH:MM:SS] IDENTIFY_EVALS
[YYYY-MM-DD HH:MM:SS] SUBMIT_EVAL: {run}/{task}/epoch{N}
[YYYY-MM-DD HH:MM:SS] STATUS_CHECK
[YYYY-MM-DD HH:MM:SS] STATE_CHANGE: {run}/{task}/epoch{N}
[YYYY-MM-DD HH:MM:SS] ALL_COMPLETE
```

## Error Handling

**If fine-tuning jobs still running:**
- Warn user
- Offer to wait or proceed
- If proceed, evaluations may fail without checkpoints

**If checkpoints missing:**
- Log warning for each missing checkpoint
- Skip those evaluations
- Continue with evaluations that have checkpoints
- Report skipped evaluations in summary

**If job submission fails:**
- Log error
- Continue with remaining evaluations
- Report failures in summary

**If an evaluation fails:**
- Record FAILED state
- Check SLURM and inspect-ai logs
- Continue monitoring other jobs
- Include in final summary

## Success Criteria

- ✓ Fine-tuning completion verified
- ✓ Model checkpoints verified
- ✓ All evaluations submitted successfully
- ✓ All jobs reached terminal states
- ✓ Evaluation logs exist for COMPLETED jobs
- ✓ Log file complete

## Important Notes

**Dependency checking is mandatory:**
Always verify fine-tuning complete and checkpoints exist before submitting evaluation jobs. Skipping this causes job failures.

**No stagger delay needed:**
Unlike fine-tuning (which needs 5-second stagger for cache), evaluations can be submitted rapidly.

**Polling efficiency:**
- Query all user jobs at once (not one by one)
- Poll every 60 seconds (not faster)
- Use sacct for completed jobs (they disappear from squeue)

**Resumability:**
Safe to re-run. Selection stage checks for already-completed evaluations and skips resubmission.

**Partial execution is acceptable:**
If some runs have missing checkpoints (fine-tuning failed), evaluations for other runs can still proceed.

**Viewing results:**
- Interactive: `inspect view --port=$(get_free_port)`
- Export: `inspect log export {run_dir}/eval/logs/*.eval --format csv`

# Logging

Create detailed log files that record all job execution actions. Logging is essential for debugging, reproducibility, and tracking experiment progress.

## Purpose

Logs enable:
1. **Debugging:** Track what jobs were submitted, status changes, and errors
2. **Reproducibility:** Another person (or Claude) can see exactly what was executed
3. **Progress tracking:** Monitor experiment progress over time
4. **Auditing:** Verify all runs/evaluations completed successfully

## Log Files

### For torchtune execution
**Location:** `{experiment_dir}/run-torchtune.log`

**Created by:** optimizers/torchtune/ modules

### For inspect-ai execution
**Location:** `{experiment_dir}/run-inspect.log`

**Created by:** evaluators/inspect/ modules

## Log Format

```
[{timestamp}] {ACTION_TYPE}: {details}
```

**Timestamp:** `YYYY-MM-DD HH:MM:SS` format

## Torchtune Log Example

```
[2025-11-11 11:00:00] DISCOVER_EXPERIMENT
Details: Found experiment at /path/to/experiment
Result: 4 runs configured

[2025-11-11 11:00:01] IDENTIFY_RUNS
Details: Llama-3.2-1B_r8_lr1e-5, Llama-3.2-1B_r8_lr5e-5, Llama-3.2-1B_r16_lr1e-5, Llama-3.2-1B_r16_lr5e-5
Result: All runs have finetune.slurm scripts

[2025-11-11 11:00:05] SUBMIT_JOB: Llama-3.2-1B_r8_lr1e-5
Details: sbatch finetune.slurm
Result: Job ID 1234567

[2025-11-11 11:00:10] SUBMIT_JOB: Llama-3.2-1B_r8_lr5e-5
Details: sbatch finetune.slurm (5-second stagger)
Result: Job ID 1234568

[2025-11-11 11:00:15] SUBMIT_JOB: Llama-3.2-1B_r16_lr1e-5
Details: sbatch finetune.slurm (5-second stagger)
Result: Job ID 1234569

[2025-11-11 11:00:20] SUBMIT_JOB: Llama-3.2-1B_r16_lr5e-5
Details: sbatch finetune.slurm (5-second stagger)
Result: Job ID 1234570

[2025-11-11 11:01:00] STATUS_CHECK
Details: squeue -u $USER --format="%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"
Result: All 4 jobs PENDING

[2025-11-11 11:02:00] STATUS_CHECK
Details: Polling interval (60 seconds)
Result: 2 PENDING, 2 RUNNING

[2025-11-11 11:02:00] STATE_CHANGE: Llama-3.2-1B_r8_lr1e-5
Details: PENDING → RUNNING
Result: Training started

[2025-11-11 11:02:00] STATE_CHANGE: Llama-3.2-1B_r8_lr5e-5
Details: PENDING → RUNNING
Result: Training started

[2025-11-11 11:15:00] STATE_CHANGE: Llama-3.2-1B_r8_lr1e-5
Details: RUNNING → COMPLETED
Result: Training finished

[2025-11-11 11:45:00] ALL_COMPLETE
Details: All 4 jobs reached terminal states
Result: 4 COMPLETED, 0 FAILED
Duration: 45 minutes
```

## Inspect-ai Log Example

```
[2025-11-11 12:00:00] DISCOVER_EXPERIMENT
Details: Found experiment at /path/to/experiment
Result: 4 runs configured for evaluation

[2025-11-11 12:00:01] VERIFY_FINETUNING
Details: Checking torchtune jobs via sacct
Result: All fine-tuning jobs COMPLETED

[2025-11-11 12:00:02] VERIFY_CHECKPOINTS
Details: Checking epoch_0, epoch_1 for 4 runs
Result: All 8 checkpoints exist

[2025-11-11 12:00:03] IDENTIFY_EVALS
Details: 4 runs × 2 tasks × 2 epochs
Result: 16 evaluations to submit

[2025-11-11 12:00:05] SUBMIT_EVAL: Llama-3.2-1B_r8_lr1e-5/capitalization/epoch0
Details: sbatch capitalization_epoch0.slurm
Result: Job ID 2345678

[2025-11-11 12:00:10] SUBMIT_EVAL: Llama-3.2-1B_r8_lr1e-5/capitalization/epoch1
Details: sbatch capitalization_epoch1.slurm (5-second stagger)
Result: Job ID 2345679

[... continues for all 16 evaluations ...]

[2025-11-11 12:01:00] STATUS_CHECK
Details: squeue -u $USER --format="%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"
Result: All 16 jobs PENDING or RUNNING

[2025-11-11 12:05:00] STATE_CHANGE: Llama-3.2-1B_r8_lr1e-5/capitalization/epoch0
Details: RUNNING → COMPLETED
Result: Evaluation finished

[2025-11-11 12:15:00] ALL_COMPLETE
Details: All 16 evaluations reached terminal states
Result: 16 COMPLETED, 0 FAILED
Duration: 15 minutes
```

## Action Types

### Torchtune Actions
- `DISCOVER_EXPERIMENT` - Find experiment directory
- `IDENTIFY_RUNS` - List runs to execute
- `SUBMIT_JOB` - Submit individual job
- `STATUS_CHECK` - Poll job statuses
- `STATE_CHANGE` - Job state transition
- `ALL_COMPLETE` - All jobs terminal

### Inspect-ai Actions
- `DISCOVER_EXPERIMENT` - Find experiment directory
- `VERIFY_FINETUNING` - Check training completion
- `VERIFY_CHECKPOINTS` - Check model checkpoints exist
- `IDENTIFY_EVALS` - List evaluations to execute
- `SUBMIT_EVAL` - Submit individual evaluation
- `STATUS_CHECK` - Poll job statuses
- `STATE_CHANGE` - Job state transition
- `ALL_COMPLETE` - All evaluations terminal

## When to Log

### During Each Stage

**Config Parsing:**
- Log which experiment_summary.md is being read
- Log extracted run/evaluation information

**Run/Evaluation Selection:**
- Log which runs/evaluations will be submitted
- Log any runs/evaluations being skipped (and why)

**Job Submission:**
- Log each job submitted (with job ID)
- Log any submission errors

**Monitoring:**
- Log status checks (every 60 seconds)
- Log state changes (PENDING → RUNNING → COMPLETED/FAILED)
- Log any monitoring errors

**Validation:**
- Log checkpoint verification
- Log completion verification
- Log final summary

## Module Responsibilities

Each module in optimizers/torchtune/ and evaluators/inspect/ should log its actions:

- **config_parsing.md:** Log experiment discovery and parsing
- **dependency_checking.md:** Log dependency verification (inspect only)
- **run_selection.md / evaluation_selection.md:** Log which runs/evals selected
- **job_submission.md:** Log each job submission
- **monitoring.md:** Log status checks and state changes
- **validation.md:** Log all validation checks

**See each module for specific logging requirements.**

## Important Notes

**Job ID tracking:**
- Always log job IDs when jobs are submitted
- Critical for monitoring and debugging

**State changes:**
- Log all state transitions (PENDING → RUNNING → COMPLETED/FAILED)
- Include timestamp to track execution time

**Errors:**
- Log all errors with sufficient detail for debugging
- Include SLURM error messages when jobs fail

**Staggering:**
- Note 5-second delays between submissions in logs
- Helps understand timeline when reviewing logs

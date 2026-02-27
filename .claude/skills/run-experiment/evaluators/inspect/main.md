# Inspect Evaluator Execution Module

Executes inspect-ai evaluation jobs for all runs in an experiment.

**CRITICAL:** Must run AFTER fine-tuning completes - evaluations require model checkpoints.

## Prerequisites

- experiment_summary.yaml exists
- Evaluation scaffolding complete (eval/*.slurm files exist)
- Fine-tuning complete (model checkpoints exist)
- SLURM cluster access

## Submodules

- **[parsing.md](parsing.md)** - Parse experiment_summary.yaml and scan for eval/*.slurm files
- **[dependency_checking.md](dependency_checking.md)** - **CRITICAL:** Verify fine-tuning complete, checkpoints exist
- **[evaluation_selection.md](evaluation_selection.md)** - Decide which evaluations to submit (skip completed, skip missing checkpoints)
- **[job_submission.md](job_submission.md)** - Submit SLURM jobs, capture IDs
- **[monitoring.md](monitoring.md)** - Poll squeue, track status, wait for completion
- **[validation.md](validation.md)** - Verify all jobs completed, evaluation logs exist

## Workflow Summary

1. Parse experiment to identify evaluations
2. **Verify fine-tuning complete** (dependency check)
3. Select which evaluations need submission
4. Submit jobs (no stagger needed - unlike fine-tuning)
5. Monitor with 60-second polling until all complete
6. Validate evaluation logs created successfully

## Output

- Creates logs/run-inspect.log with detailed execution history
- Evaluation logs in `{run_dir}/eval/logs/*.eval`

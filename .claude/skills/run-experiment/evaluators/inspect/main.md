# Inspect Evaluator Execution Module

Executes inspect-ai evaluation jobs for all runs in an experiment.

**CRITICAL:** Must run AFTER fine-tuning completes - evaluations require model checkpoints.

## Prerequisites

- experiment_summary.yaml exists
- Evaluation scaffolding complete (eval/*.slurm files exist)
- Fine-tuning complete (model checkpoints exist)
- SLURM cluster access

## Workflow

1. **[parsing.md](parsing.md)** - Parse experiment to identify evaluations
2. **[dependency_checking.md](dependency_checking.md)** - Verify fine-tuning complete, checkpoints exist
3. **[cache_prebuilding.md](cache_prebuilding.md)** - Pre-build HF datasets cache to prevent race conditions
4. **[evaluation_selection.md](evaluation_selection.md)** - Select which evaluations need submission
5. **[job_submission.md](job_submission.md)** - Submit SLURM jobs, capture IDs
6. **[monitoring.md](monitoring.md)** - Poll squeue, track status, wait for completion
7. **[validation.md](validation.md)** - Verify all jobs completed, evaluation logs exist

Every step is required. Execute steps in order — each step depends on the output of the previous one.

## Output

- Creates run-inspect.log with detailed execution history
- Evaluation logs in `{run_dir}/eval/logs/*.eval`

# Torchtune Optimizer Execution Module

Executes torchtune fine-tuning jobs for all runs in an experiment.

## Prerequisites

- experiment_summary.yaml exists
- Fine-tuning scaffolding complete (finetune.slurm files exist)
- SLURM cluster access

## Submodules

- **[parsing.md](parsing.md)** - Parse experiment_summary.yaml and scan for finetune.slurm files
- **[run_selection.md](run_selection.md)** - Decide which runs to submit (skip completed/running)
- **[job_submission.md](job_submission.md)** - Submit SLURM jobs, capture IDs, stagger submissions
- **[monitoring.md](monitoring.md)** - Poll squeue, track status, wait for completion
- **[validation.md](validation.md)** - Verify all jobs completed, model checkpoints exist

## Workflow Summary

1. Parse experiment to identify runs
2. Select which runs need submission
3. Submit jobs with 5-second stagger (prevents cache collision)
4. Monitor with 60-second polling until all complete
5. Validate checkpoints created successfully

## Output

- Creates logs/run-torchtune.log with detailed execution history
- Model checkpoints in `{output_dir_base}/ck-out-{run_name}/epoch_{N}/`

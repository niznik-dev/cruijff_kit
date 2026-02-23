# Torchtune Optimizer Execution Module

Executes torchtune fine-tuning jobs for all runs in an experiment.

## Prerequisites

- experiment_summary.yaml exists
- Fine-tuning scaffolding complete (finetune.slurm files exist)
- SLURM cluster access

## Workflow

1. **[parsing.md](parsing.md)** - Parse experiment to identify runs
2. **[run_selection.md](run_selection.md)** - Select which runs need submission
3. **[job_submission.md](job_submission.md)** - Submit jobs with 5-second stagger (prevents cache collision)
4. **[monitoring.md](monitoring.md)** - Monitor with 60-second polling until all complete
5. **[validation.md](validation.md)** - Validate checkpoints created successfully

Every step is required. Execute steps in order — each step depends on the output of the previous one.

## Output

- Creates run-torchtune.log with detailed execution history
- Model checkpoints in `{output_dir_base}/ck-out-{run_name}/epoch_{N}/`

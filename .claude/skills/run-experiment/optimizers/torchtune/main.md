# Torchtune Optimizer Execution Module

Executes torchtune fine-tuning jobs for all fine-tuned runs in an experiment by invoking the canonical submitter:

```
python -m cruijff_kit.tools.run.submit_torchtune <experiment_dir>
```

The submitter (`src/tools/run/submit_torchtune.py`) handles every operational step in code:

1. Parses `experiment_summary.yaml` and discovers fine-tuned runs (skips controls).
2. Drip-feeds `sbatch` for each run's `finetune.slurm`, capping concurrency at the `MAX_SUBMIT` env var (default 25, the gpu-test QoS limit).
3. Staggers 5 seconds between submissions to dodge the HF datasets cache race.
4. Polls SLURM (`squeue` first, `sacct` fallback) every 60 seconds until every job reaches a terminal state.
5. Emits the canonical 4-line `SUBMIT_JOB:` / `Job ID:` / `Result:` blocks to `logs/run-torchtune.log` (consumed by `analyze-experiment`'s compute step via `harvest_jids_from_run_logs()` in `tools/slurm/compute_metrics.py`).
6. Persists `logs/run-torchtune.state.json` for resume after interruption — keyed by `{relative_path}/finetune.slurm` so eval jobs in different runs don't collide.

## Prerequisites

- `experiment_summary.yaml` exists.
- Fine-tuning scaffolding complete (`finetune.slurm` files exist).
- SLURM cluster access.

## Outputs

- `logs/run-torchtune.log` — canonical submission + state-change records.
- `logs/run-torchtune.state.json` — resume state file.
- Model checkpoints under `{output_dir_base}/{run_name}/artifacts/epoch_{N}/`.

## Schemas

The remaining files in this directory describe individual concerns for downstream readers. The submitter is the single source of truth for runtime behavior:

- [parsing.md](parsing.md) — how runs are identified from `experiment_summary.yaml`.
- [run_selection.md](run_selection.md) — fine-tuned vs control filtering.
- [job_submission.md](job_submission.md) — canonical 4-line log block shape (must match `harvest_jids_from_run_logs` regex).
- [monitoring.md](monitoring.md) — terminal-state polling.
- [validation.md](validation.md) — checkpoint verification after completion.

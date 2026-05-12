# Torchtune Optimizer Execution Module

Executes torchtune fine-tuning jobs for all fine-tuned runs in an experiment by invoking the canonical submitter:

```
python -m cruijff_kit.tools.run.submit_torchtune <experiment_dir>
```

The submitter (`src/tools/run/submit_torchtune.py`) handles every operational step in code:

1. Parses `experiment_summary.yaml` and discovers fine-tuned runs (skips controls).
2. Drip-feeds `sbatch` for each run's `finetune.slurm`, capping concurrency at `--max-submit` (CLI flag) > `MAX_SUBMIT` (env var) > 25 (gpu-test QoS default).
3. Staggers between submissions to dodge the HF datasets cache race (`--stagger-sec` > `STAGGER_SEC` env > 5s default).
4. Polls SLURM (`squeue` first, `sacct` fallback) until every job reaches a terminal state, at the configured cadence (`--poll-sec` > `POLL_SEC` env > 60s default).
5. Emits the canonical 4-line `SUBMIT_JOB:` / `Job ID:` / `Result:` blocks to `logs/run-torchtune.log` (consumed by `analyze-experiment`'s compute step via `harvest_jids_from_run_logs()` in `tools/slurm/compute_metrics.py`).
6. Persists `logs/run-torchtune.state.json` for resume after interruption — keyed by `{relative_path}/finetune.slurm` so eval jobs in different runs don't collide.

## Detach and resume

The submitter exits cleanly on `SIGINT`, `SIGTERM`, or when `<experiment_dir>/logs/.detach` is present. State is flushed and a `MONITOR_DETACHED` block is appended to the log; SLURM jobs are untouched. To re-attach the monitor without resubmitting:

```
python -m cruijff_kit.tools.run.submit_torchtune <experiment_dir> --resume-monitor
```

Or for a one-shot read across both submitters: `python -m cruijff_kit.tools.run.status <experiment_dir>`.

## Live monitor settings

The watcher re-reads `<experiment_dir>/logs/monitor.json` on every poll iteration. Any of `poll_sec`, `stagger_sec`, `max_submit` listed there take effect on the next poll, overriding the CLI / env / default values. Changes are recorded as `MONITOR_CONFIG` blocks in the per-tool log. See [the run-experiment skill](../../SKILL.md#live-monitor-settings) for the full precedence chain.

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

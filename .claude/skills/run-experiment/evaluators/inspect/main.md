# Inspect Evaluator Execution Module

Executes inspect-ai evaluation jobs for all runs in an experiment by invoking the canonical submitter:

```
python -m cruijff_kit.tools.run.submit_inspect <experiment_dir>
```

**CRITICAL:** Must run AFTER fine-tuning completes — evaluations require model checkpoints.

The submitter (`src/tools/run/submit_inspect.py`) handles every operational step in code:

1. Globs `*/eval/*.slurm` under the experiment directory.
2. Drip-feeds `sbatch`, capping concurrency at `max_submit`. Staggers `stagger_sec` between submissions — the HF datasets cache race hits whenever multiple jobs hit `datasets.load_dataset` at once, not just on fine-tunes.
3. Polls SLURM (`squeue` first, `sacct` fallback) until every job reaches a terminal state, at `poll_sec` cadence.
4. Emits canonical 4-line `SUBMIT_EVAL:` / `Job ID:` / `Result:` blocks to `logs/run-inspect.log`. The eval-job identifier is `{run_name}/{task}/epoch{N}` (or `{run_name}/{task}` when no epoch suffix exists), matching the regex in `analyze-experiment`'s compute step.
5. Persists `logs/run-inspect.state.json` keyed by `{run_name}/eval/{slurm_filename}` so distinct runs with the same eval slurm filename don't collide on a single state-file key (the bug from issue #451 comment #2).

The three knobs (`max_submit`, `stagger_sec`, `poll_sec`) resolve in this order: `--max-submit` / `--stagger-sec` / `--poll-sec` (CLI) > `<repo>/.config/config.json` > built-in defaults (25 / 5 / 60).

## Detach and resume

The submitter exits cleanly on `SIGINT`, `SIGTERM`, or when `<experiment_dir>/logs/.detach` is present. State is flushed and a `MONITOR_DETACHED` block is appended to the log; SLURM jobs are untouched. To re-attach the monitor without resubmitting:

```
python -m cruijff_kit.tools.run.submit_inspect <experiment_dir> --resume-monitor
```

Or for a one-shot read across both submitters: `python -m cruijff_kit.tools.run.status <experiment_dir>`.

## Live monitor settings

The watcher re-reads `<experiment_dir>/logs/monitor.json` on every poll iteration. Any of `poll_sec`, `stagger_sec`, `max_submit` listed there take effect on the next poll, overriding the CLI and user-config values. Changes are recorded as `MONITOR_CONFIG` blocks in the per-tool log. See [the run-experiment skill](../../SKILL.md#tuning-watcher-cadence-monitorjson--repo-defaults) for the full precedence chain.

## Prerequisites

- `experiment_summary.yaml` exists.
- Evaluation scaffolding complete (`*/eval/*.slurm` files exist).
- Fine-tuning complete (model checkpoints exist).
- SLURM cluster access.

## Outputs

- `logs/run-inspect.log` — canonical submission + state-change records.
- `logs/run-inspect.state.json` — resume state file.
- Evaluation logs in `{run_dir}/eval/logs/*.eval`.

## Schemas

The remaining files in this directory describe individual concerns for downstream readers. The submitter is the single source of truth for runtime behavior:

- [parsing.md](parsing.md) — how evaluations are identified.
- [dependency_checking.md](dependency_checking.md) — checkpoint existence preconditions.
- [cache_prebuilding.md](cache_prebuilding.md) — pre-building HF datasets cache.
- [evaluation_selection.md](evaluation_selection.md) — which evals need submission.
- [job_submission.md](job_submission.md) — canonical 4-line log block shape.
- [monitoring.md](monitoring.md) — terminal-state polling.
- [validation.md](validation.md) — eval log verification after completion.

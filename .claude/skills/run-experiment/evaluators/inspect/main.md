# Inspect Evaluator Execution Module

Executes inspect-ai evaluation jobs for all runs in an experiment by invoking the canonical submitter:

```
python -m cruijff_kit.tools.run.submit_inspect <experiment_dir>
```

**CRITICAL:** Must run AFTER fine-tuning completes — evaluations require model checkpoints.

The submitter (`src/tools/run/submit_inspect.py`) handles every operational step in code:

1. Globs `*/eval/*.slurm` under the experiment directory.
2. Drip-feeds `sbatch`, capping concurrency at `MAX_SUBMIT` (default 25, override via env). The 5-second stagger applies on the eval side too — the HF datasets cache race hits whenever multiple jobs hit `datasets.load_dataset` at once, not just on fine-tunes.
3. Polls SLURM (`squeue` first, `sacct` fallback) every 60 seconds until every job reaches a terminal state.
4. Emits canonical 4-line `SUBMIT_EVAL:` / `Job ID:` / `Result:` blocks to `logs/run-inspect.log`. The eval-job identifier is `{run_name}/{task}/epoch{N}` (or `{run_name}/{task}` when no epoch suffix exists), matching the regex in `analyze-experiment`'s compute step.
5. Persists `logs/run-inspect.state.json` keyed by `{run_name}/eval/{slurm_filename}` so distinct runs with the same eval slurm filename don't collide on a single state-file key (the bug from issue #451 comment #2).

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

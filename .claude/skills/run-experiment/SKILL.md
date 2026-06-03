---
name: run-experiment
description: Execute the complete experimental workflow - model optimization followed by evaluation - for all runs in a scaffolded experiment. Use after scaffold-experiment to submit jobs to SLURM.  
---

## Your Task

Orchestrate experiment execution by reading tool specifications from experiment_summary.yaml and calling the appropriate tool modules **sequentially**:

1. Read experiment_summary.yaml to identify tools being used
2. Execute model optimization (fine-tuning) for all runs
3. Wait for optimization to complete (REQUIRED)
4. Execute model evaluation for all runs

This ensures the entire experiment runs from training through evaluation with proper dependency management.

## Prerequisites

- experiment_summary.yaml exists (from design-experiment skill)
- Scaffolding complete (from scaffold-experiment skill)
- SLURM cluster access

## Dependency version check

Run before proceeding to catch stale envs (user pulled new pins but didn't re-run `pip install -e .`):

```bash
python scripts/check_env.py
```

- **Exit 0**: proceed.
- **Exit 1**: show the printed `STALE ENV` table to the user, ask whether to `pip install -e .` first or continue anyway.

## Workflow

### High-Level Steps

1. **Locate experiment** - Find experiment directory (current dir or ask user)
2. **Verify scaffolding** - Ensure configs exist for optimization and evaluation
3. **Read tool specifications** - Parse experiment_summary.yaml "tools" section
4. **Execute optimization** - Call optimizer module (torchtune)
5. **Execute evaluation** - Call evaluator module (inspect) - **MUST wait for optimization**
6. **Create orchestration log** - Document process in `logs/run-experiment.log`
7. **Report combined summary** - Show complete status

### Tool Modules

This skill invokes callable Python submitters that handle SLURM submission, drip-feed under the gpu-test QoS cap, monitoring to terminal state, and canonical log emission as a side-effect. The Python is the source of truth; the prose modules under `optimizers/` and `evaluators/` describe the schemas and flow for downstream readers.

**Optimizer modules:** See [optimizers/](optimizers/) for the schema description.
- torchtune (fine-tuning) → `python -m cruijff_kit.tools.run.submit_torchtune <experiment_dir>` (writes `logs/run-torchtune.log` + `logs/run-torchtune.state.json`)

**Evaluator modules:** See [evaluators/](evaluators/) for the schema description.
- inspect-ai → `python -m cruijff_kit.tools.run.submit_inspect <experiment_dir>` (writes `logs/run-inspect.log` + `logs/run-inspect.state.json`)

Both submitters are resume-safe: re-invoking after an interruption reads the JSON state file and skips already-submitted entries. Both emit canonical `SUBMIT_JOB:` / `SUBMIT_EVAL:` lines that `explore`'s compute-utilization step harvests. Future tools (DSPy, custom trainers, custom evaluators) plug in here as additional submitters.

## Reading Tool Specifications

Parse experiment_summary.yaml "tools" section to identify frameworks:

**Expected format:**
```yaml
tools:
  preparation: "torchtune"
  evaluation: "inspect-ai"
```

**Tool to module mapping:**
- `torchtune` → [optimizers/torchtune/](optimizers/torchtune/)
- `inspect-ai` → [evaluators/inspect/](evaluators/inspect/)

**If tools section missing:** Error out and ask the user to add a `tools` section to `experiment_summary.yaml` — don't silently assume defaults.

## Sequential Execution

**CRITICAL:** Evaluation MUST wait for optimization to complete.

**Why?** Evaluation jobs need optimized model checkpoints.

**Implementation:**
1. Run `python -m cruijff_kit.tools.run.submit_torchtune <experiment_dir>` — submits, drip-feeds, monitors, writes `logs/run-torchtune.log` + `logs/run-torchtune.state.json`. Blocks until all jobs reach terminal state.
2. Only after step 1 returns successfully, run `python -m cruijff_kit.tools.run.submit_inspect <experiment_dir>` — same pattern for evaluations, writes `logs/run-inspect.log` + `logs/run-inspect.state.json`. Blocks until terminal.
3. Append a high-level summary to `logs/run-experiment.log` (the orchestration log). The detailed per-job records already live in the per-tool logs.

## Long-Running / Detachable Monitoring

The default flow above blocks until all jobs finish. For experiments that may run for hours, the submitters support **detach** so a watcher can stop without killing the jobs. Three trigger paths converge on a clean exit (flush state → emit canonical `MONITOR_DETACHED` block → return; jobs keep running):

| Mechanism | Who uses it |
|---|---|
| `touch <experiment_dir>/logs/.detach` | Anyone — humans, agents, `/loop`, cron. Sentinel is sticky: remove it before re-attaching. |
| `SIGINT` (Ctrl+C) | A human at the terminal where the submitter is running |
| `SIGTERM` | A parent process killing the submitter subprocess |

After a clean detach, **re-attaching** is one of:

```
# One-shot snapshot (read-only; refreshes state from squeue/sacct):
python -m cruijff_kit.tools.run.status <experiment_dir>

# Continuous re-monitor without resubmitting:
python -m cruijff_kit.tools.run.submit_torchtune <experiment_dir> --resume-monitor
python -m cruijff_kit.tools.run.submit_inspect   <experiment_dir> --resume-monitor
```

`status` is composable with `/loop` (e.g. periodic check-ins from an agent or cron job). It never submits anything and never writes `MONITOR_DETACHED`; it refreshes state, appends `STATE_CHANGE` blocks when SLURM has moved a job since the last refresh, and emits a per-tool `ALL_COMPLETE` block the first time refresh observes all-terminal — so a finished experiment closes its log cleanly without needing a follow-up `--resume-monitor`. `ALL_COMPLETE` is idempotent: at most one block per per-tool log, no matter how many `status` or submitter calls observe completion.

## Tuning Watcher Cadence: monitor.json + Repo Defaults

The submitter has three numeric knobs:

| Knob | Built-in | What it controls |
|---|---|---|
| `poll_sec` | 60 | Seconds between SLURM state polls |
| `stagger_sec` | 5 | Delay between successive `sbatch` calls (HF cache race) |
| `max_submit` | 25 | Cap on simultaneous queued jobs (gpu-test QoS limit) |

Built-in defaults live in `<repo>/.config/config.json`, a tracked file. To change personal defaults across all future experiments, edit that file. If you want to keep local edits out of `git status`, run `git update-index --skip-worktree .config/config.json` once.

For a single experiment, use the CLI flags (`--poll-sec`, `--stagger-sec`, `--max-submit`) at submitter invocation.

For **live mid-run** tuning without detach + re-attach, edit `<experiment_dir>/logs/monitor.json` — the watcher re-reads it on every poll iteration. Same filesystem-as-control-plane shape as `.detach`: anyone with FS access (human, agent, `/loop`, cron) can edit. Recognized keys (all optional):

```json
{"poll_sec": 30, "stagger_sec": 5, "max_submit": 25}
```

**Precedence (high → low):** `monitor.json` > CLI flag > `<repo>/.config/config.json` > built-in default. Missing files are silent no-ops. Malformed JSON or out-of-range values emit a `WARNING:` to stderr and the watcher keeps the previous values. Every applied `monitor.json` change writes a canonical `MONITOR_CONFIG` block to the per-tool log.

**Filesystem control channels under `<experiment_dir>/logs/`:**

| File | Action | Effect |
|---|---|---|
| `.detach` | `touch` | Watcher exits cleanly; jobs continue. Sticky — `rm` it before re-attaching. |
| `monitor.json` | edit | Adjust `poll_sec` / `stagger_sec` / `max_submit` live. Takes effect on next poll. |
| `run-*.state.json` | read | Authoritative resume state; produced by the submitters. |

## Logging

Execution is logged in three files (see logging.md for details).
All logs live under the `logs/` subdirectory per the canonical artifact layout:
- `{experiment_dir}/logs/run-experiment.log` - Orchestration log (high-level flow, kept short)
- `{experiment_dir}/logs/run-torchtune.log` - Fine-tuning execution (detailed)
- `{experiment_dir}/logs/run-inspect.log` - Evaluation execution (detailed)

**Log format:**
```
[YYYY-MM-DD HH:MM:SS] ACTION: Description
Details: {specifics}
Result: {outcome}
```

**What to log:**
- Experiment discovery and validation
- Scaffolding verification
- Tool module invocations (timestamps, results, durations)
- Completion status (successes/failures)
- Errors or warnings
- Final combined summary
- Paths to results and module logs

## Expected Outputs

After successful execution:

**Logs created** (in `{experiment_dir}/logs/`):
- `run-experiment.log` - Orchestration log (high-level flow, kept short)
- `run-torchtune.log` - Fine-tuning execution log (detailed)
- `run-inspect.log` - Evaluation execution log (detailed)

**Status updated:**
- Run tracking logs updated with job IDs, timestamps, states
- All execution details recorded in module logs

**Artifacts created:**
- Model checkpoints from optimization
- Evaluation logs from evaluation

---

## Error Handling

**If experiment_summary.yaml not found:**
- Suggest running design-experiment skill first
- Do not proceed

**If scaffolding incomplete:**
- Report which parts missing
- Suggest running scaffold-experiment skill
- Can proceed with optimization only if just evaluation configs missing

**If optimization fails:**
- Log failure details
- Do NOT proceed to evaluation (missing model checkpoints)
- Report failure and stop

**If evaluation fails:**
- Log failure details
- Optimization results still valid
- Report partial success

**If user cancels (Ctrl+C / SIGTERM / `touch logs/.detach`):**
- Watcher exits cleanly, writes `MONITOR_DETACHED` to the per-tool log
- SLURM jobs continue running independently
- To resume monitoring without resubmitting:
  - `python -m cruijff_kit.tools.run.status <experiment_dir>` for a one-shot snapshot
  - `python -m cruijff_kit.tools.run.submit_{torchtune,inspect} <experiment_dir> --resume-monitor` to watch until terminal
- If detach was triggered by the sentinel file, `rm <experiment_dir>/logs/.detach` first (sticky)

## Validation Checklist

Before reporting success, verify:
- ✓ experiment_summary.yaml found and read
- ✓ Scaffolding verified
- ✓ Optimizer module executed and completed
- ✓ Evaluator module executed and completed
- ✓ Model checkpoints exist
- ✓ Evaluation logs exist
- ✓ Orchestration log created
- ✓ All module logs exist

## Output Summary

Provide comprehensive summary after completion:

```markdown
## Run Experiment Complete

Experiment: `{experiment_dir}`

### Optimization Results

✓ {N}/{M} runs completed successfully
Duration: {duration}

**Completed runs:** [list with times]
**Failed runs:** [list with errors]
**Model checkpoints:** {paths}

### Evaluation Results

✓ {N}/{M} evaluations completed successfully
Duration: {duration}

**Completed evaluations:** [list with times]
**Failed evaluations:** [list with errors]
**Evaluation logs:** {paths}

### Total Time

Complete workflow: {total_duration}
- Optimization: {opt_duration}
- Evaluation: {eval_duration}

### GPU Utilization

| Run | Type | Wall Time | Time Limit | GPU Util | GPU Mem (GB) | Power (W) |
| --- | --- | --- | --- | --- | --- | --- |
| {run_name} | finetune | {wall_time} | {time_limit} | {gpu_util}% | {gpu_mem} | {power}W |
| {run_name} | eval | {wall_time} | {time_limit} | {gpu_util}% | {gpu_mem} | {power}W |

*GPU metrics from nvidia-smi background monitoring. seff data captured per job.*

### Next Steps

1. View results: `inspect view --port=$(get_free_port)`
2. Export data: `inspect log export ...`
3. Summarize results: run `summarize-experiment` (the required post-run step). `explore` is optional and can be run any time afterward.
```

### Next Step: Summarize Results

After completing the experiment, run `summarize-experiment` — this is the standard post-run step and produces the `summary.md` every experiment should have:

> Experiment complete! Running `summarize-experiment` to capture key metrics into summary.md.

**Then, optionally:** offer a deeper analysis.

> Want me to also run `explore` for visualizations and a full report? It's optional and can be run any time. [y/N]

**If yes:** Invoke the `explore` skill to create interactive plots and `analysis/report.md`.

**If no:** Skip it — `explore` can be run any time later against the same evaluation logs.

## Important Notes

**Orchestration principles:**
- This skill invokes callable Python submitters (`src/tools/run/submit_*.py`) that handle the operational work and emit canonical execution logs as a side-effect. No prose-by-prose recipe; the script is the contract.
- Each submitter maintains its own detailed log (`logs/run-torchtune.log`, `logs/run-inspect.log`) AND a JSON resume state file alongside it.
- Sequential execution is mandatory (evaluation requires optimization complete).
- Partial success is acceptable (some runs succeed, others fail) — the state files capture which ones.
- Each submitter can be invoked directly (CLI or `from cruijff_kit.tools.run import submit_torchtune; submit_torchtune.run(experiment_dir)`).

**Relationship to other skills:**
- **Before:** design-experiment, scaffold-experiment
- **After:** summarize-experiment (the required next step), then optionally explore (any time)
- **Standalone:** Individual tool modules can run independently

**Resumability:**
- Re-running run-experiment is safe
- Tool modules check for completed jobs
- Won't re-submit successful jobs

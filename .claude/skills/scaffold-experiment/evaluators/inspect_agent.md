# scaffold-inspect Subagent - Evaluation

This module contains the specification for launching the scaffold-inspect subagent, which generates evaluation configurations for all runs.

---

## When to Use

Launch this subagent when:
- `tools.evaluation` in experiment_summary.yaml is set to `"inspect-ai"`
- You need to generate evaluation configurations for experimental runs

---

## Subagent Invocation

**Subagent type:** `scaffold-inspect`

**Launch via:** Task tool (use in parallel with preparation subagent)

---

## Prompt Template

```
Set up inspect-ai evaluation configurations for all runs in the experiment located at {experiment_dir}.

Your tasks:
1. Read experiment_summary.yaml to extract evaluation configurations
2. Read claude.local.md for environment-specific settings
3. Verify that inspect-ai task scripts exist at the specified paths
4. For each run and evaluation combination:
   a. Create eval/ subdirectory (with logs/ inside) in the run directory
   b. Generate eval_config.yaml with experiment-specific values (see Step 4 below)
   c. Call setup_inspect.py to render the SLURM script from the template (see Step 5 below)
5. Create a detailed log at {experiment_dir}/scaffold-inspect.log

Report back:
- Summary of all created evaluation scripts (paths)
- Any errors or warnings encountered
- Verification results for task script existence
- Path to the log file for detailed information
```

---

## What scaffold-inspect Does

The subagent performs these operations autonomously:

1. **Parses evaluation plan** from experiment_summary.yaml (which runs, which tasks, which epochs)
2. **Verifies task scripts exist** at paths specified in experiment_summary.yaml
3. **Creates `eval/` subdirectories** (with `logs/` inside) in each run directory
4. **Generates `eval_config.yaml`** in each eval directory with all experiment-specific config
5. **Calls `setup_inspect.py`** to render SLURM scripts from the template (see below)
6. **Creates detailed log** at `scaffold-inspect.log` with complete process information

### Step 4: Generate eval_config.yaml

For each evaluation, create `eval_config.yaml` in the eval directory with all experiment-specific configuration. See `agents/scaffold-inspect.md` for the full schema (required keys, optional task args, metadata, scorer config).

### Step 5: Render SLURM scripts via setup_inspect.py

After writing `eval_config.yaml`, call `setup_inspect.py` to render the SLURM script from the template. See `agents/scaffold-inspect.md` for full CLI reference and details on what the renderer handles.

### Eval time limit (ask user)

Eval time depends on multiple factors (model size, holdout set size, evaluation task complexity, hardware), so it is **not** a model property. Ask the user during scaffolding:

> What time limit should eval jobs use? (HH:MM:SS)
>
> This depends on model size, holdout set size, and task complexity.
> For reference, training time for this experiment is set to {training_time}.
> Eval jobs are typically much faster than training.
>
> 1. 0:10:00 (10 min — good for small models/datasets) (Recommended)
> 2. 0:20:00 (20 min — good for larger models/datasets)
> 3. 0:30:00 (30 min — conservative)

Use the same time limit for all eval jobs in the experiment. If the user doesn't have a preference, default to `0:10:00`.

**Why this matters:** Eval jobs often finish in seconds but hold a GPU for the full allocation. A 30-minute default for a 4-second job wastes 29+ minutes of GPU time per eval.

---

## Expected Output Structure

After scaffold-inspect completes, the experiment directory will contain:

```
{experiment_dir}/
├── Llama-3.2-1B-Instruct_base/    # Control run
│   └── eval/
│       ├── eval_config.yaml
│       ├── {task_name}_base.slurm
│       └── logs/
├── Llama-3.2-1B-Instruct_rank4/   # Fine-tuned run
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   ├── finetune.slurm
│   └── eval/
│       ├── eval_config.yaml
│       ├── {task_name}_epoch0.slurm
│       └── logs/
├── Llama-3.2-1B-Instruct_rank8/   # Fine-tuned run
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   ├── finetune.slurm
│   └── eval/
│       ├── eval_config.yaml
│       ├── {task_name}_epoch0.slurm
│       └── logs/
├── scaffold-torchtune.log
└── scaffold-inspect.log
```

---

## Subagent Report Format

When scaffold-inspect completes, it will report back with:

**On success:**
- Number of evaluation scripts created
- List of created eval directories and scripts
- Task script verification results
- Path to scaffold-inspect.log

**On failure:**
- Which evaluations failed and why
- Missing task scripts
- Path to scaffold-inspect.log for detailed diagnostics

---

## Error Handling

**If scaffold-inspect fails:**
- The subagent will report errors in its response
- Log the failure in orchestration log (see [logging.md](../logging.md))
- Note which evaluations couldn't be scaffolded
- Fine-tuning can still proceed (evaluation optional)
- Report the failure in final summary
- Direct user to scaffold-inspect.log for details

**Common failure scenarios:**
- Missing inspect-ai task scripts at specified paths
- Invalid evaluation matrix in experiment_summary.yaml
- Permission issues creating eval/ directories
- Missing environment variables in claude.local.md

---

## Integration with Orchestrator

1. **Launched by:** scaffold-experiment orchestrator (SKILL.md)
2. **Runs in parallel with:** scaffold-torchtune subagent
3. **Logs to:** scaffold-inspect.log (detailed), scaffold-experiment.log (high-level status)
4. **Reports back to:** scaffold-experiment orchestrator upon completion
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

For each evaluation, create `eval_config.yaml` in the eval directory. This file contains all experiment-specific configuration — both what the SLURM renderer needs and what the inspect task reads at runtime.

**Required keys** (used by `setup_inspect.py` for SLURM rendering):

```yaml
task_script: /path/to/experiments/task.py@task_name
task_name: acs_income
model_path: /outputs/run1/epoch_0
model_hf_name: hf/1B_ft_epoch_0
output_dir: /outputs/run1/
```

**Optional keys** (task args passed as `-T`, metadata passed as `--metadata`):

```yaml
# Task args (-T key=value)
data_path: /data/acs_income.json
vis_label: 1B_ft
use_chat_template: "true"

# Metadata (--metadata key=value)
epoch: 0
finetuned: true
source_model: Llama-3.2-1B-Instruct
```

**Experiment-specific config** (not used by SLURM renderer, read by inspect task at runtime):

```yaml
scorer:
  - name: match
  - name: risk_scorer
    params:
      option_tokens: ["0", "1"]
system_prompt: ""
prompt: "{input}\n"
```

Note: `config_path` and `eval_dir` are **auto-derived** from the location of the YAML file — do not include them in eval_config.yaml.

### Step 5: Call setup_inspect.py

After writing `eval_config.yaml`, generate the SLURM script by running:

```bash
cd {run_dir}/eval
python tools/inspect/setup_inspect.py \
  --config eval_config.yaml \
  --model_name "Llama-3.2-1B-Instruct" \
  --time "0:10:00" \
  --account "msalganik" \
  --conda_env "cruijff"
```

This renders `tools/inspect/templates/eval_template.slurm` with the correct values. The template includes GPU monitoring, proper SLURM headers, and SLURM log management automatically.

**What setup_inspect.py handles:**
- GPU resources (`mem`, `cpus`, `partition`, `constraint`) from `model_configs.py`
- GPU monitoring (nvidia-smi background logging)
- `cd` to eval_dir before running inspect
- SLURM log move on success
- Output filename: `{task_name}_epoch{epoch}.slurm` (or `{task_name}.slurm` if no epoch)

**CLI arguments** (infrastructure — shared across all evals in the experiment):

| Arg | Required | Description |
|-----|----------|-------------|
| `--config` | Yes | Path to eval_config.yaml |
| `--model_name` | Yes | Key in MODEL_CONFIGS for SLURM resource lookup |
| `--time` | No | SLURM time limit (default: `0:10:00`) |
| `--account` | No | SLURM account |
| `--conda_env` | No | Conda environment (default: `cruijff`) |
| `--mem` | No | Override model_configs memory |
| `--partition` | No | Override model_configs partition |
| `--constraint` | No | Override model_configs constraint |
| `--output_slurm` | No | Override output filename |

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
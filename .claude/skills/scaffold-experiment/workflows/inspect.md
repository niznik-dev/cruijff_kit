# Inspect-ai Scaffolding Workflow

This document describes the detailed step-by-step process for scaffolding inspect-ai evaluation configurations.

## Prerequisites

- experiment_summary.md exists with evaluation tasks defined
- claude.local.md exists with environment settings
- Run directories exist (created by torchtune scaffolding)
- Inspect-ai task scripts exist (or note as prerequisite)

## Step-by-Step Process

### 1. Parse Evaluation Configuration

**Read experiment_summary.md:**
- Extract evaluation tasks from "Evaluation Tasks" table
  - Task names
  - Script paths
  - Dataset paths (if specified)
- Parse "Evaluation Plan" section
  - Which epochs to evaluate (last, all, specific list)
  - Which runs get which evaluations
  - Base model evaluations
- Extract system prompt from Configuration section

**Read claude.local.md:**
- Extract conda_env
- Extract scratch_dir
- Extract account (optional, for SLURM)

### 2. Verify Inspect-AI Tasks

**For each evaluation task:**

```bash
# Check if task script exists
if [ -f "{task_script_path}" ]; then
  echo "✓ Task script found: {task_script_path}"
else
  echo "✗ Task script not found: {task_script_path}"
  echo "  Suggest running create-inspect-task skill"
  continue
fi

# List available tasks in the script
module load anaconda3/2025.6
conda activate {conda_env}
inspect list {task_script_path}

# Verify expected task name appears in output
```

**Example output:**
```
cap_task.py
  capitalization
```

**If task name doesn't match:**
- Report the mismatch
- Show available tasks from `inspect list` output
- Suggest updating experiment_summary.md or renaming task function

### 3. Determine Evaluation Matrix

**Parse evaluation plan to determine:**

```python
# Which epochs to evaluate
if evaluation_plan == "last":
    epochs = [last_epoch]  # e.g., [0] for 1 epoch of training
elif evaluation_plan == "all":
    epochs = list(range(num_epochs))  # e.g., [0, 1] for 2 epochs
else:
    epochs = parse_epoch_list(evaluation_plan)  # e.g., "0,2" → [0, 2]

# Which runs to evaluate
fine_tuned_runs = [r for r in runs if r['type'] == 'Fine-tuned']
base_runs = [r for r in runs if r['type'] == 'Control']

# Build evaluation matrix
evaluations = []
for run in fine_tuned_runs:
    for task in tasks:
        for epoch in epochs:
            evaluations.append({
                'run': run,
                'task': task,
                'epoch': epoch,
                'type': 'fine-tuned'
            })

for run in base_runs:
    for task in tasks:
        evaluations.append({
            'run': run,
            'task': task,
            'type': 'base'
        })
```

**Example matrix:**
```
8 runs × 1 task × 1 epoch = 8 evaluations (fine-tuned)
1 run × 1 task = 1 evaluation (base model)
Total: 9 evaluations
```

### 4. Create Evaluation Directories

**For each run directory:**

```bash
mkdir -p {experiment_dir}/{run_directory}/eval
mkdir -p {experiment_dir}/{run_directory}/eval/logs
```

**For base model runs (if applicable):**

```bash
mkdir -p {experiment_dir}/{run_directory}_base/eval
mkdir -p {experiment_dir}/{run_directory}_base/eval/logs
```

### 5. Generate SLURM Scripts

**For each evaluation in the matrix:**

#### 5.1 Determine Evaluation Scenario

```python
if eval['type'] == 'fine-tuned':
    if task_supports_config_path:
        scenario = 'fine_tuned_with_config'  # PREFERRED
    else:
        scenario = 'fine_tuned_explicit'
elif eval['type'] == 'base':
    scenario = 'base_model'
```

#### 5.2 Build Model Path

**CRITICAL: All paths must be absolute (start with /), never relative (../file). SLURM working directories are unpredictable.**

```python
if scenario == 'fine_tuned_with_config':
    model_path = f"{output_dir_base}/ck-out-{run_name}/epoch_{epoch_num}"
    config_path = f"{experiment_dir}/{run_dir}/setup_finetune.yaml"  # absolute path required
elif scenario == 'base_model':
    model_path = base_model_path  # From experiment_summary.md
    config_path = f"{experiment_dir}/{run_dir}/setup_finetune.yaml"  # absolute path required
```

#### 5.3 Generate SLURM Script

**Script name:**
- Fine-tuned: `{run_dir}/eval/{task_name}_epoch{N}.slurm`
- Base model: `{run_dir}_base/eval/{task_name}_base.slurm`

**Script content:**

```bash
#!/bin/bash
#SBATCH --job-name=eval-{task_name}-{run_id}
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
{if account specified: #SBATCH --account={account}}

# Load environment
module load anaconda3/2025.6
conda activate {conda_env}

# CRITICAL: Paths must be absolute (start with /), never relative (../file)
{if fine-tuned:}
MODEL_PATH="{output_dir_base}/ck-out-{run_name}/epoch_{N}"
CONFIG_PATH="{experiment_dir}/{run_dir}/setup_finetune.yaml"

{if base model:}
MODEL_PATH="{base_model_path}"
CONFIG_PATH="{experiment_dir}/{run_dir}/setup_finetune.yaml"

# Run inspect-ai evaluation
cd {experiment_dir}/{run_dir}/eval

{if scenario == 'fine_tuned_with_config':}
inspect eval {task_script_path}@{task_name} \\
  --model hf/local \\
  -M model_path="$MODEL_PATH" \\
  -T config_path="$CONFIG_PATH" \\
  --log-dir ./logs \\
  --log-level info

{if scenario == 'base_model':}
inspect eval {task_script_path}@{task_name} \\
  --model hf/local \\
  -M model_path="$MODEL_PATH" \\
  -T dataset_path="{eval_dataset_path}" \\
  -T system_prompt="{system_prompt}" \\
  --log-dir ./logs \\
  --log-level info

echo "Evaluation complete"
```

#### 5.4 Make Script Executable

```bash
chmod +x {experiment_dir}/{run_dir}/eval/{task_name}_epoch{N}.slurm
```

### 6. Log Results

**Create scaffold.log entries:**

```
[YYYY-MM-DD HH:MM:SS] SCAFFOLD_INSPECT_START
Details: Using evaluators/inspect/main.md logic
Result: Analyzing experiment_summary.md for evaluation tasks and plan

[YYYY-MM-DD HH:MM:SS] PARSE_EVAL_TASKS
Details: Found {N} tasks: {task_list}
Result: Task script paths verified

[YYYY-MM-DD HH:MM:SS] VERIFY_TASK_SCRIPTS
Command: inspect list {task_script_path}
Result: Task '{task_name}' found and verified

[YYYY-MM-DD HH:MM:SS] DETERMINE_EVAL_MATRIX
Details: {N_runs} runs × {N_tasks} tasks × {N_epochs} epochs
Result: Will create {total} evaluation scripts

[YYYY-MM-DD HH:MM:SS] CREATE_EVAL_DIRS
Details: Creating eval/ and logs/ in {N} run directories
Result: All directories created successfully

[YYYY-MM-DD HH:MM:SS] GENERATE_SLURM_SCRIPTS
Details: {scenario description}
Result: {N} evaluation scripts created

[YYYY-MM-DD HH:MM:SS] SCAFFOLD_INSPECT_COMPLETE
Details: {summary}
Duration: {elapsed_time}
```

## Error Handling

**If evaluation tasks table is missing:**
- Report error
- Ask user to add evaluation tasks to experiment_summary.md
- Do not proceed

**If task script doesn't exist:**
- Log warning for that specific task
- Continue with other tasks
- Suggest running `create-inspect-task` skill
- Note in summary that some tasks need creation

**If evaluation plan is unclear:**
- Ask user for clarification
- Examples: "last", "all", "0,2"
- Remind about 0-indexing

**If unclear which scenario to use:**
- Check if task accepts `config_path` parameter
- Fall back to `dataset_path` + `system_prompt` approach
- Log the decision

## Success Criteria

- ✓ All evaluation tasks verified (or noted as missing)
- ✓ Evaluation matrix determined
- ✓ All eval/ directories created
- ✓ All SLURM scripts generated with correct paths
- ✓ Scripts use appropriate scenario (config_path vs dataset_path)
- ✓ scaffold.log contains complete process details

## Important Notes

### Path Requirements

**CRITICAL: All paths in SLURM scripts must be absolute (start with /), never relative.** SLURM working directories are unpredictable; relative paths like `../setup_finetune.yaml` will fail with FileNotFoundError. This applies to MODEL_PATH, CONFIG_PATH, and all paths passed via -T or -M parameters.

### Epoch Indexing

**CRITICAL: Epochs are 0-indexed**
- Training for 1 epoch produces checkpoint at `epoch_0/`
- Training for 2 epochs produces `epoch_0/` and `epoch_1/`
- "Last epoch" after 1 epoch of training = `epoch_0`
- Evaluation script names: `{task_name}_epoch0.slurm`, NOT `epoch1`

### Model Paths

Fine-tuned model paths reference output directories that **don't exist yet**:
- Paths like `/scratch/.../ck-out-r8_lr1e-5/epoch_0/` won't exist until training completes
- This is expected - evaluation scripts are generated before training
- Evaluations should only be submitted AFTER fine-tuning completes

### System Prompt Consistency

**CRITICAL: System prompt must match training configuration**
- Extract system prompt from experiment_summary.md Configuration section
- Use same prompt in both training and evaluation
- Often an empty string `""`
- Mismatch will cause evaluation to test wrong behavior

### Config Integration

**Preferred approach:** Use `config_path` parameter
- Task reads dataset path and system prompt from `setup_finetune.yaml`
- Ensures configuration consistency for both fine-tuned and base models
- Simpler command (fewer parameters)

**Fallback approach:** Use explicit parameters
- Task accepts `dataset_path` and `system_prompt` directly
- More explicit but harder to maintain consistency

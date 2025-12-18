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

### 4. Extract Evaluation Parameters

**Purpose:** Extract `data_path`, `prompt`, and `system_prompt` values that will be baked into SLURM scripts.

**For fine-tuned runs:**
Read `setup_finetune.yaml` from each run directory and extract:
- `data_path` = `input_dir_base` + `dataset_label` + `dataset_ext`
- `prompt` = the `prompt` field
- `system_prompt` = the `system_prompt` field

**For base/control runs:**
Extract from experiment_summary.md:
- `data_path`: From Resources → Dataset → Path
- `prompt`: From Configuration → prompt
- `system_prompt`: From Configuration → System prompt
- `model_path`: From Resources → Models (base model path)

**No config files needed:** Values are baked directly into SLURM scripts at scaffolding time.

### 5. Create Evaluation Directories

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

### 6. Generate SLURM Scripts

**For each evaluation in the matrix:**

#### 6.0 Determine Model-Aware Resources

Different model sizes require different SLURM resources. Parse the model name from experiment_summary.md and set resources accordingly:

| Model Size | Memory | GPUs | Constraint | CPUs |
|------------|--------|------|------------|------|
| 1B | 32G | 1 | - | 4 |
| 3B | 64G | 1 | gpu80 | 4 |
| 8B | 96G | 1 | gpu80 | 8 |
| 70B | 256G | 4 | gpu80 | 8 |

**Detection logic:**
1. Parse model name from experiment_summary.md Resources → Models section
2. Look for size indicator: "1B", "3B", "8B", "70B"
3. Apply corresponding resource configuration
4. Default to 1B settings if model size cannot be determined

#### 6.1 Build Model Path

**CRITICAL: All paths must be absolute (start with /), never relative (../file). SLURM working directories are unpredictable.**

```python
if eval['type'] == 'fine-tuned':
    model_path = f"{output_dir_base}/ck-out-{run_name}/epoch_{epoch_num}"
elif eval['type'] == 'base':
    model_path = base_model_path  # From experiment_summary.md
```

#### 6.2 Generate SLURM Script

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
#SBATCH --cpus-per-task={cpus_from_model_size}
#SBATCH --mem={mem_from_model_size}
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:{gpus_from_model_size}
{if account specified: #SBATCH --account={account}}
{if 3B or larger: #SBATCH --constraint=gpu80}

# Load environment
module load anaconda3/2025.6
conda activate {conda_env}

# CRITICAL: All paths must be absolute (start with /), never relative (../file)
MODEL_PATH="{model_path}"

# Dataset and prompt configuration
# (extracted from setup_finetune.yaml or experiment_summary.md at scaffolding time)
DATA_PATH="{data_path}"
PROMPT="{prompt}"
SYSTEM_PROMPT="{system_prompt}"

# Run inspect-ai evaluation
cd {experiment_dir}/{run_dir}/eval

inspect eval {task_script_path}@{task_name} \\
  --model hf/local \\
  -M model_path="$MODEL_PATH" \\
  -T data_path="$DATA_PATH" \\
  -T prompt="$PROMPT" \\
  -T system_prompt="$SYSTEM_PROMPT" \\
  --log-dir ./logs \\
  --log-level info

echo "Evaluation complete"
```

#### 6.3 Make Script Executable

```bash
chmod +x {experiment_dir}/{run_dir}/eval/{task_name}_epoch{N}.slurm
```

### 7. Log Results

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

[YYYY-MM-DD HH:MM:SS] EXTRACT_EVAL_PARAMS
Details: Extracted data_path, prompt, system_prompt for {N} runs
Result: Parameters ready for SLURM script generation

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

## Success Criteria

- ✓ All evaluation tasks verified (or noted as missing)
- ✓ Evaluation matrix determined
- ✓ All eval/ directories created
- ✓ All SLURM scripts generated with correct paths and baked-in parameters
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

### Evaluation Parameters

**Approach:** Extract and bake parameters into SLURM scripts
- `data_path`, `prompt`, `system_prompt` are extracted at scaffolding time
- Values come from `setup_finetune.yaml` (fine-tuned) or `experiment_summary.md` (base models)
- Parameters are hardcoded into SLURM scripts - no runtime config parsing
- Ensures explicit, visible configuration and training/eval parity

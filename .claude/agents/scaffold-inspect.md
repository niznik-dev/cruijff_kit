---
name: scaffold-inspect
description: Sets up inspect-ai evaluation configurations for all runs in a designed experiment. Reads experiment_summary.yaml and generates {task}_epoch{N}.slurm scripts for each run/evaluation combination.
tools: Read, Edit, Write, Grep, Glob, Bash
permissionMode: default
---

You help automatically set up inspect-ai evaluation configurations for all runs in a designed experiment. Your task is to read an `experiment_summary.yaml` file and generate all the necessary inspect-ai files (`{task}_epoch{N}.slurm` scripts) so that evaluation runs are ready to submit to SLURM after fine-tuning completes.

## Invocation Context

This subagent can be invoked in two ways:

1. **By orchestrator** (scaffold-experiment skill): The orchestrator provides the experiment directory path in the invocation prompt. Work autonomously and report back results in a single comprehensive response.

2. **Standalone** (direct invocation): A user invokes this subagent directly. You may ask clarifying questions if needed.

**When reporting back to an orchestrator:** Provide a complete summary including all created evaluation scripts, any errors encountered, verification results, and the path to the log file. The orchestrator cannot send follow-up messages.

## Core Responsibilities Workflow

1. **Locate experiment** - Find the experiment directory (usually current directory or ask user)
2. **Read experiment_summary.yaml** - Parse the experiment plan to extract evaluation configuration
3. **Read claude.local.md** - Get environment-specific settings (conda env, etc.)
4. **Verify inspect-ai tasks exist** - Check if evaluation task scripts are available
5. **For each run and evaluation combination:**
   - Generate `inspect.slurm` script for that run/epoch
   - Configure model paths, task parameters, output locations
   - Verify configuration
6. **Create scaffold log** - Document all actions taken in `scaffold-inspect.log`
7. **Report summary** - Show user what was created and any issues

## Input Format

### Finding the Experiment

**If user invokes subagent without arguments:**
- Check if current directory contains `experiment_summary.yaml`
- If not, ask user for the experiment directory path

**If user provides a path:**
- Use that path as the experiment directory

### Parsing experiment_summary.yaml

Extract the following information from the YAML structure:

1. **Experiment metadata:**
   - `experiment.name` - Experiment identifier
   - `experiment.directory` - Full path to experiment directory

2. **Models:**
   - `models.base[0].name` - Model identifier
   - `models.base[0].path` - Full path to model directory

3. **Dataset:**
   - `data.training.path` - Full path to training dataset
   - `data.training.label` - Dataset filename without extension
   - `data.training.format` - "json" or "parquet"

4. **Output configuration:**
   - `output.base_directory` - Where checkpoints are saved
   - `controls.system_prompt` - System prompt (must match training)

5. **Runs:**
   - `runs[]` - List of all runs (fine-tuned + control)
   - For each run: `name`, `type`, `model`, `parameters`

6. **Evaluation configuration:**
   - `evaluation.system_prompt` - Must match training
   - `evaluation.temperature` - Evaluation temperature
   - `evaluation.scorer[]` - List of scorers with optional params (see Parsing Scorer Configuration below)
   - `evaluation.tasks[]` - List of evaluation tasks
   - `evaluation.matrix[]` - Which runs evaluate on which tasks/epochs

#### Parsing Evaluation Tasks

From `evaluation.tasks[]` in YAML:
```yaml
tasks:
  - name: "capitalization"
    script: "path/to/cap_task.py"
    dataset: "path/to/test.json"  # Optional
    description: "Tests capitalization"
```

Extract for each task:
- `name` - Task identifier (for naming evaluation outputs)
- `script` - Full path to inspect-ai task file
- `dataset` - Path to eval dataset (optional, if different from training)
- `description` - Human-readable description

#### Parsing Evaluation Matrix

From `evaluation.matrix[]` in YAML:
```yaml
matrix:
  - run: "Llama-3.2-1B-Instruct_rank4"
    vis_label: "rank4"  # optional, defaults to run name
    tasks: ["capitalization"]
    epochs: [0]
  - run: "Llama-3.2-1B-Instruct_control"
    vis_label: "1B_control"
    tasks: ["capitalization"]
    epochs: null  # null for control/base runs
```

Determine for each run:
- Which tasks to evaluate on
- Which epochs to evaluate (0-indexed, or null for control models)
- Whether this is a fine-tuned or control run
- The `vis_label` for visualization (defaults to `run` name if not specified)

#### Parsing Scorer Configuration

From `evaluation.scorer[]` in YAML:
```yaml
scorer:
  - name: "match"
  - name: "includes"
  - name: "risk_scorer"
    params:
      option_tokens: ["0", "1"]
```

Extract for each scorer:
- `name` - Scorer identifier (e.g., "match", "includes", "risk_scorer")
- `params` - Optional dict of parameters to pass to the scorer (e.g., `{option_tokens: ["0", "1"]}`)

**Backward compatibility:** If `evaluation.scorer` is a plain string (e.g., `"match"`), treat it as a single scorer with no params: `[{name: "match"}]`.

The scorer configuration is written into `eval_config.yaml` so that task files can read it at runtime and instantiate scorers dynamically.

### Reading claude.local.md

Extract environment-specific settings:
- `conda_env` - Which conda environment to use
- `account` - SLURM account to use (OPTIONAL)
- `working_directory` - Path to cruijff_kit repo (for PYTHONPATH export)

### Parsing Output Directory from experiment_summary.yaml

**IMPORTANT:** Read `output.base_directory` from experiment_summary.yaml to construct model paths.

The base_directory contains the full path: `{output_base}/ck-outputs/{experiment_name}`
- Example: `/scratch/gpfs/MSALGANIK/sarahep/ck-outputs/workflow_test_2025-11-28`

For generating inspect.slurm scripts:
- Use base_directory directly to construct OUTPUT_BASE paths
- Fine-tuned model path: `{base_directory}/ck-out-{run_name}/epoch_{N}`
- Example: `/scratch/gpfs/MSALGANIK/sarahep/ck-outputs/workflow_test_2025-11-28/ck-out-rank4/epoch_0`

## Verifying Inspect-AI Tasks

For each evaluation task in the experiment:

1. **Check if task script exists:**
   ```bash
   ls {task_script_path}
   ```

2. **List available tasks in the script:**
   ```bash
   module load anaconda3/2025.6
   conda activate {conda_env}
   inspect list {task_script_path}
   ```
   This shows all `@task` decorated functions in the file, confirming:
   - The task name exists
   - The exact spelling/capitalization
   - What other tasks are available

   Example output:
   ```
   cap_task.py
     cap_task
   ```

   Use this to verify the task name in experiment_summary.yaml matches what's actually in the file.

3. **If task doesn't exist:**
   - Note in log that task needs to be created
   - Suggest running `create-inspect-task` skill first
   - Continue with other tasks (don't fail completely)

4. **Verify task is compatible with experiment:**
   - Task should accept `data_path`, `prompt`, and `system_prompt` parameters
   - These are the standard parameters for chat_completion-trained models
   - Check docstring/parameters if accessible

## Handling Control Model Evaluation

Control runs don't undergo fine-tuning, so they won't have a `setup_finetune.yaml` file. For these runs, scaffold-inspect extracts values directly from experiment_summary.yaml and bakes them into the SLURM script.

### Detection Logic

From the `runs[]` list in experiment_summary.yaml, identify control runs:
- Look for Type column = "Control"
- These runs have no LoRA rank (or LoRA Rank = "-")
- Example: `| Llama-3.2-1B-Instruct_control | Llama-3.2-1B-Instruct | - | Control | N/A |`

### Extracting Values for Control Evaluation

Extract the following from experiment_summary.yaml:
- `data_path`: Full path from Resources → Dataset → Path
- `prompt`: From Configuration → prompt (e.g., `"{input}"`)
- `system_prompt`: From Configuration → System prompt
- `model_path`: From Resources → Models (the control model path)

1. **Create directories for control model evaluation:**
   ```bash
   # Example for run named "Llama-3.2-1B-Instruct_control"
   mkdir -p {experiment_dir}/Llama-3.2-1B-Instruct_control/eval
   mkdir -p {experiment_dir}/Llama-3.2-1B-Instruct_control/eval/logs
   ```

2. **Generate eval_config.yaml** with content extracted from experiment_summary.yaml:
   ```yaml
   # Evaluation configuration for base model
   # Generated by scaffold-inspect
   #
   # Run: {run_name}
   # Generated: {timestamp}

   # Dataset configuration
   input_dir_base: {from Dataset section}
   dataset_label: {dataset name without extension}
   dataset_ext: {.json or / for parquet}
   dataset_type: {text_completion or chat_dataset}

   # System prompt (from Configuration → Evaluation section or this run's config)
   system_prompt: "{system_prompt for this run}"

   # Scorer configuration (from evaluation.scorer in experiment_summary.yaml)
   scorer:
     - name: "match"
     - name: "risk_scorer"
       params:
         option_tokens: ["0", "1"]
   ```

3. **Extract values from experiment_summary.yaml:**
   - `input_dir_base`: Path portion of dataset path (e.g., `/path/to/data/capitalization/`)
   - `dataset_label`: Dataset filename without extension (e.g., `words_5L_80P_1000`)
   - `dataset_ext`: File extension (`.json` for JSON, `/` for parquet directory)
   - `dataset_type`: Usually `text_completion`, or `chat_dataset` if using chat format
   - `system_prompt`: From the run's configuration (may vary per run!)

### Handling Multiple Control Runs with Different System Prompts

When experiment_summary.yaml includes multiple base runs with different system prompts:

```yaml
runs:
  - name: "Llama-3.2-1B-Instruct_control_helpful"
    type: "control"
    model: "Llama-3.2-1B-Instruct"
    parameters: {}
    # Could have run-specific system_prompt if needed
  - name: "Llama-3.2-1B-Instruct_control_concise"
    type: "control"
    model: "Llama-3.2-1B-Instruct"
    parameters: {}
```

Each run's SLURM script will have its own `SYSTEM_PROMPT` variable set appropriately.

### Determining Chat Template Usage

When generating inspect.slurm scripts, determine whether to use chat templates based on the model type and dataset type used during training.

**Detection Logic:**

1. **For fine-tuned models:** Read `dataset_type` from `setup_finetune.yaml`
   - If `dataset_type: chat_completion` → `use_chat_template=true`
   - If `dataset_type: text_completion` → `use_chat_template=false`

2. **For control models:** Check if model name ends with "-Instruct"
   - Instruct models (e.g., `Llama-3.2-1B-Instruct`) → `use_chat_template=true`
   - Base models (e.g., `Llama-3.2-1B`) → `use_chat_template=false`

**In SLURM script generation:**

```bash
# For instruct models / chat_completion:
USE_CHAT_TEMPLATE="true"

# For base models / text_completion:
USE_CHAT_TEMPLATE="false"
```

**Note:** When `use_chat_template=false`:
- The `system_prompt` parameter is still passed but will be ignored by the inspect task
- Base models receive prompts without chat formatting, matching training behavior

## Generating Inspect SLURM Scripts

For each evaluation to be performed, generate an `inspect.slurm` script.

### Evaluation Naming Convention

**IMPORTANT: Epochs are 0-indexed**
- First epoch after training is `epoch_0`, not `epoch_1`
- Training for 1 epoch produces checkpoint at `epoch_0/`
- Training for 2 epochs produces `epoch_0/` and `epoch_1/`
- Evaluation script names must match: `{task_name}_epoch0.slurm`, not `epoch1`
- When experiment_summary.yaml evaluation matrix specifies epoch 0 after 1 epoch of training, use `epoch_0`

Organize evaluations within run directories:

**For fine-tuned models:**
```
{experiment_dir}/{run_dir}/
├── finetune.slurm
├── finetune.yaml
├── setup_finetune.yaml
└── eval/
    ├── {task_name}_epoch0.slurm
    ├── {task_name}_epoch1.slurm
    └── ...
```

**For control models (not fine-tuned):**
```
{experiment_dir}/{run_dir}_control/
└── eval/
    └── {task_name}_base.slurm
```

### Model-Aware Resource Allocation

Different model sizes require different SLURM resources for evaluation. Parse the model name from experiment_summary.yaml and set resources accordingly:

| Model Size | Memory | GPUs | Constraint | CPUs | Time |
|------------|--------|------|------------|------|------|
| 1B (Llama-3.2-1B-Instruct) | 32G | 1 | - | 4 | 0:30:00 |
| 3B (Llama-3.2-3B-Instruct) | 64G | 1 | gpu80 | 4 | 0:30:00 |
| 8B (Llama-3.1-8B-Instruct, etc.) | 96G | 1 | gpu80 | 8 | 0:30:00 |
| 70B (Llama-3.3-70B-Instruct, etc.) | 256G | 4 | gpu80 | 8 | 0:30:00 |

**Detection logic:**
1. Parse model name from experiment_summary.yaml Resources → Models section
2. Look for size indicator in model name: "1B", "3B", "8B", "70B"
3. Apply corresponding resource configuration
4. Default to 1B settings if model size cannot be determined

**Example parsing:**
- `Llama-3.2-1B-Instruct` → 1B resources
- `Llama-3.2-3B-Instruct` → 3B resources
- `Llama-3.1-8B-Instruct` → 8B resources
- `Llama-3.3-70B-Instruct` → 70B resources

### SLURM Script Template

Generate a SLURM script for each evaluation with model-appropriate resources.

**Key principle:** Extract dataset path, prompt, and system prompt from `setup_finetune.yaml` (or experiment_summary.yaml for control models) at scaffolding time, then pass them directly to inspect eval. This avoids config file parsing at runtime.

```bash
#!/bin/bash
#SBATCH --job-name=eval-{task_name}-{run_id}-ep{N}
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_from_model_size}
#SBATCH --mem={mem_from_model_size}
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:{gpus_from_model_size}
{optional: #SBATCH --account={account}}
{if 3B or larger: #SBATCH --constraint=gpu80}

# Load environment
module load anaconda3/2025.6
conda activate {conda_env}

# Add cruijff_kit to PYTHONPATH for custom scorers/tools
export PYTHONPATH="{working_directory}:$PYTHONPATH"

# Model path
{if fine-tuned:}
OUTPUT_BASE="{base_directory}/ck-out-{run_name}"
MODEL_PATH="$OUTPUT_BASE/epoch_{N}"
CONFIG_PATH="$OUTPUT_BASE/setup_finetune.yaml"
{if control model:}
MODEL_PATH="{control_model_path}"

# Dataset and prompt configuration
# (extracted from setup_finetune.yaml at scaffolding time)
DATA_PATH="{data_path}"
PROMPT="{prompt}"
SYSTEM_PROMPT="{system_prompt}"
{if chat_completion or instruct model:}USE_CHAT_TEMPLATE="true"
{if text_completion or base model:}USE_CHAT_TEMPLATE="false"

# Run inspect-ai evaluation
cd {experiment_dir}/{run_dir}/eval

inspect eval {task_script_path}@{task_name} \\
  {if fine-tuned:}--model hf/{run_name}_epoch_{N} \\
  {if control model:}--model hf/{run_name}_control \\
  -M model_path="$MODEL_PATH" \\
  {if fine-tuned:}--metadata epoch={N} \\
  {if fine-tuned:}--metadata finetuned=true \\
  {if control model:}--metadata finetuned=false \\
  --metadata source_model="{model_name from experiment_summary.yaml}" \\
  -T data_path="$DATA_PATH" \\
  -T prompt="$PROMPT" \\
  -T system_prompt="$SYSTEM_PROMPT" \\
  -T vis_label="{matrix[].vis_label or matrix[].run}" \\
  -T use_chat_template="$USE_CHAT_TEMPLATE" \\
  --log-dir ./logs \\
  --log-level info

echo "Evaluation complete"
```

### Script Configuration

**Important:** Write the shebang `#!/bin/bash` as the first line with NO escaping (not `#\!/bin/bash`).

**SLURM parameters:**
- Time: Default to 30 minutes (adjust based on experiment estimates if available)
- GPUs/Memory/CPUs: Set based on model size (see Model-Aware Resource Allocation table above)
- Constraint: gpu80 required for 3B+ models
- Account: Use from claude.local.md if specified

**Model paths:**
- Fine-tuned: `{output.base_directory}/ck-out-{run_name}/epoch_{N}`
- Control model: Original model path from `models.base[0].path`

**Task parameters (passed directly to inspect eval):**
- `data_path`: Full path to dataset file (constructed from `input_dir_base` + `dataset_label` + `dataset_ext`)
- `prompt`: The prompt template used during training (e.g., `"{input}"` or `"Capitalize: {input}"`)
- `system_prompt`: System message used during training (often empty string)

**Extracting values for fine-tuned models:**
Read `setup_finetune.yaml` from the run directory and extract:
- `data_path` = `input_dir_base` + `dataset_label` + `dataset_ext`
- `prompt` = the `prompt` field
- `system_prompt` = the `system_prompt` field

**Writing scorer config for all runs (fine-tuned and control):**
When generating `eval_config.yaml`, include the `scorer` list from `evaluation.scorer` in experiment_summary.yaml. This allows task files to read scorer configuration at runtime and instantiate scorers dynamically (e.g., `risk_scorer(option_tokens=["0", "1"])`).

**Extracting values for control models:**
Use values from experiment_summary.yaml Configuration section (same as fine-tuned runs use).

**Output location:**
- Log directory: `{run_dir}/eval/logs/`
- SLURM output: `{run_dir}/eval/slurm-{job_id}.out`

### Required Metadata Flags

**CRITICAL:** All evaluation SLURM scripts MUST include these flags for inspect-viz filtering.

**Model args (`-M`) - used by HF provider:**
| Flag | Value | Source |
|------|-------|--------|
| `-M model_path` | `"$MODEL_PATH"` | Variable set above |

**Eval metadata (`--metadata`) - stored in .eval log for filtering:**

For fine-tuned models:
| Flag | Value | Source |
|------|-------|--------|
| `--metadata epoch` | `{N}` | From matrix epochs list |
| `--metadata finetuned` | `true` | Literal |
| `--metadata source_model` | `"{model_name}"` | From `models.base[].name` |

For control models:
| Flag | Value | Source |
|------|-------|--------|
| `--metadata finetuned` | `false` | Literal |
| `--metadata source_model` | `"{model_name}"` | From `models.base[].name` |

**Task args (`-T`) - passed to task function:**
| Flag | Value | Source |
|------|-------|--------|
| `-T vis_label` | `"{vis_label}"` | From `matrix[].vis_label` or defaults to `matrix[].run` |

## Handling Different Evaluation Scenarios

**Standard approach:** Extract all values at scaffolding time and bake them into SLURM scripts as variables. This ensures consistency and avoids runtime config parsing.

> **Note:** Some legacy inspect tasks support a `config_dir` parameter for runtime config reading. This is not used by scaffold-inspect - we always bake values directly.

### Scenario 1: Fine-tuned Model Evaluation

Fine-tuned models use `setup_finetune.yaml` from the base output directory:
```bash
OUTPUT_BASE="{base_directory}/ck-out-{run_name}"
MODEL_PATH="$OUTPUT_BASE/epoch_0"
CONFIG_PATH="$OUTPUT_BASE/setup_finetune.yaml"
```

```bash
# Add cruijff_kit to PYTHONPATH for custom scorers/tools
export PYTHONPATH="{working_directory}:$PYTHONPATH"

# Values extracted from setup_finetune.yaml at scaffolding time:
MODEL_PATH="/absolute/path/to/ck-out-{run_name}/epoch_0"
DATA_PATH="/path/to/data/green/capitalization/words_5L_80P_1000.json"
PROMPT="{input}"
SYSTEM_PROMPT=""
USE_CHAT_TEMPLATE="true"  # from dataset_type: chat_completion

inspect eval capitalization.py@capitalization \\
  --model hf/{run_name}_epoch_0 \\
  -M model_path="$MODEL_PATH" \\
  --metadata epoch=0 \\
  --metadata finetuned=true \\
  --metadata source_model="Llama-3.2-1B-Instruct" \\
  -T data_path="$DATA_PATH" \\
  -T prompt="$PROMPT" \\
  -T system_prompt="$SYSTEM_PROMPT" \\
  -T use_chat_template="$USE_CHAT_TEMPLATE" \\
  -T vis_label="rank4" \\
  --log-dir ./logs
```

**Key points:**
- The `--model` argument uses a descriptive name (`hf/{run_name}_epoch_{N}`) that gets recorded in the `.eval` file for identification
- Metadata flags (`--metadata epoch`, `--metadata finetuned`, `--metadata source_model`) are stored in `log.eval.metadata` for inspect-viz filtering/grouping
- The `vis_label` task arg sets a dynamic task name suffix (e.g., `capitalization_rank4`) for visualization
- Values are extracted from `setup_finetune.yaml` **at scaffolding time** and baked into the SLURM script
- `use_chat_template` is determined from `dataset_type` in setup_finetune.yaml
- No config file parsing happens at eval runtime
- Ensures exact match between training and evaluation parameters

### Scenario 2: Control Model Evaluation

For control (not fine-tuned) models, scaffold-inspect generates `eval_config.yaml` (see "Extracting Values for Control Models" above), then reads from it at scaffolding time:

**For instruct models:**
```bash
# Add cruijff_kit to PYTHONPATH for custom scorers/tools
export PYTHONPATH="{working_directory}:$PYTHONPATH"

# Values extracted from eval_config.yaml at scaffolding time:
MODEL_PATH="/path/to/pretrained-llms/Llama-3.2-1B-Instruct"
DATA_PATH="/path/to/data/green/capitalization/words_5L_80P_1000.json"
PROMPT="{input}"
SYSTEM_PROMPT=""
USE_CHAT_TEMPLATE="true"  # Instruct model, use chat template

inspect eval capitalization.py@capitalization \\
  --model hf/{run_name}_base \\
  -M model_path="$MODEL_PATH" \\
  --metadata finetuned=false \\
  --metadata source_model="Llama-3.2-1B-Instruct" \\
  -T data_path="$DATA_PATH" \\
  -T prompt="$PROMPT" \\
  -T system_prompt="$SYSTEM_PROMPT" \\
  -T use_chat_template="$USE_CHAT_TEMPLATE" \\
  -T vis_label="1B_Instruct_control" \\
  --log-dir ./logs
```

**For base/foundation models (non-instruct):**
```bash
# Values for base model without instruct training:
MODEL_PATH="/path/to/pretrained-llms/Llama-3.2-1B"
DATA_PATH="/path/to/data/green/capitalization/words_5L_80P_1000.json"
PROMPT="{input}"
SYSTEM_PROMPT=""
USE_CHAT_TEMPLATE="false"  # Base model, no chat template

inspect eval capitalization.py@capitalization \\
  --model hf/{run_name}_base \\
  -M model_path="$MODEL_PATH" \\
  --metadata finetuned=false \\
  --metadata source_model="Llama-3.2-1B" \\
  -T data_path="$DATA_PATH" \\
  -T prompt="$PROMPT" \\
  -T system_prompt="$SYSTEM_PROMPT" \\
  -T use_chat_template="$USE_CHAT_TEMPLATE" \\
  -T vis_label="1B_control" \\
  --log-dir ./logs
```

**Key points:**
- The `--model` argument uses a descriptive name (`hf/{run_name}_control`) that gets recorded in the `.eval` file for identification
- The `vis_label` task arg sets a dynamic task name suffix (e.g., `capitalization_1B_control`) for visualization
- Control models use the same dataset/prompt/system_prompt as fine-tuned runs for fair comparison
- Values come from `eval_config.yaml` (generated from experiment_summary.yaml)
- Mirrors fine-tuned approach: config file → SLURM script for auditability

## Directory Structure Creation

Create eval directories as needed:

```bash
# For each run directory
mkdir -p {experiment_dir}/{run_dir}/eval
mkdir -p {experiment_dir}/{run_dir}/eval/logs

# Write SLURM script
cat > {experiment_dir}/{run_dir}/eval/{task_name}_epoch{N}.slurm << 'EOF'
{script content}
EOF

chmod +x {experiment_dir}/{run_dir}/eval/{task_name}_epoch{N}.slurm
```

## Error Handling

**If experiment_summary.yaml not found:**
- Ask user for experiment directory path
- Verify file exists before proceeding

**If evaluation task information missing:**
- Report what's missing (task script path, dataset, etc.)
- Ask user to update experiment_summary.yaml or regenerate with design-experiment
- Don't proceed without complete information

**If inspect-ai task script doesn't exist:**
- Log warning for that task
- Continue with other tasks
- Note in summary that some tasks need creation
- Suggest running `create-inspect-task` skill

**If unclear which evaluation approach to use:**
- Check if task file has `config_dir` parameter (preferred for experiments)
- Fall back to `dataset_path` + `system_prompt` approach
- Log the decision

## Logging

Create a detailed log file at `{experiment_dir}/scaffold-inspect.log`:

### Log Format

```
[YYYY-MM-DD HH:MM:SS] ACTION: Description
Details: {specifics}
Result: {outcome}

```

### What to Log

- Experiment discovery and validation
- Parsing experiment_summary.yaml evaluation configuration
- Verification of inspect-ai task scripts
- Evaluation matrix analysis (which runs, which epochs, which tasks)
- Directory creation
- SLURM script generation for each evaluation
- Any errors or warnings
- Final summary of created evaluation configs

### Example Log Entries

```
[2025-10-24 17:00:00] DISCOVER_EXPERIMENT: Found experiment
Details: /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/experiment_summary.yaml
Result: Successfully read experiment plan (8 runs, 1 evaluation task)

[2025-10-24 17:00:05] PARSE_EVAL_TASKS: Extracting evaluation configuration
Details: Found 1 task: capitalization (cap_task.py)
Result: Task script path verified: /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/experiments/capitalization/cap_task.py

[2025-10-24 17:00:10] PARSE_EVAL_PLAN: Determining evaluation matrix
Details: Evaluate last epoch only, all runs on all tasks
Result: Will generate 8 evaluations (8 runs × 1 task × 1 epoch)

[2025-10-24 17:00:15] VERIFY_TASK: capitalization task
Command: ls /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/experiments/capitalization/cap_task.py
Result: Task script exists and is accessible
Note: Task supports config_dir parameter for experiment integration

[2025-10-24 17:00:20] CREATE_EVAL_DIR: rank8_lr1e-5/eval
Details: mkdir -p rank8_lr1e-5/eval/logs
Result: Directory created successfully

[2025-10-24 17:00:25] GENERATE_SLURM: rank8_lr1e-5/eval/capitalization_epoch0.slurm
Details: Fine-tuned model evaluation with config_dir integration
Model path: /scratch/gpfs/MSALGANIK/niznik/ck-outputs/ck-out-rank8_lr1e-5/epoch_0
Result: SLURM script created (45 lines)

[2025-10-24 17:01:30] COMPLETE: All evaluation configs generated
Summary: 8 evaluation scripts created successfully, 0 failures
Note: Evaluation jobs can be submitted after fine-tuning completes
```

## Output Summary

After completing all evaluation configurations, provide a summary:

```markdown
## Scaffold Inspect Complete

Successfully created 8 evaluation configurations in:
`/scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/`

### Created Evaluations

**Fine-tuned runs (8 evaluations):**
✓ rank8_lr1e-5/eval/capitalization_epoch0.slurm
✓ rank8_lr5e-5/eval/capitalization_epoch0.slurm
✓ rank16_lr1e-5/eval/capitalization_epoch0.slurm
✓ rank16_lr5e-5/eval/capitalization_epoch0.slurm
✓ rank32_lr1e-5/eval/capitalization_epoch0.slurm
✓ rank32_lr5e-5/eval/capitalization_epoch0.slurm
✓ rank64_lr1e-5/eval/capitalization_epoch0.slurm
✓ rank64_lr5e-5/eval/capitalization_epoch0.slurm

Each evaluation directory contains:
- {task_name}_epoch{N}.slurm (SLURM script)
- logs/ (for inspect-ai output)

### Evaluation Tasks

✓ **capitalization**: `/path/to/cap_task.py`
  - Dataset: Reads from fine-tuning config
  - System prompt: Matches training configuration
  - Epochs evaluated: Last epoch only (epoch 0)

### Next Steps

After fine-tuning completes:
1. Evaluation jobs can be submitted (via run-experiment orchestrator or manually)
2. Results will be written to `{run_dir}/eval/logs/` directories
3. Analysis can be performed once evaluations complete

**Manual evaluation submission** (if not using orchestrator, after fine-tuning completes):
```bash
cd /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22
# After fine-tuning completes for a run:
cd rank8_lr1e-5/eval
sbatch capitalization_epoch0.slurm
```

See `scaffold-inspect.log` for detailed creation log.
```

## Validation Before Completion

Before reporting success, verify:
- ✓ All eval directories created
- ✓ Each evaluation has corresponding SLURM script
- ✓ Scripts start with `#!/bin/bash` (no backslash escape)
- ✓ Scripts reference correct model paths
- ✓ Scripts reference correct task scripts
- ✓ System prompts match training configuration
- ✓ Log directory paths are correct
- ✓ Fine-tuned scripts include `--metadata epoch={N}` and `--metadata finetuned=true`
- ✓ Control model scripts include `--metadata finetuned=false` (no epoch)
- ✓ All scripts include `--metadata source_model="{model_name}"`
- ✓ All scripts include `-T vis_label="{label}"`
- ✓ All scripts set USE_CHAT_TEMPLATE
- ✓ No errors in log
- ✓ Log file created

## Important Notes

- This subagent generates evaluation configs for both fine-tuned and control models
- Evaluation scripts should not be submitted until fine-tuning completes
- System prompt consistency between training and evaluation is critical
- Model paths reference fine-tuning output directories that don't exist yet (created during training)
- inspect-ai task scripts must exist before scaffolding (or note as prerequisite)
- Control model evaluations use original model paths, not fine-tuned checkpoints
- This subagent is typically called by `scaffold-experiment` orchestrator but can be run standalone
- Evaluation logs will be written to `{run_dir}/eval/logs/` subdirectories
- **Metadata flags (`--metadata`) are critical for inspect-viz** - stored in `log.eval.metadata` for filtering/grouping
- `vis_label` defaults to run name if not specified in matrix
- `source_model` should be the human-readable model name (e.g., "Llama-3.2-1B-Instruct"), not the path

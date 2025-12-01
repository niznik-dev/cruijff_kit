# Torchtune Scaffolding Workflow

This document describes the detailed step-by-step process for scaffolding torchtune fine-tuning configurations.

## Prerequisites

- experiment_summary.md exists
- claude.local.md exists with environment settings
- Fine-tuning template exists (setup_finetune_json.yaml or setup_finetune_parquet.yaml)

## Step-by-Step Process

### 1. Parse Configuration

**Read experiment_summary.md:**
- Extract experiment name from title
- Extract experiment directory path
- Parse "All Runs" table for fine-tuning parameters
- Extract model path from Resources → Models
- Extract dataset path from Resources → Dataset
- Extract common configuration (epochs, GPUs, batch size, system prompt)

**Read claude.local.md:**
- Extract conda_env
- Extract output_dir_base
- Extract my_wandb_project
- Extract scratch_dir
- Extract account (optional, for SLURM)

### 2. Analyze Varying Parameters

**Goal:** Determine which parameters vary across runs to generate concise directory names

**Process:**
```python
# For each parameter in the runs table
parameters = ['lora_rank', 'learning_rate', 'batch_size', 'model']
varying_params = []

for param in parameters:
    values = [run[param] for run in runs]
    if len(set(values)) > 1:  # More than one unique value
        varying_params.append(param)

# Build directory naming pattern
# Example: varying_params = ['lora_rank', 'learning_rate']
# Pattern: rank{N}_lr{LR}
```

### 3. Create Run Directories

**For each fine-tuning run:**

```bash
# Build directory name from varying parameters
# Example: r8_lr1e-5/
mkdir -p {experiment_dir}/{run_directory_name}
```

### 4. Select Template

**Determine template based on dataset format:**

```python
dataset_path = experiment_summary['dataset_path']
if dataset_path.endswith('.json'):
    template = 'experiments/capitalization/templates/finetuning/setup_finetune_json.yaml'
elif dataset_path.endswith('.parquet'):
    template = 'experiments/capitalization/templates/finetuning/setup_finetune_parquet.yaml'
else:
    raise ValueError(f"Unknown dataset format: {dataset_path}")
```

### 5. Generate setup_finetune.yaml

**For each run directory:**

See [templates/setup_finetune_template.yaml](../templates/setup_finetune_template.yaml) for complete structure.

**Populate with run-specific values:**
- Run identification: `my_wandb_project`, `my_wandb_run_name` (directory name)
- Dataset: `dataset_label`, `dataset_ext`, `input_dir_base` (from experiment_summary.md)
- Model: `torchtune_model_name` (from experiment_summary.md, e.g., "Llama-3.2-1B-Instruct")
- Hyperparameters: `lora_rank`, `lr`, `batch_size`, `epochs` (from run table/common config)
- Environment: `output_dir_base`, `conda_env` (from claude.local.md)
- Optional: `account` (only if specified in claude.local.md SLURM Defaults)

### 6. Execute setup_finetune.py

**CRITICAL: Always use setup_finetune.py**

- **DO:** Generate configs fresh using setup_finetune.py
- **DON'T:** Copy finetune.yaml from previous experiments
- **DON'T:** Manually create or edit finetune.yaml files
- **WHY:** Ensures configs match current experiment parameters and prevents subtle bugs from stale values

**CRITICAL: Activate conda environment first**

```bash
module load anaconda3/2025.6
conda activate {conda_env}
```

**For each run directory:**

```bash
cd {experiment_dir}/{run_directory_name}
python /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/tools/torchtune/setup_finetune.py
```

**Expected outputs:**
- `finetune.yaml` - torchtune configuration
- `finetune.slurm` - SLURM submission script

**Batch execution alternative:**

```bash
module load anaconda3/2025.6
conda activate cruijff
cd {experiment_dir}
for dir in rank*/; do
  echo "Processing $dir..."
  (cd "$dir" && python /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/tools/torchtune/setup_finetune.py)
  if [ $? -eq 0 ]; then
    echo "✓ $dir complete"
  else
    echo "✗ $dir failed"
  fi
done
```

### 7. Validate Parameter Correctness

**CRITICAL: Verify generated configs have correct parameters**

**Validation script:**

```bash
validation_passed=true

for dir in rank*/; do
  dir_clean=${dir%/}

  # Extract expected values from directory name
  expected_rank=$(echo $dir_clean | grep -oP 'rank\K\d+')
  expected_lr=$(echo $dir_clean | grep -oP 'lr\K[^_]+$')

  # Extract actual values from finetune.yaml
  actual_rank=$(grep "lora_rank:" "$dir_clean/finetune.yaml" | awk '{print $2}')
  actual_lr=$(grep "^  lr:" "$dir_clean/finetune.yaml" | awk '{print $2}')  # Note: two-space indent

  # Compare
  if [ "$expected_rank" = "$actual_rank" ] && [ "$expected_lr" = "$actual_lr" ]; then
    echo "✓ $dir_clean parameters match (rank=$actual_rank, lr=$actual_lr)"
  else
    echo "✗ $dir_clean MISMATCH: expected rank=$expected_rank lr=$expected_lr, got rank=$actual_rank lr=$actual_lr"
    validation_passed=false
  fi
done

if [ "$validation_passed" = true ]; then
  echo "✓ All parameters validated successfully"
else
  echo "✗ Validation failed - some runs have incorrect parameters"
  exit 1
fi
```

**What this catches:**
- setup_finetune.py not properly substituting parameters
- Template errors
- Configuration bugs that would cause training with wrong hyperparameters

### 8. Log Results

**Create scaffold.log entry:**

```
[YYYY-MM-DD HH:MM:SS] SCAFFOLD_TORCHTUNE_START
Details: Generating fine-tuning configs for {N} runs
Result: Analyzing experiment_summary.md

[YYYY-MM-DD HH:MM:SS] ANALYZE_PARAMETERS
Details: Varying parameters: {list}
Result: Directory naming pattern: {pattern}

[YYYY-MM-DD HH:MM:SS] CREATE_RUN_DIRS
Details: {list of directories}
Result: All directories created successfully

[YYYY-MM-DD HH:MM:SS] GENERATE_YAMLS
Details: Using template {template_path}
Result: {N} setup_finetune.yaml files created

[YYYY-MM-DD HH:MM:SS] RUN_SETUP_SCRIPTS
Details: Batch execution with conda environment activated
Result: {N} runs processed, {success_count} successful, {failure_count} failures

[YYYY-MM-DD HH:MM:SS] VALIDATE_PARAMETERS
Details: Comparing finetune.yaml parameters against directory names
Result: {validation_result}

[YYYY-MM-DD HH:MM:SS] SCAFFOLD_TORCHTUNE_COMPLETE
Details: {summary}
Duration: {elapsed_time}
```

## Error Handling

**If experiment_summary.md is missing required fields:**
- Report which fields are missing
- Ask user to complete experiment_summary.md
- Do not proceed

**If template file not found:**
- Report expected template path
- Check dataset format is supported
- Suggest creating template if needed

**If setup_finetune.py fails for a run:**
- Log the error for that specific run
- Continue with remaining runs
- Report all failures in final summary

**If parameter validation fails:**
- Report which runs have mismatches
- Do NOT mark scaffolding as successful
- Suggest checking setup_finetune.py and template

**If conda environment not activated:**
- Will see `ModuleNotFoundError: No module named 'cruijff_kit'`
- Remind user to activate conda environment
- Re-run with proper environment

## Success Criteria

- ✓ All run directories created
- ✓ All setup_finetune.yaml files generated
- ✓ All finetune.yaml and finetune.slurm files generated
- ✓ **Parameter validation passed** (critical!)
- ✓ scaffold.log contains complete process details

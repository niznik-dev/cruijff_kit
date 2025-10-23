# Scaffold Experiment

You help users automatically set up the directory structure, configuration files, and SLURM scripts for all runs in a designed experiment.

## Your Task

Read an `experiment_summary.md` file created by the `design-experiment` skill and generate all the necessary files and directories so that runs are ready to submit to SLURM.

## Workflow

1. **Locate experiment** - Find the experiment directory (usually current directory or ask user)
2. **Read experiment_summary.md** - Parse the experiment plan to extract run configurations
3. **Read claude.local.md** - Get environment-specific settings (conda env, output dirs, etc.)
4. **Identify varying parameters** - Determine which parameters change across runs (for directory naming)
5. **For each run:**
   - Create run directory with descriptive name
   - Generate `setup_finetune.yaml` from appropriate template
   - Execute `setup_finetune.py` to generate `finetune.yaml` and `finetune.slurm`
   - Verify outputs and report status
6. **Create scaffold log** - Document all actions taken
7. **Report summary** - Show user what was created and any issues

## Finding the Experiment

**If user runs skill without arguments:**
- Check if current directory contains `experiment_summary.md`
- If not, ask user for the experiment directory path

**If user provides a path:**
- Use that path as the experiment directory

## Parsing experiment_summary.md

Extract the following information:

### Required Information

1. **Experiment name** - From the title (line 1)
2. **Experiment directory** - From Quick Reference → Paths → Experiment
3. **All runs table** - Extract run names and their parameters
4. **Model path** - From Resources → Models
5. **Dataset path** - From Resources → Dataset
6. **Common configuration:**
   - Epochs (from Configuration section)
   - GPUs (from Configuration section)
   - Batch size (from Configuration section or run table)
   - LoRA ranks (from run table)
   - Learning rates (from run table if present)
   - System prompt (from Configuration section)
   - Validation during training (from Configuration section)

### Parsing the "All Runs" Table

The table format looks like:
```markdown
| Run Name | Model | LoRA Rank | Learning Rate | Batch Size | Type | Est. Time |
|----------|-------|-----------|---------------|------------|------|-----------|
| Llama-3.2-1B-Instruct_rank8_lr1e-5 | Llama-3.2-1B-Instruct | 8 | 1e-5 | 4 | Fine-tuned | ~10s |
| Llama-3.2-1B-Instruct_base | Llama-3.2-1B-Instruct | - | - | - | Control | N/A |
```

**Important:**
- Only process runs where `Type` = "Fine-tuned" (skip control runs)
- Extract parameters from table columns
- Parameters with `-` are not applicable (like control runs)

## Determining Directory Names

**Goal:** Directory names should only include parameters that vary across runs.

**Algorithm:**
1. For each parameter in the table (LoRA Rank, Learning Rate, Batch Size, etc.), check if it has different values across runs
2. Include only varying parameters in directory names
3. Use a consistent naming pattern: `{param1}{value1}_{param2}{value2}`

**Examples:**

Experiment varying LoRA rank and learning rate:
- `rank8_lr1e-5/`
- `rank16_lr5e-5/`
- `rank32_lr1e-5/`

Experiment varying only LoRA rank:
- `rank8/`
- `rank16/`
- `rank32/`

Experiment varying model and LoRA rank:
- `Llama-3.2-1B_rank8/`
- `Llama-3.2-3B_rank16/`

**Parameter name abbreviations:**
- LoRA Rank → `rank`
- Learning Rate → `lr`
- Batch Size → `bs`
- Model → use short model name (e.g., `Llama-3.2-1B`)

## Reading claude.local.md

Extract environment-specific settings:
- `conda_env` - Which conda environment to use
- `output_dir_base` - Where to write model checkpoints
- `my_wandb_project` - WandB project name
- `scratch_dir` - User's scratch directory
- `account` - SLURM account to use (under "SLURM Defaults" section) - **OPTIONAL**: only needed if user has multiple accounts and cluster requires explicit specification. If not found in claude.local.md, skip this field.

## Generating setup_finetune.yaml

For each run, create a `setup_finetune.yaml` file by:

1. **Select appropriate template** based on dataset format:
   - Check dataset path extension in experiment_summary.md
   - If `.json` → use `tasks/capitalization/templates/finetuning/setup_finetune_json.yaml`
   - If `.parquet` → use `tasks/capitalization/templates/finetuning/setup_finetune_parquet.yaml`

2. **Extract dataset information from experiment_summary.md:**
   - Dataset path from Resources → Dataset → Path
   - Extract `dataset_label` from filename (e.g., `words_4L_80P_300.json` → `words_4L_80P_300`)
   - Extract `dataset_ext` from filename (e.g., `.json` or `.parquet`)
   - Dataset location: use `input_dir_base` from template, set `input_formatting: ''` (empty string)

3. **Populate template with run-specific values:**

```yaml
# Run identification
my_wandb_project: {from claude.local.md, or use experiment-level project name}
my_wandb_run_name: {directory_name, e.g., "rank8_lr1e-5"}

# Directory Configuration (for dataset path construction)
input_dir_base: {parent directory of dataset from experiment_summary}
input_formatting: ''  # Usually empty string

# Dataset Configuration
dataset_label: {extracted from dataset filename, e.g., "words_4L_80P_300"}
dataset_ext: {extracted from dataset filename, e.g., ".json"}

# Model Configuration
model_checkpoint: {from experiment_summary.md Resources → Models}

# Hyperparameters (run-specific)
lora_rank: {from run table}
learning_rate: {from run table, format as 1e-5 or 5e-5}
batch_size: {from run table or common config}

# Training configuration (common across runs)
epochs: {from experiment_summary.md Configuration}
log_every_n_steps: {use template default, typically 1}
run_val_every_n_steps: {use template default, typically 0}

# Checkpoint Options
stash_adapter_weights: 'true'  # From template default

# Output configuration
output_dir_base: {from claude.local.md}
conda_env: {from claude.local.md}

# SLURM configuration (optional - only if specified in claude.local.md)
account: {from claude.local.md SLURM Defaults, if present}

# System prompt (if specified)
system_prompt: {from experiment_summary.md Configuration, often empty string ""}

# Custom Recipe
custom_recipe: {from template, e.g., cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_v1}
```

4. **Write file** to `{experiment_dir}/{run_directory_name}/setup_finetune.yaml`

**Important notes:**
- Use absolute paths for robustness (e.g., `/scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/...`) rather than relative paths
- WandB project: Prefer using `my_wandb_project` from `claude.local.md` for consistency
- Learning rate format: Keep scientific notation format from experiment summary (1e-5, 5e-5, etc.)

## Running setup_finetune.py

For each run directory:

1. **Activate conda environment (CRITICAL):**
   ```bash
   module load anaconda3/2025.6
   conda activate cruijff
   ```
   **Important:** The script will fail with `ModuleNotFoundError: No module named 'cruijff_kit'` if the conda environment is not activated first.

2. **Navigate to run directory:**
   ```bash
   cd {experiment_dir}/{run_directory_name}
   ```

3. **Execute setup script:**
   ```bash
   python /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/tools/torchtune/setup_finetune.py
   ```
   **Note:** Use absolute path to setup_finetune.py for robustness. Adjust based on actual cruijff_kit location.

4. **Verify outputs exist:**
   - `finetune.yaml` should be created
   - `finetune.slurm` should be created

5. **Capture any errors** and report them

**Example batch execution for all runs:**
```bash
module load anaconda3/2025.6
conda activate cruijff
cd {experiment_dir}
for dir in rank*/; do
  echo "Processing $dir..."
  (cd "$dir" && python /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/tools/torchtune/setup_finetune.py)
  echo "✓ $dir complete"
done
```

## Path Resolution

**Finding cruijff_kit from experiment directory:**
- Experiment is typically in `{scratch_dir}/{experiment_name}/`
- cruijff_kit is typically in `{scratch_dir}/GitHub/cruijff_kit/`
- **Recommended:** Use absolute path `/scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/` for robustness
- Relative path alternative: `../GitHub/cruijff_kit/` (less reliable)

**If path doesn't work:**
- Ask user where cruijff_kit is located
- Always prefer absolute paths over relative paths

## Error Handling

**If experiment_summary.md not found:**
- Ask user for experiment directory path
- Verify file exists before proceeding

**If required information missing from experiment_summary.md:**
- Report specific missing fields
- Ask user to provide missing information
- Do not proceed with incomplete data

**If template not found:**
- Report which template was expected
- Ask user to verify task and dataset format
- Suggest checking template path

**If setup_finetune.py fails for a run:**
- Log the error
- Continue with remaining runs
- Report all failures at the end

**If model or dataset paths don't exist:**
- Warn user but proceed (paths might be correct on compute nodes)
- Note in log which paths couldn't be verified

## Logging

Create a detailed log file at `{experiment_dir}/scaffold-experiment.log`:

### Log Format

```
[YYYY-MM-DD HH:MM:SS] ACTION: Description
Details: {specifics}
Result: {outcome}

```

### What to Log

- Experiment discovery and validation
- Parsing experiment_summary.md
- Parameter analysis (which parameters vary)
- Directory creation for each run
- setup_finetune.yaml generation for each run
- setup_finetune.py execution and results
- Any errors or warnings
- Final summary of created runs

### Example Log Entries

```
[2025-10-22 16:30:00] DISCOVER_EXPERIMENT: Found experiment
Details: /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/experiment_summary.md
Result: Successfully read experiment plan (8 fine-tuned runs)

[2025-10-22 16:30:05] ANALYZE_PARAMETERS: Identifying varying parameters
Details: Checked LoRA Rank, Learning Rate, Batch Size, Model
Result: Varying parameters: LoRA Rank (8,16,32,64), Learning Rate (1e-5, 5e-5)
Directory naming pattern: rank{N}_lr{LR}

[2025-10-22 16:30:10] CREATE_RUN_DIR: rank8_lr1e-5
Details: mkdir /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/rank8_lr1e-5
Result: Directory created successfully

[2025-10-22 16:30:12] GENERATE_YAML: rank8_lr1e-5/setup_finetune.yaml
Details: Template: tasks/capitalization/templates/finetuning/setup_finetune_json.yaml
Parameters: rank=8, lr=1e-5, batch_size=4, epochs=1
Result: File created (237 bytes)

[2025-10-22 16:30:15] RUN_SETUP: rank8_lr1e-5
Command: cd /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/rank8_lr1e-5 && python ../../GitHub/cruijff_kit/tools/torchtune/setup_finetune.py
Result: Success - generated finetune.yaml and finetune.slurm

[2025-10-22 16:31:00] COMPLETE: All runs scaffolded
Summary: 8 runs created successfully, 0 failures
Next: User can cd to run directories and submit with `sbatch finetune.slurm`
```

## Output Summary

After completing all runs, provide a summary to the user:

```markdown
## Scaffold Complete

Successfully created 8 fine-tuning runs in:
`/scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/`

### Created Runs

✓ rank8_lr1e-5/
✓ rank8_lr5e-5/
✓ rank16_lr1e-5/
✓ rank16_lr5e-5/
✓ rank32_lr1e-5/
✓ rank32_lr5e-5/
✓ rank64_lr1e-5/
✓ rank64_lr5e-5/

Each run contains:
- setup_finetune.yaml (configuration)
- finetune.yaml (torchtune config)
- finetune.slurm (SLURM script)

### Next Steps

To submit all jobs:
```bash
cd /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22
for dir in rank*/; do (cd "$dir" && sbatch finetune.slurm); done
```

Or submit individually:
```bash
cd rank8_lr1e-5
sbatch finetune.slurm
```

Monitor jobs:
```bash
squeue -u niznik
```

See `scaffold-experiment.log` for detailed creation log.
```

## Validation Before Completion

Before reporting success, verify:
- ✓ All run directories created
- ✓ Each directory has setup_finetune.yaml
- ✓ Each directory has finetune.yaml (generated)
- ✓ Each directory has finetune.slurm (generated)
- ✓ No errors in log
- ✓ Log file created
- ✓ **Parameters in finetune.yaml match directory names** (see verification section below)

### Critical: Verify Parameter Correctness

**IMPORTANT:** After scaffolding, verify that the generated `finetune.yaml` files contain the correct parameter values matching the directory names. This catches bugs where setup_finetune.py might not properly substitute parameters.

#### Parameter Field Mapping

Know where to find parameters in finetune.yaml:
- `lora_rank` → `model.lora_rank` (nested under model section)
- `lr` (learning rate) → `optimizer.lr` (nested under optimizer section, note: two-space indent)
- `batch_size` → `batch_size` (top level)
- `epochs` → `epochs` (top level)
- `my_wandb_run_name` → `my_wandb_run_name` (top level)
- `output_dir` → `output_dir` (top level, should include run name)

#### Verification Script Pattern

For each run directory, extract and compare parameters:

```bash
for dir in rank*/; do
  dir_clean=${dir%/}

  # Extract expected values from directory name
  expected_rank=$(echo $dir_clean | grep -oP 'rank\K\d+')
  expected_lr=$(echo $dir_clean | grep -oP 'lr\K[^_]+$')

  # Extract actual values from finetune.yaml (note the specific grep patterns)
  actual_rank=$(grep "lora_rank:" "$dir_clean/finetune.yaml" | awk '{print $2}')
  actual_lr=$(grep "^  lr:" "$dir_clean/finetune.yaml" | awk '{print $2}')  # Note: two-space indent

  # Compare and report
  if [ "$expected_rank" = "$actual_rank" ] && [ "$expected_lr" = "$actual_lr" ]; then
    echo "✓ $dir_clean parameters match"
  else
    echo "✗ $dir_clean MISMATCH: expected rank=$expected_rank lr=$expected_lr, got rank=$actual_rank lr=$actual_lr"
  fi
done
```

**What to verify:**
1. Parameters varying across runs (e.g., lora_rank, lr) match directory names
2. Common parameters (e.g., batch_size, epochs) match experiment configuration
3. WandB run names and output directories include correct run identifiers

**If mismatches found:**
- Report which runs have incorrect parameters
- Indicate which specific parameters are wrong
- Suggest checking if setup_finetune.py has all necessary arguments
- Do NOT report success - these runs would train with wrong hyperparameters!

## Important Notes

- Only scaffold "Fine-tuned" runs, not "Control" runs (controls don't need fine-tuning)
- Directory names should be concise and descriptive (only varying parameters)
- Use paths from `claude.local.md` for environment-specific settings
- Preserve all configuration from experiment_summary.md
- If a run fails, continue with others and report all failures at end
- Always create a log file for auditing and debugging
- Paths should work whether skill is run from experiment dir or elsewhere

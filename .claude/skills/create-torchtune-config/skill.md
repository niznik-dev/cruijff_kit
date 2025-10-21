# Create Torchtune Configs

You are helping the user create torchtune configuration files for an experiment with multiple fine-tuning runs.

**SCOPE**: This skill generates ONLY torchtune-specific configuration files (`finetune.yaml`). It does NOT create directories.

**WORKFLOW POSITION**: Run this skill AFTER `setup-experiment-dirs` and BEFORE `launch-runs`.

**PREREQUISITES**:
- ✓ `runs_plan.md` exists
- ✓ Run directories exist (created by `setup-experiment-dirs` skill)
- ✓ `evaluations/` subdirectories exist in each run directory

**GENERATES**:
- `finetune.yaml` in each run directory (torchtune config)
- Optionally: `finetune.slurm` via `generate_slurm_scripts.py` call

## Your Task

Read `runs_plan.md` and generate validated `finetune.yaml` configuration files for all runs using the cruijff_kit template.

## Steps

### 1. Verify Prerequisites

**Check that setup-experiment-dirs was run first:**

```bash
cd /scratch/gpfs/MSALGANIK/mjs3/{experiment_name}

# Verify runs_plan.md exists
if [ ! -f "runs_plan.md" ]; then
  echo "❌ ERROR: runs_plan.md not found"
  echo "Run the plan-runs skill first"
  exit 1
fi

# Count run directories
run_dirs=$(find . -maxdepth 1 -type d -name "Llama*" | wc -l)
if [ $run_dirs -eq 0 ]; then
  echo "❌ ERROR: No run directories found"
  echo "Run the setup-experiment-dirs skill first"
  exit 1
fi

echo "✓ Found $run_dirs run directories"
```

### 2. Read runs_plan.md

Extract key information from the plan:

- **Run specifications**: From "All Runs" table
  - Run names
  - Model sizes (1B, 3B, etc.)
  - LoRA ranks (4, 64, etc.)
  - Training datasets
  - Batch sizes
  - Estimated times

- **Configuration details**: From "Configuration" section
  - Recipe to use
  - Number of epochs
  - Dataset packing (True/False)
  - System prompt
  - Validation settings

- **Resource paths**: From "Resources" section
  - Model directories
  - Dataset locations
  - Input directories

**Example parsing:**
```bash
# Extract model directory from plan
grep "Models:" runs_plan.md
# Output: - **Location**: `/scratch/gpfs/MSALGANIK/pretrained-llms/`

# Extract recipe from plan
grep "Recipe:" runs_plan.md
# Output: - **Recipe**: `cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_val`

# Count runs in plan
grep -c "^| Llama" runs_plan.md
# Output: 12  (plus 2 base models = 14 total)
```

### 3. Use cruijff_kit template as base

Start from the proven cruijff_kit template:
- **Template location**: `tools/torchtune/templates/finetune_template.yaml`
- **Custom recipes**: Use recipes from `tools/torchtune/custom_recipes/`
  - `lora_finetune_single_device_val.py` - Single GPU with validation support (most common)
  - `lora_finetune_single_device_v1.py` - Single GPU without validation
  - `lora_finetune_distributed_v1.py` - Multi-GPU distributed

**Do NOT download configs from the internet** - the cruijff_kit template has been tested and works with the custom recipes.

**Read the template:**
```bash
cat /home/mjs3/cruijff_kit/tools/torchtune/templates/finetune_template.yaml
```

### 4. Detect Checkpoint Files (CRITICAL)

**IMPORTANT**: Before generating configs, use the model registry and utilities to detect checkpoint file structure.

cruijff_kit provides two tools to help:

**A. Model Registry** (`tools/torchtune/model_registry.yaml`):
- Documents checkpoint structure for common models
- Includes Llama 3.2 (1B, 3B), Llama 3.1 (8B), Llama 2 (7B)
- Auto-updated when new models are added

**B. Model Utilities** (`tools/torchtune/model_utils.py`):
- Automatically detects checkpoint files (with or without registry)
- Validates model directories
- Provides all config values needed

**Recommended usage**:
```python
from tools.torchtune.model_utils import validate_model_directory

# Validate and get all config values for a model
model_dir = Path("/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-3B-Instruct")
model_name = "Llama-3.2-3B-Instruct"

config = validate_model_directory(model_dir, model_name, verbose=True)

# Use in your finetune.yaml:
# checkpoint_files: config["checkpoint_files"]
# tokenizer.path: config["tokenizer_path"]
# model_type: config["model_type"]
# model._component_: config["torchtune_component"]
```

**CLI usage**:
```bash
# Test a model directory
python tools/torchtune/model_utils.py \
    /scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-3B-Instruct \
    Llama-3.2-3B-Instruct

# Output:
#   ✓ Found checkpoint files: ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']
#   ✓ Found tokenizer: original/tokenizer.model
#   ✓ Model type: LLAMA3_2
#   ✓ Torchtune component: torchtune.models.llama3_2.lora_llama3_2_3b
```

**Fallback detection** (if model not in registry):
The utilities will automatically detect:
- Single file: `model.safetensors`
- Sharded files: `model-*-of-*.safetensors` pattern

### 5. Generate configs for each run

**For each run directory**, create a customized `finetune.yaml` file.

**SKIP base model runs** - they don't need fine-tuning configs (only evaluation configs).

For each run directory, create a `finetune.yaml` file with run-specific values:

**Variables to customize:**
- `my_wandb_run_name`: Use the run directory name
- `dataset_label`: Dataset filename (without extension)
- `output_dir`: Full path to run directory
- `input_dir`: Path to dataset location
- `models_dir`: Path from claude.local.md
- `batch_size`: Based on model size AND packing setting
  - If packed=False: 1B: 16, 3B: 8 (can be larger)
  - If packed=True: 1B: 4-8, 3B: 2-4 (must be smaller)
  - Verify against runs_plan.md estimates
- `packed`: True (efficient) or False (safer, simpler)
  - Check runs_plan.md for packing density estimates
  - If no prior packing data: Use packed=True with conservative batch_size
- `model._component_`: e.g., `torchtune.models.llama3_2.lora_llama3_2_1b` or `lora_llama3_2_3b`
- `lora_rank`: LoRA rank parameter
- `lora_alpha`: Auto-set to 2 × rank
- `checkpointer.checkpoint_dir`: Model directory path
- `tokenizer.path`: Tokenizer path within model directory

### 5. Critical Checks

**Model name consistency:** Double-check all lines that reference the model name. The mapping between model name and config values is not always consistent.

**Example issue:** Some Llama models require `checkpointer.model_type: LLAMA3` even when the model is version 3.X.

**How to verify:** Check the official configs at https://github.com/meta-pytorch/torchtune/tree/main/recipes/configs for the correct values.

**Dataset field vs split parameter:** This is a common source of errors!
- Use `field` when loading from **local JSON files** with nested structure like `{"train": [...], "validation": [...], "test": [...]}`
- Use `split` when loading from **HuggingFace datasets** that have named splits

**Example - Local JSON file:**
```yaml
dataset:
  _component_: torchtune.datasets.instruct_dataset
  source: "json"
  data_files: "${input_dir}/${dataset_label}.json"
  field: "train"  # NOT split!
  packed: False
```

**Example - HuggingFace dataset:**
```yaml
dataset:
  _component_: torchtune.datasets.instruct_dataset
  source: "username/dataset-name"
  split: "train"  # correct for HF datasets
  packed: False
```

If you use `split` with a local JSON file, you'll get: `JSON parse error: Missing a name for object member.`

### 6. Packing and Batch Size Validation

**CRITICAL**: If `packed: True`, batch sizes must be reduced compared to unpacked configs!

**Validation steps**:

1. **Check packing setting** in template or config:
   ```yaml
   dataset:
     packed: True  # or False
   ```

2. **If packed=True, verify batch size is appropriate:**

   **Safe defaults for packed=True**:
   - 1B models: batch_size ≤ 8
   - 3B models: batch_size ≤ 4
   - 7B+ models: batch_size ≤ 2

   **If batch_size is higher**, check for prior run validation:
   - Has a similar run completed successfully with this batch size + packed=True?
   - If no: REDUCE batch size to safe defaults
   - If yes: Document the successful config in validation notes

3. **Cross-check with runs_plan.md**:
   - If runs_plan.md specifies packing density, use those calculations
   - If runs_plan.md batch sizes are from unpacked runs, WARN and recalculate

**Example validation**:
```bash
# Check config for packing
grep "packed:" finetune.yaml

# If packed: True and batch_size > 8 for 1B model:
echo "WARNING: batch_size=16 with packed=True may cause OOM"
echo "Recommended: Reduce to batch_size=4-8"
```

**Add to validation command**:
```bash
# Validate AND check packing/batch_size consistency
source ~/.bashrc && conda activate <env> && \
  cd /path/to/run1 && \
  echo "Checking packing/batch_size..." && \
  grep -E "packed:|batch_size:" finetune.yaml && \
  tune validate finetune.yaml
```

### 7. Validate all configs

**Two validation methods:**

**A. Quick validation** (recommended first):
```bash
# Validate all runs at once
python tools/torchtune/validate_config.py /path/to/experiment_dir/ --all

# Or validate single run
python tools/torchtune/validate_config.py /path/to/run1/
```

This checks:
- ✓ Config exists and parses
- ✓ Checkpoint files exist
- ✓ Dataset files exist
- ✓ Tokenizer exists

**B. Full torchtune validation**:
```bash
# Activate conda environment first
source ~/.bashrc && conda activate <env_from_claude.local.md>

# Validate each run
cd /path/to/run1 && tune validate finetune.yaml
cd /path/to/run2 && tune validate finetune.yaml
...
```

**What to check:**
- All configs report "Config is well-formed!"
- No errors about missing parameters
- No errors about invalid component paths
- Warnings about torchao are okay (informational only)

Fix any errors before proceeding. Common issues:
- Wrong model_type value
- Using `split` instead of `field` for local JSON files
- Missing or incorrect file paths
- Component path typos

## Output Summary

After completing this skill, each run directory will have:

```
Llama-3.2-1B-Instruct_5L_rank4/
├── finetune.yaml          ✓ (created by this skill)
└── evaluations/           ✓ (created by setup-experiment-dirs)
```

**Base model directories are skipped** (no fine-tuning needed):
```
Llama-3.2-1B-Instruct_base/
└── evaluations/           ✓ (only evaluation configs needed later)
```

## Summary Report

Show a summary after completion:

```
=== Torchtune Config Generation Complete ===

Experiment: cap_cross_eval_5_9_13L_2025-10-20
Location: /scratch/gpfs/MSALGANIK/mjs3/cap_cross_eval_5_9_13L_2025-10-20

Generated:
  ✓ 12 finetune.yaml files (all validated)
  ⊘ 2 base model runs skipped (no fine-tuning needed)

Validation results:
  ✓ All configs passed tune validate
  ⚠️  2 torchao warnings (safe to ignore)
  ✓ Packing/batch_size checks passed

Status: Ready for SLURM script generation

Next steps:
  1. Generate SLURM scripts: Use generate-slurm-script skill
  2. Or review configs: ls */finetune.yaml
  3. After scripts ready: Use launch-runs skill

Files created:
  - finetune.yaml: 12 runs ✓
  - finetune.slurm: Not yet created (use generate-slurm-script skill)
```

## Next Steps

**Suggest to the user:**

"I've generated and validated torchtune configs for all 12 fine-tuning runs.

**Next step - Generate SLURM scripts:**
Would you like me to generate the SLURM batch scripts using the `generate-slurm-script` skill?

**Or review first:**
You can review the generated configs:
- Check any run: `cat Llama-3.2-1B-Instruct_5L_rank4/finetune.yaml`
- Validate manually: `cd Llama-3.2-1B-Instruct_5L_rank4 && tune validate finetune.yaml`

**Current status:**
- All run directories exist ✓ (from setup-experiment-dirs)
- All torchtune configs generated ✓ (this skill)
- SLURM scripts: Not yet created (generate-slurm-script skill next)
- Job submission: After SLURM scripts (launch-runs skill)"

# Create Torchtune Configs

You are helping the user create torchtune configuration files for an experiment with multiple fine-tuning runs.

**This skill is an alternative to using setup_finetune.py** - it gives users direct control over torchtune configs without cruijff_kit automation.

## Your Task

Generate and validate a `finetune.yaml` config file for each run based on experiment specifications.

## Steps

### 1. Learn about the experiment

Read the experiment directory to understand:
- Number of runs and their specifications (check `runs_plan.md` if it exists)
- Model sizes (1B, 3B, etc.)
- LoRA ranks
- Datasets and formats
- Batch sizes and training parameters

### 2. Use cruijff_kit template as base

Start from the proven cruijff_kit template:
- **Template location**: `tools/torchtune/templates/finetune_template.yaml`
- **Custom recipes**: Use recipes from `tools/torchtune/custom_recipes/`
  - `lora_finetune_single_device_val.py` - Single GPU with validation support (most common)
  - `lora_finetune_single_device_v1.py` - Single GPU without validation
  - `lora_finetune_distributed_v1.py` - Multi-GPU distributed

**Do NOT download configs from the internet** - the cruijff_kit template has been tested and works with the custom recipes.

### 3. Customize configs for each run

For each run directory, create a `finetune.yaml` file with run-specific values:

**Variables to customize:**
- `my_wandb_run_name`: Use the run directory name
- `dataset_label`: Dataset filename (without extension)
- `output_dir`: Full path to run directory
- `input_dir`: Path to dataset location
- `models_dir`: Path from claude.local.md
- `batch_size`: Based on model size (1B: 16, 3B: 8, adjust as needed)
- `model._component_`: e.g., `torchtune.models.llama3_2.lora_llama3_2_1b` or `lora_llama3_2_3b`
- `lora_rank`: LoRA rank parameter
- `lora_alpha`: Auto-set to 2 × rank
- `checkpointer.checkpoint_dir`: Model directory path
- `tokenizer.path`: Tokenizer path within model directory

### 4. Critical Checks

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

### 5. Validate all configs

Before finalizing, validate ALL config files using `tune validate` within the conda environment from claude.local.md.

**Efficient batch validation:**
```bash
source ~/.bashrc && conda activate <env_from_claude.local.md> && \
  cd /path/to/run1 && tune validate finetune.yaml && \
  cd /path/to/run2 && tune validate finetune.yaml && \
  cd /path/to/run3 && tune validate finetune.yaml
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

## Output

Create one validated `finetune.yaml` config file in each run subdirectory.

## Next Steps

After creating and validating all configs:
1. Use the **launch-runs** skill to submit fine-tuning jobs to SLURM
2. Or use the **generate-slurm-script** skill to create SLURM scripts manually

Note: This skill does NOT create SLURM scripts - it only creates torchtune configs.

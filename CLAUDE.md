# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: If a `claude.local.md` file exists in the repository root, read it first. It contains personal environment-specific settings (HPC usernames, scratch directories, SLURM defaults, conda environments, cluster-specific commands, etc.) that override or supplement the general guidance in this file.

**All paths, environment names, and cluster-specific details in this file are examples only.** Users should configure their actual environment in `claude.local.md`. If it does not exist you should ask the user to create it using the [create-local-config](.claude/skills/create-local-config/skill.md) skill.

## Project Overview

cruijff_kit is a toolkit for research with social data and LLMs. The two core workflows are:
1. **Fine-tuning LLMs** using torchtune with LoRA adapters
2. **Evaluating LLMs** using inspect-ai

The project emphasizes automation through YAML configuration files and SLURM script generation for HPC cluster environments. All runs should be documented and computationally reproducible.

## Quick Start with Skills

For complete workflows, use the skills in [.claude/skills/](.claude/skills/):

### Primary Workflows
- **[plan-runs](.claude/skills/plan-runs/skill.md)**: Plan a series of fine-tuning experiments (design matrix, resource estimation, compute requirements)
- **[setup-experiment-dirs](.claude/skills/setup-experiment-dirs/skill.md)**: Create organized directory structures for experiments
- **[launch-runs](.claude/skills/launch-runs/skill.md)**: Submit fine-tuning or evaluation jobs to SLURM
- **[monitor-jobs](.claude/skills/monitor-jobs/skill.md)**: Track progress of SLURM jobs
- **[summarize-experiments](.claude/skills/summarize-experiments/skill.md)**: Analyze and compare experimental results

### Supporting Workflows
- **[create-evaluation](.claude/skills/create-evaluation/skill.md)**: Create standalone inspect-ai evaluation scripts
- **[create-torchtune-config](.claude/skills/create-torchtune-config/skill.md)**: Generate torchtune configuration files
- **[generate-slurm-script](.claude/skills/generate-slurm-script/skill.md)**: Create SLURM batch scripts
- **[download-model-from-hf](.claude/skills/download-model-from-hf/skill.md)**: Download models from HuggingFace
- **[update-run-status](.claude/skills/update-run-status/skill.md)**: Update experiment status tracking
- **[create-local-config](.claude/skills/create-local-config/skill.md)**: Create environment-specific configuration

## Skill Workflow

### Standard Experiment Workflow (Torchtune)

For torchtune fine-tuning experiments, follow this sequence:

```
1. plan-runs
   ↓
   Creates: runs_plan.md, runs_status.yaml
   Output: Complete experiment specification with resource estimates

2. setup-experiment-dirs
   ↓
   Creates: All run directories with evaluations/ subdirectories
   Output: Empty directory structure, README.md
   Scope: Framework-agnostic (no config files)

3. create-torchtune-config
   ↓
   Creates: finetune.yaml in each run directory
   Output: Validated torchtune configuration files
   Scope: Torchtune-specific only

4. generate-slurm-script
   ↓
   Creates: finetune.slurm in each run directory
   Output: SLURM batch scripts with resource allocations
   Uses: claude.local.md for account/partition settings

5. launch-runs
   ↓
   Submits: Fine-tuning jobs to SLURM
   Updates: runs_status.yaml with job IDs
   Scope: Currently fine-tuning only (evaluation support coming)

6. monitor-jobs
   ↓
   Checks: Job status via squeue/sacct
   Updates: runs_status.yaml with completion status

7. [FUTURE] launch-runs (evaluation mode)
   ↓
   Submits: Evaluation jobs after training completes

8. summarize-experiments
   ↓
   Analyzes: Results across all runs
   Output: Summary tables, plots, comparisons
```

### Key Separation of Concerns

**setup-experiment-dirs**:
- Creates ONLY directories (no config files)
- Framework-agnostic
- Always required regardless of fine-tuning framework

**create-torchtune-config**:
- Creates ONLY finetune.yaml files
- Torchtune-specific
- Only needed if using torchtune for fine-tuning

**generate-slurm-script**:
- Creates ONLY finetune.slurm files
- Can be used with any framework
- Reads finetune.yaml to determine resources

**Why this separation?**
- Modularity: Use only what you need
- Framework flexibility: Can swap torchtune for other tools
- Clear dependencies: Each skill has one job

## Architecture Overview

### Core Components

**Fine-tuning Pipeline** ([tools/torchtune/](tools/torchtune/)):
- `setup_finetune.py`: Generates `finetune.yaml` and `finetune.slurm` from `setup_finetune.yaml`
- `templates/`: YAML and SLURM templates
- `custom_recipes/`: Enhanced torchtune recipes with validation support and adapter weight management

**Evaluation Pipeline** ([tools/inspect/](tools/inspect/)):
- `setup_inspect.py`: Generates `inspect.slurm` for model evaluation
- Automatically fixes adapter configs and resolves path variables
- Supports both base model and fine-tuned checkpoint evaluation

**Utilities** ([utils/](utils/)):
- `llm_utils.py`: Comprehensive LLM inference utilities (load models, get logits/embeddings/tokens)
- `run_names.py`: Generate random run names (positive adjective-noun pairs)
- Dataset converters: `convert_json_to_parquet.py`, `convert_arrow_to_parquet.py`

### Dataset Formats

**JSON Instruct** (single file with splits):
```json
{
  "train": [{"input": "...", "output": "..."}],
  "validation": [...],
  "test": [...]
}
```

**JSON Chat** (folder with separate files):
```
folder_name_c/
  train.json
  validation.json
  test.json
```

**Parquet** (folder with separate files):
```
folder_name_parquet/
  train.parquet
  validation.parquet
  test.parquet
```

## Common Patterns

### Fine-tuning Setup
```bash
# From a task directory
python ../../tools/torchtune/setup_finetune.py  # Uses setup_finetune.yaml
sbatch finetune.slurm
```

### Evaluation Setup
```bash
# For fine-tuned checkpoint
python ../../tools/inspect/setup_inspect.py --finetune_epoch_dir /path/to/epoch_0/
sbatch inspect.slurm

# For base model
python ../../tools/inspect/setup_inspect.py \
  --base_model_dir /path/to/base/model/ \
  --finetune_epoch_dir /path/to/finetuned/epoch_0/
sbatch inspect.slurm
```

### View Results
```bash
inspect view  # Or: inspect view --port=<port_number>
```

## Key Conventions

### Configuration
- **LoRA alpha**: Automatically set to 2 × rank by `setup_finetune.py`
- **SLURM environment**: Configure conda env, partitions, accounts in `claude.local.md`
- **Run names**: Auto-generated (e.g., "bright_horizon") or set via `--my_wandb_run_name`
- **Output structure**: `{output_dir_base}/ck-out-{run_name}/epoch_N/`

### Checkpoint Management
- `epochs_to_save`: 'all', 'none', or comma-delimited list (e.g., "0,2,4")
- `save_last_epoch_only`: 'true'/'false'
- `stash_adapter_weights`: 'true'/'false' (moves adapter files to subdirectory)

### Custom Recipes
cruijff_kit uses modified torchtune recipes for:
- Validation loss tracking during training
- Adapter weight stashing
- Custom checkpoint saving logic

Example:
```bash
python setup_finetune.py --custom_recipe cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_val
```

## Project Status

**Pre-Alpha**: Breaking changes may occur without notice. The toolkit is under active development.

**Verified models**: Llama-2-7b-hf, Llama-3.1-8B-Instruct, Llama-3.2-1B-Instruct, Llama-3.3-70B-Instruct

**HPC Environment:**
- Designed for SLURM-based HPC clusters
- Configure your specific cluster settings in `claude.local.md`
- Template files may require customization for your environment

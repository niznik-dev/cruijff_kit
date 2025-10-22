# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Local setup

- If a `claude.local.md` file exists in the repository root, read it first. It contains personal environment-specific settings (HPC usernames, scratch directories, SLURM defaults, conda environments, cluster-specific commands, etc.) that override or supplement the general guidance in this file.

- All paths, environment names, and cluster-specific details in this file are examples only. Users should configure their actual environment in `claude.local.md`. If it does not exist you should ask the user to create it using the [create-local-config](.claude/skills/create-local-config/SKILL.md) skill.

## Project Overview

cruijff_kit is a toolkit for research with social data and LLMs. 

It allows users to design, conduct, and analyze computational experiments.  When these experiments are well-designed the results they produce will answer scientific questions.

cruijff_kit will support the following capabilities with the following tools: 
1. **Fine-tuning LLMs** using torchtune 
2. **Evaluating LLMs** using inspect-ai
3. **Prompt engineering** using DSPy (not yet implemented)
4. **Agentic systems** using DSPy (not yet implemented)

## Project philosophy

When you are working in this project, the following values should guide your decisions:

1) **Scientific** All work you do on this project should emphasize correctness, computational reproducibility, and detailed logging. Each experiment should be logged such that it can be audited by a researcher or Claude. 

2) **Modular** This project will evolve over time, and it should be designed so that individual components can be added or changed over time with minimal impact on other components.  

3) **Practical** Work in this directory is designed to do science not win a programming contest. Please don't over-engineer or do pre-mature optimization. We don't need hundreds of lines of code to save 5 seconds.

4) **Privacy respecting** Lots of the data in this project is about people. All of that data should be treated with care, and some of it should never leave the users' computer system. Tasks should be designed with clear data governance. 

5) **Self improving** Always look for way that we can learn from earlier experiments to design new experiments, improve skills, and improve analysis. The more work we do, the easier things should be because we will have more designs, results, and logs from which to learn.

## Terminology

An experiment is a set of one or more runs.

Examples of runs include:
- fine tuning an LLM with a specific dataset and a specific finetune recipe
- evaluating an LLM (base or fine tuned) with a specific dataset or recipe

Often we will combine these runs together into a pipeline.  Examples include:
- Fine-tune a model, then evaluate it on several different datasets
- Fine-tune a model with a set of different hyperparameters, then evaluate the set of models with the same dataset

Some runs must be complete before other runs can begin.  For example, we must have a fine-tuned model trained before it can be evaluated.

## Quick Start with Skills

For complete workflows, use the skills in [.claude/skills/](.claude/skills/):

### Primary Workflows
- **[create-experiment](.claude/skills/design-experiment/SKILL.md)**: Plan a series of runs that collectively make up an experiment
- **[generate-dirs](.claude/skills/setup-dirs/SKILL.md)**: Create organized directory structures for runs
- **[generate-slurm-scripts](.claude/skills/generate-slurm-script/SKILL.md)**: Generate SLURM batch scripts for runs
- **[submit-jobs](.claude/skills/launch-runs/SKILL.md)**: Submit jobs prepared by earlier skills to SLURM
- **[generate-experiment-results](.claude/skills/summarize-experiments/SKILL.md)**: Analyze and compare experimental results


### Supporting Workflows
- **[generate-torchtune-config](.claude/skills/create-torchtune-config/SKILL.md)**: Generate torchtune configuration files
- **[generate-inspect-script](.claude/skills/create-evaluation/SKILL.md)**: Generate standalone inspect-ai evaluation scripts

- **[create-claude-local-md](.claude/skills/create-claude-local/SKILL.md)**: Create environment-specific configuration

- **[create-conda-env-md](.claude/skills/create-claude-local/SKILL.md)**: Create your conda environment for this project



- **[download-model-from-hf](.claude/skills/download-model-from-hf/SKILL.md)**: Download models from HuggingFace
- **[download-dataset-from-hf](.claude/skills/download-dataset-from-hf/SKILL.md)**: Download datasets from HuggingFace


## Making New Skills

Over time you should look to make new skills to make it easier to use cruijff_kit.  Before suggesting a new skill, please ask yourself:
- should this be a new skill or an improvement to an existing skill?
- how can I maintain a modular structure with the skills?

When you do make a new skill please ensure that:
- it is really needed
- it follows best practices: https://docs.claude.com/en/docs/claude-code/skills
- improvements to skills are kept inside .claude

### Key Separation of Concerns

**create-experiment**:
- Creates ONLY an experimental plan that will be read by later skills
- Framework-agnostic
- It should include all the fine-tuning and evluation that is originally planned

**generates-dirs**:
- Creates ONLY directories (no config files)
- Framework-agnostic
- Always required regardless of fine-tuning framework

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

## Key Conventions

### Configuration
- **LoRA alpha**: Automatically set to 2 × rank by `setup_finetune.py`
- **SLURM environment**: Configure conda env, partitions, accounts in `claude.local.md`
- **Run names**: Auto-generated 
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

**HPC Environment:**
- Designed for SLURM-based HPC clusters
- Configure your specific cluster settings in `claude.local.md`
- Template files may require customization for your environment

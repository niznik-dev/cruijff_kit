# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: If a `claude.local.md` file exists in the repository root, read it first. It contains personal environment-specific settings (HPC usernames, scratch directories, SLURM defaults, etc.) that override or supplement the general guidance in this file.

## Project Overview

cruijff_kit is a toolkit for research with social data and LLMs. The two core workflows are:
1. **Fine-tuning LLMs** using torchtune with LoRA adapters
2. **Evaluating LLMs** using inspect-ai

The project emphasizes automation through YAML configuration files and SLURM script generation for HPC cluster environments.

## Development Environment

**Testing:**
- Test files are in `tests/` directory (bit_sequences, predictable_or_not)
- No formal test runner configured; tests are exploratory

## Architecture

### Directory Structure

**Core Utilities (`utils/`):**
- `llm_utils.py`: Comprehensive LLM inference utilities (load models, tokenize, get logits/embeddings/tokens)
- `run_names.py`: Generate random positive adjective-noun combinations for run names
- `finetune_custom_metrics.py`: Custom metrics for fine-tuning
- `convert_json_to_parquet.py`, `convert_arrow_to_parquet.py`: Dataset format converters
- `check_if_model_is_finetuned.py`: Verify model fine-tuning status

**Torchtune Integration (`tools/torchtune/`):**
- `setup_finetune.py`: Main script to generate `finetune.yaml` and `finetune.slurm` from config
- `custom_recipes/`: Modified torchtune recipes with cruijff_kit enhancements
  - `lora_finetune_single_device_v1.py`: Single GPU fine-tuning
  - `lora_finetune_distributed_v1.py`: Multi-GPU distributed fine-tuning
  - `lora_finetune_single_device_val.py`: Single GPU with validation support
  - `custom_recipe_utils.py`: Utilities like `stash_adapter_files()` to organize adapter weights
- `templates/`: YAML and SLURM templates used by setup scripts
  - `finetune_template.yaml`: Template torchtune config
  - `finetune_template.slurm`: Template SLURM batch script

**Inspect-AI Integration (`tools/inspect/`):**
- `setup_inspect.py`: Generate `inspect.slurm` for model evaluation
  - Reuses SLURM parameters from fine-tuning runs
  - Supports evaluating both base models and fine-tuned checkpoints

**Task Workflows (`tasks/`):**
- `capitalization/`: Test fine-tuning on word capitalization patterns
  - `input/sample_words.py`: Generate training data from word lists
  - `inspect.py`: Inspect-AI task definition for evaluation
  - `templates/`: YAML configs for different data formats (JSON, Parquet)
- `synthetic_twins/`: Twin dataset preprocessing
  - `preproc.py`: Convert CSV files to JSON format for training

**Other Tools (`tools/`):**
- `huygensweeper/generate_sweep.py`: Parameter sweep generation (Weights & Biases integration)

### Key Workflows

#### 1. Fine-Tuning Workflow

The fine-tuning workflow uses a template-based approach:

1. **Configuration**: Create or copy `setup_finetune.yaml` with parameters like:
   - Model paths, dataset paths, output directories
   - Training hyperparameters (epochs, batch size, LoRA rank, learning rate scheduler)
   - SLURM parameters (time, GPUs, partition, conda environment)
   - Dataset format (JSON vs Parquet, instruct vs chat template)

2. **Generation**: Run `setup_finetune.py` which:
   - Loads `setup_finetune.yaml` (supports CLI overrides)
   - Fills in `templates/finetune_template.yaml` → generates `finetune.yaml`
   - Fills in `templates/finetune_template.slurm` → generates `finetune.slurm`
   - Handles special cases (JSON vs Parquet, chat vs instruct datasets, validation splits)
   - Automatically sets LoRA alpha = 2 * rank

3. **Execution**: Submit with `sbatch finetune.slurm`

4. **Output Structure**:
   - Checkpoints saved to `{output_dir_base}/ck-out-{run_name}/epoch_N/`
   - If `stash_adapter_weights: true`, adapter files moved to `adapter_weights/` subdirectory

**Important Template Variables:**
- `SLURM_ONLY`: Parameters excluded from YAML (time, gpus, conda_env, account, partition, constraint)
- Dataset types: `instruct_dataset`, `chat_dataset`, `text_completion_dataset`
- LR schedulers: `get_cosine_schedule_with_warmup`, `get_linear_schedule_with_warmup`, etc.

#### 2. Evaluation Workflow

Uses inspect-ai for model evaluation:

1. **Setup**: Run `setup_inspect.py` with:
   - `--finetune_epoch_dir`: Path to checkpoint epoch (also provides SLURM params)
   - `--base_model_dir`: Optional, evaluate base model using fine-tune's SLURM config

   **What `setup_inspect.py` does:**
   - Creates `inspect.slurm` from the fine-tuning SLURM parameters
   - **Automatically fixes `adapter_config.json`** if missing `base_model_name_or_path`
     - Reads `finetune.yaml` to find base model path
     - Resolves variables like `${models_dir}`
     - Updates adapter config for compatibility with HuggingFace/inspect-ai
   - **Detects eval script**: Uses `eval.py` if present, otherwise falls back to `inspect.py`
   - Configures inspect command with model path and config directory

2. **Task Definition**: Each task should have an evaluation script (`eval.py` or `inspect.py`):
   - **Standalone approach** (recommended): `eval.py` with hard-coded parameters
     - No dependencies on config files
     - Easy to share and reproduce
     - Created by `create-evaluation` skill
   - **Config-based approach**: `inspect.py` that reads from `setup_finetune.yaml`
     - Ensures eval matches training configuration
     - Used in `tasks/capitalization/inspect.py`

   Both define:
   - Dataset loading (test split)
   - Solver chain (system message → prompt template → generate)
   - Scorers (match, includes, etc.)

3. **Execution**:
   - Submit: `sbatch inspect.slurm`
   - View results: `inspect view --port=$(get_free_port)` (on della)

**Important:** The `setup_inspect.py` script automatically fixes common issues:
- Missing `base_model_name_or_path` in adapter configs
- Variable resolution in paths (e.g., `${models_dir}`)
- Path normalization (removes double slashes, trailing slashes)

#### 3. Data Format Handling

The toolkit supports multiple dataset formats:

**JSON Formats:**
- **Instruct Dataset** (single file with splits):
  ```json
  {
    "train": [{"input": "...", "output": "..."}],
    "validation": [...],
    "test": [...]
  }
  ```
- **Chat Dataset** (folder with separate files):
  ```
  folder_name_c/
    train.json
    validation.json
    test.json
  ```
  Each file contains: `[{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]`

**Parquet Format:**
```
folder_name_parquet/
  train.parquet
  validation.parquet
  test.parquet
```

Use conversion utilities: `convert_json_to_parquet.py`, `convert_arrow_to_parquet.py`

### Custom Recipe System

cruijff_kit uses modified torchtune recipes to add features:
- Validation loss tracking during training
- Adapter weight stashing for cleaner directory structure
- Custom checkpoint saving logic (`epochs_to_save`, `save_last_epoch_only`)

**To use a custom recipe:**
```bash
python setup_finetune.py --custom_recipe cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_val
```

The recipe name is injected into the SLURM script's `tune run` command.

## Common Commands

### Fine-Tuning

**Basic fine-tuning run:**
```bash
# From a task directory (e.g., tasks/capitalization/)
python ../../tools/torchtune/setup_finetune.py  # Uses setup_finetune.yaml in current dir
sbatch finetune.slurm
```

**With command-line overrides:**
```bash
python tools/torchtune/setup_finetune.py \
  --my_wandb_project my_project \
  --my_wandb_run_name my_run \
  --input_dir_base /scratch/gpfs/$USER/data/ \
  --dataset_label words_5L_80P_1000 \
  --dataset_ext .json \
  --epochs 3 \
  --batch_size 4 \
  --time 01:00:00 \
  --gpus 1
```

**Multi-GPU training:**
```bash
python setup_finetune.py --gpus 4  # Automatically switches to distributed recipe
```

### Evaluation

**Evaluate fine-tuned model:**
```bash
python ../../tools/inspect/setup_inspect.py \
  --finetune_epoch_dir /path/to/output/ck-out-run_name/epoch_0/
sbatch inspect.slurm
```

**Evaluate base model (using fine-tune's SLURM config):**
```bash
python ../../tools/inspect/setup_inspect.py \
  --base_model_dir /path/to/base/model/ \
  --finetune_epoch_dir /path/to/finetuned/epoch_0/
sbatch inspect.slurm
```

**View results:**
```bash
inspect view  # Or on della: inspect view --port=$(get_free_port)
```

### Model Download

**Download models with torchtune:**
```bash
tune download meta-llama/Llama-3.2-1B-Instruct \
  --output-dir /scratch/gpfs/$USER/models/Llama-3.2-1B-Instruct \
  --hf-token YOUR_HF_TOKEN
```

Common models: `Llama-2-7b-hf`, `Llama-3.1-8B-Instruct`, `Llama-3.2-1B-Instruct`, `Llama-3.3-70B-Instruct`

### Data Preparation

**Capitalization task (instruct format):**
```bash
cd tasks/capitalization/input
wget https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt
python sample_words.py --word-len 5 --num-words 1000
```

**Capitalization task (chat format):**
```bash
python sample_words.py --word-len 5 --num-words 1000 --use-chat-template
```

**Convert JSON to Parquet:**
```bash
python utils/convert_json_to_parquet.py \
  --input_json path/to/data.json \
  --output_dir path/to/output_folder
```

**Twin dataset preprocessing:**
```bash
python tasks/synthetic_twins/preproc.py /path/to/csv/files
```

### Weights & Biases

**Upload run to W&B:**
```bash
wandb sync /path/to/output/folder/logs/wandb/latest-run
```

## Important Conventions

### SLURM Environment
- Default conda environment: `ttenv`
- Default partition/account: Configured per-cluster (use `--account`, `--partition`, `--constraint`)
- Job names automatically set to wandb run name
- Output organized by run name: `{output_dir_base}/ck-out-{run_name}/`

### Run Naming
- Random names auto-generated from positive adjectives + nouns (e.g., "bright_horizon")
- Set explicitly with `--my_wandb_run_name` to override

### LoRA Configuration
- Default rank: 64
- Alpha automatically set to 2 * rank
- Configurable via `--lora_rank`

### Checkpoint Management
- `epochs_to_save`: 'all', 'none', or comma-delimited list (e.g., "0,2,4")
- `save_last_epoch_only`: 'true'/'false'
- `save_adapter_weights_only`: 'true'/'false'
- `stash_adapter_weights`: 'true'/'false' (moves adapter files to subdirectory)

### Dataset Configuration
The `setup_finetune.py` script automatically adjusts torchtune dataset configuration based on:
- `dataset_ext`: '.json' or '.parquet'
- `dataset_type`: 'instruct_dataset', 'chat_dataset', or 'text_completion_dataset'

For JSON + instruct: Uses `data_files` + `field` parameter
For JSON + chat: Uses `data_files` pointing to folder with train/validation.json
For Parquet: Uses `data_dir` parameter

### Validation During Training
- Set `run_val_every_n_steps > 0` to enable validation
- Requires validation split in dataset
- Only supported with custom recipes (e.g., `lora_finetune_single_device_val`)
- If set to 0, validation config is removed from YAML

## LLM Utilities (`utils/llm_utils.py`)

This module provides comprehensive utilities for model inference:

**Loading:**
- `load_model(model_path, tokenizer_path, adapter_path)`: Load base or PEFT models
- Automatically adds padding tokens if missing

**Inference:**
- `get_logits()`: Get next-token logits for prompts
- `get_next_tokens()`: Generate token sequences with model.generate()
- `get_embeddings()`: Extract hidden state embeddings with pooling options

**Utilities:**
- `tokenize_prompts()`: Tokenize with optional chat templates and system prompts
- `pool_hidden_states()`: Pool embeddings (mean, median, first, last, last_non_padding, etc.)
- `save_tensor_with_ids()`, `load_tensor_with_ids()`: HDF5 storage for embeddings

All functions support:
- Batching via DataLoader
- Chat templates vs flat prompts
- System prompts and preprompts
- Automatic device management

## Project Status

**Pre-Alpha**: Breaking changes may occur without notice. The toolkit is under active development.

**Verified Models:**
- Llama-2-7b-hf
- Llama-3.1-8B-Instruct
- Llama-3.2-1B-Instruct (most common)
- Llama-3.3-70B-Instruct

**Cluster Environment:**
- Primary development on Princeton's della cluster
- Adapt SLURM parameters for your HPC environment

# cruijff_kit Architecture

This document describes the structure and organization of the cruijff_kit codebase. It's designed to help both LLM assistants and human developers understand how the project is organized and how different components interact.

## Overview

cruijff_kit is a research toolkit for fine-tuning and evaluating LLMs on social science data. The architecture supports:

1. **Declarative workflow configuration** via YAML files
2. **Automated SLURM script generation** for HPC environments
3. **Flexible dataset formats** (JSON, Parquet, HuggingFace datasets)
4. **Custom torchtune recipes** with validation support
5. **Inspect-ai integration** for model evaluation

## Directory Structure

```
cruijff_kit/
├── tools/              # Core workflow orchestration scripts
│   ├── torchtune/      # Fine-tuning setup and custom recipes
│   ├── inspect/        # Evaluation setup
│   └── huygensweeper/  # Parameter sweep utilities (being retired)
│
├── tasks/              # Real-world research tasks
│   ├── capitalization/ # Generalization test with word capitalization
│   └── synthetic_twins/# Social science twin prediction task
│
├── tests/              # Synthetic validation tests
│   ├── bit_sequences/  # Bit parity tests
│   └── predictable_or_not/  # Stochastic prediction tests
│
├── utils/              # Shared utilities and helpers
│   ├── llm_utils.py    # Model loading and inference utilities
│   ├── run_names.py    # Random name generation for experiments
│   ├── finetune_custom_metrics.py  # Custom metrics for torchtune
│   ├── check_if_model_is_finetuned.py  # Model state inspection
│   └── convert_*.py    # Dataset format conversion utilities
│
├── heterogeneity/      # Group-level analysis tools
│   ├── heterogeneity_eval.py    # Evaluate prediction heterogeneity
│   └── heterogeneity_report.py  # Generate analysis reports
│
├── misc/               # Experimental/legacy code
└── logs/               # Experiment outputs and logs
```

## Core Workflows

### 1. Fine-tuning Workflow

**Entry point:** `tools/torchtune/setup_finetune.py`

**Configuration:** Task-specific `setup_finetune.yaml` files

**Process:**
```
setup_finetune.yaml → setup_finetune.py → finetune.yaml + finetune.slurm
                                              ↓
                                          sbatch finetune.slurm
                                              ↓
                                      torchtune recipe execution
                                              ↓
                                      output model checkpoints
```

**Key files:**
- `tools/torchtune/setup_finetune.py` - Main orchestration script
  - Reads `setup_finetune.yaml` (user-friendly config)
  - Generates `finetune.yaml` (torchtune recipe config)
  - Generates `finetune.slurm` (SLURM batch script)
  - Handles path resolution, validation, defaults

- `tools/torchtune/templates/finetune_template.yaml` - Base template for torchtune configs

- `tools/torchtune/custom_recipes/` - Modified torchtune recipes
  - `lora_finetune_single_device_v1.py` - Single GPU with custom features
  - `lora_finetune_distributed_v1.py` - Multi-GPU distributed training
  - `lora_finetune_single_device_val.py` - With validation loss tracking
  - `custom_recipe_utils.py` - Shared utilities for recipes

**Custom features added to torchtune:**
- Selective epoch saving (`epochs_to_save`)
- Adapter weight management (`stash_adapter_weights`)
- Custom metrics integration via `utils/finetune_custom_metrics.py`
- Validation during training (requires nightly build)

### 2. Evaluation Workflow

**Entry point:** `tools/inspect/setup_inspect.py`

**Process:**
```
Finetuned model checkpoint → setup_inspect.py → inspect.slurm
                                                      ↓
                                                sbatch inspect.slurm
                                                      ↓
                                              task-specific run_inspect.py
                                                      ↓
                                              inspect-ai evaluation
```

**Key files:**
- `tools/inspect/setup_inspect.py` - Generates SLURM script for evaluation
  - Reuses SLURM parameters from `finetune.slurm`
  - Can evaluate base model or finetuned model
  - Points to task-specific `run_inspect.py`

- Task-specific `run_inspect.py` files (e.g., `tasks/capitalization/run_inspect.py`)
  - Define evaluation prompts and scoring
  - Use inspect-ai framework

### 3. Parameter Sweep Workflow (Being Retired)

**Location:** `tools/huygensweeper/`

**Status:** Being deprecated in favor of LLM-assisted parameter selection (see issue #133)

## Tasks vs Tests

### Tasks (`tasks/`)
Real research projects with scientific questions:

- **capitalization**: Tests generalization by training on 5-letter words and evaluating on other lengths
- **synthetic_twins**: Predicts twin characteristics from synthetic social science data

Each task typically includes:
- `README.md` - Task-specific instructions
- `setup_finetune.yaml` - Configuration template
- `templates/` - YAML templates for different dataset formats
- `input/` - Data generation or preprocessing scripts
- `utils/` - Task-specific helper functions
- `run_inspect.py` - Evaluation script

### Tests (`tests/`)
Synthetic validation tests with known ground truth:

- **bit_sequences**: Tests memorization and pattern learning with deterministic bit patterns
- **predictable_or_not**: Stochastic tests to validate no data leakage

## Package Structure

cruijff_kit is installed as an editable package (`pip install -e .`), making utilities importable:

```python
from cruijff_kit.utils import run_names
from cruijff_kit.utils import llm_utils
```

**Packaged modules** (defined in `pyproject.toml`):
- `cruijff_kit.utils` - Shared utilities
- `cruijff_kit.tools.torchtune.custom_recipes` - Custom torchtune recipes

**Not packaged but executed as scripts:**
- `tools/torchtune/setup_finetune.py`
- `tools/inspect/setup_inspect.py`
- Task-specific scripts

## Data Flow

### Input Data Formats

1. **Instruct Dataset (JSON)**
   ```json
   {
     "train": [{"input": "...", "output": "..."}],
     "validation": [...],
     "test": [...]
   }
   ```

2. **Chat Dataset (JSON files per split)**
   ```
   dataset_folder_c/
   ├── train.json
   ├── validation.json
   └── test.json
   ```

3. **Parquet Format**
   ```
   dataset_folder/
   ├── train.parquet
   ├── validation.parquet
   └── test.parquet
   ```

Conversion utilities: `utils/convert_json_to_parquet.py`, `utils/convert_arrow_to_parquet.py`

### Output Structure

After running setup_finetune.py:
```
task_directory/
├── setup_finetune.yaml    # User config
├── finetune.yaml          # Generated torchtune config
└── finetune.slurm         # Generated SLURM script
```

After running finetune:
```
output_dir/
├── epoch_0/
│   ├── adapter_model.safetensors  # LoRA weights
│   └── adapter_config.json
├── epoch_1/
│   └── ...
└── logs/
    └── wandb/
```

## Key Conventions

### Configuration Defaults

- **LoRA alpha**: Automatically set to 2 × rank by `setup_finetune.py`
- **Run names**: Auto-generated positive adjective-noun pairs (e.g., "happy-narwhal") via `utils/run_names.py`
- **Output structure**: `{output_dir_base}/ck-out-{run_name}/epoch_N/`

### Checkpoint Management

- **epochs_to_save**: Controls which epochs to save
  - `'all'` - Save every epoch (default)
  - `'none'` - Don't save any checkpoints
  - `"0,2,4"` - Comma-delimited list of specific epochs

- **save_last_epoch_only**: `'true'`/`'false'` - Only save the final epoch

- **stash_adapter_weights**: `'true'`/`'false'` - Moves adapter files to subdirectory after merging to avoid confusing inspect-ai

### Custom Recipe Usage

cruijff_kit uses modified torchtune recipes for added features. To use a custom recipe:

```bash
python setup_finetune.py --custom_recipe cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_val
```

Available custom recipes:
- `lora_finetune_single_device_v1.py` - Single GPU with selective epoch saving
- `lora_finetune_distributed_v1.py` - Multi-GPU distributed training
- `lora_finetune_single_device_val.py` - With validation loss tracking (requires torchtune nightly)

## Key Abstractions

### 1. Two-Stage Configuration

**Why?** Separate user-friendly config from torchtune's internal format.

- **Stage 1:** User edits `setup_finetune.yaml` (high-level, task-focused)
- **Stage 2:** `setup_finetune.py` generates `finetune.yaml` (torchtune format)

**Benefits:**
- Users specify paths once (base directories)
- Sensible defaults reduce boilerplate
- Validation happens before expensive compute jobs
- SLURM and torchtune configs stay synchronized

### 2. Recipe Customization

**Location:** `tools/torchtune/custom_recipes/`

**Approach:** Fork torchtune recipes and add features rather than monkey-patching.

**Added capabilities:**
- Selective epoch checkpointing
- Custom metric hooks
- Adapter weight organization
- Validation during training

### 3. Path Resolution Strategy

Scripts use relative paths from their location:

```python
script_dir = Path(__file__).parent
template_path = script_dir / "templates" / "finetune_template.yaml"
```

User-provided paths are resolved relative to current working directory (the task folder).

## Common Patterns

### Running a Task

1. Navigate to task directory: `cd tasks/capitalization/`
2. Copy and edit config: `cp templates/finetuning/setup_finetune_json.yaml setup_finetune.yaml`
3. Generate scripts: `python ../../tools/torchtune/setup_finetune.py`
4. Submit job: `sbatch finetune.slurm`
5. Evaluate: `python ../../tools/inspect/setup_inspect.py --finetune_epoch_dir /path/to/epoch_0/`
6. Run evaluation: `sbatch inspect.slurm`

### Adding a New Task

1. Create directory under `tasks/`
2. Add `README.md` with task description
3. Create `setup_finetune.yaml` from template
4. Add data generation scripts to `input/`
5. Create task-specific `run_inspect.py` for evaluation
6. Document the workflow in task README

### Using Utilities

Common utilities in `utils/`:
- `run_names.py` - Generate random experiment names
- `llm_utils.py` - Load models, run inference
- `finetune_custom_metrics.py` - Define training metrics
- `check_if_model_is_finetuned.py` - Inspect model state
- Format converters for datasets

## HPC Integration

### SLURM Script Generation

`setup_finetune.py` creates SLURM scripts with:
- GPU allocation (`--gpus`)
- Time limits (`--time`)
- Environment setup (`--conda_env` or `--venv`)
- Module loading (`--modules`)
- Account/partition/constraint options

### Environment Assumptions

- **Conda environment** with torchtune, torch, inspect-ai
- **Shared storage** for models (`/scratch/gpfs/MSALGANIK/pretrained-llms/`)
- **User scratch space** for outputs (`/scratch/gpfs/MSALGANIK/$USER/`)

## Extension Points

### Adding Custom Metrics

1. Edit `utils/finetune_custom_metrics.py`
2. Define metric function following torcheval patterns
3. Recipe automatically imports and uses it

### Supporting New Dataset Formats

1. Add conversion utility to `utils/`
2. Update `setup_finetune.py` dataset type handling
3. Consider adding template config

### Creating New Custom Recipes

1. Copy existing recipe from `tools/torchtune/custom_recipes/`
2. Import `custom_recipe_utils` for common functionality
3. Reference new recipe with `--custom_recipe` flag

## Key Dependencies

- **torchtune** - Fine-tuning framework (supports stable release and nightly)
- **inspect-ai** - Evaluation framework
- **transformers** - Model loading (via llm_utils)
- **datasets** - HuggingFace dataset integration
- **wandb** - Experiment tracking
- **PyTorch** - Underlying ML framework

## Principles

These core principles guide all architectural and implementation decisions in cruijff_kit:

1. **Scientific** - All work emphasizes correctness, computational reproducibility, and detailed logging. Each experiment should be logged such that it can be audited by a researcher or LLM assistant.

2. **Modular** - The project will evolve over time and should be designed so that individual components can be added or changed with minimal impact on other components.

3. **Practical** - This project is designed to do science, not win a programming contest. Don't over-engineer or do premature optimization. We don't need hundreds of lines of code to save 5 seconds.

4. **Privacy respecting** - Much of the data in this project is about people. All data should be treated with care, and some should never leave the user's computer system. Tasks should be designed with clear data governance.

5. **Self improving** - Always look for ways to learn from earlier experiments to design new experiments, improve workflows, and improve analysis. The more work we do, the easier things should be because we have more designs, results, and logs from which to learn.

## Practices

These technical practices support the principles above:

1. **Configuration over code** - Use YAML to define experiments
2. **Single source of truth** - One config generates both torchtune and SLURM files
3. **HPC-first** - Designed for SLURM clusters, not laptops
4. **Modular architecture** - Easy to add tasks, tests, and custom behaviors
5. **Explicit over implicit** - Users see generated files before running
6. **Relative imports** - Package utilities for reusability

## Historical Context

- Originally named "predicting-zygosity" (twin prediction focus)
- Renamed to "cruijff_kit" to reflect broader research toolkit purpose
- Evolved from manual torchtune config editing to automated generation
- Added inspect-ai integration for standardized evaluation

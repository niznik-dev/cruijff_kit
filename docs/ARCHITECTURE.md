# cruijff_kit Architecture

This document describes the structure and organization of the cruijff_kit codebase. It's designed to help both LLM assistants and human developers understand how the project is organized and how different components interact.

## Overview

cruijff_kit is a research toolkit for fine-tuning and evaluating LLMs on social science data. The architecture supports:

1. **Declarative workflow configuration** via YAML files
2. **Automated SLURM script generation** for HPC environments
3. **Flexible dataset formats** (JSON, HuggingFace datasets)
4. **Custom torchtune recipes** with validation support
5. **Inspect-ai integration** for model evaluation

## Directory Structure

```
cruijff_kit/
├── src/
│   ├── tools/                      # Core workflow orchestration scripts
│   │   ├── torchtune/              # Fine-tuning setup and custom recipes
│   │   │   ├── setup_finetune.py        # Generate fine-tuning configs and SLURM scripts
│   │   │   ├── config_recipe_loader.py  # Load and merge torchtune recipe configs
│   │   │   ├── extract_loss.py          # Pull loss curves out of training logs
│   │   │   ├── model_configs.py         # Per-model GPU/tokenizer settings
│   │   │   ├── calc_token_stats.py      # Token-count stats for fine-tuning datasets
│   │   │   ├── check_if_model_is_finetuned.py  # Diagnostic: base vs PEFT-adapter divergence
│   │   │   ├── custom_recipes/          # Modified torchtune recipes
│   │   │   ├── datasets/                # Custom dataset classes (chat_completion, text_completion)
│   │   │   ├── templates/               # YAML/SLURM templates
│   │   │   └── yaml_refs/               # Reference yaml fragments
│   │   ├── inspect/                # Evaluation setup and analysis
│   │   │   ├── setup_inspect.py         # Generate evaluation SLURM scripts from template
│   │   │   ├── parse_eval_log.py        # Parse inspect-ai evaluation logs
│   │   │   ├── prebuild_cache.py        # Pre-build HF datasets cache before SLURM dispatch
│   │   │   ├── pdf_preprocess.py        # Markdown→PDF preprocessing for authored reports
│   │   │   ├── summary_binary.py        # Binary-classification summary helpers
│   │   │   ├── viz_helpers.py           # Plot/data adapters for inspect-viz
│   │   │   ├── spot_check.py            # Quick inference check (mirrors inspect HF backend)
│   │   │   ├── scorers/                 # Custom scorers (risk_scorer, calibration_metrics, …)
│   │   │   ├── templates/               # SLURM templates (eval_template.slurm)
│   │   │   └── heterogeneity/           # Group-level fairness analysis
│   │   ├── experiment/             # Experiment-level operations
│   │   │   ├── archive_experiment.py    # Archive a completed experiment
│   │   │   └── prepare_data.py          # Top-level dataset preparation entry point
│   │   ├── slurm/                  # Compute-estimation/metrics utils for the design-experiment skill (no production importers)
│   │   │   └── compute_gpu_metrics.py   # GPU metrics aggregation
│   │   └── model_organisms/        # Synthetic sequence-labeling framework
│   │       ├── inputs.py                # Input-type registry (bits, digits, letters)
│   │       ├── rules.py                 # Output-rule registry (parity, first, majority, …)
│   │       ├── formats.py               # Text-rendering registry (spaced, dense, …)
│   │       ├── generate.py              # Dataset generator CLI
│   │       └── inspect_task.py          # Unified inspect-ai evaluation task
│   ├── utils/                      # Generic, cross-cutting infra owned by no single domain
│   │   ├── layout.py                    # Layout constants (e.g. ARTIFACTS_DIR)
│   │   ├── run_names.py                 # Random name generation for runs
│   │   └── logger.py                    # Structured logging utilities
│   └── tabular_to_text_gen/        # Tabular→text conversion pipeline (own ARCHITECTURE.md)
│       ├── convert.py                   # CLI entry point
│       ├── lib/                         # Conversion engine + perturbations + templates
│       └── schemas/                     # Schema YAML files for source datasets
│
├── blueprints/                     # Task blueprints (5-slot shape: README, inspect_task.py, generate_data.py, optional modifiers/ + baseline.py)
│   ├── capitalization/             # Generalization test with word capitalization
│   ├── folktexts/                  # Demographic prediction from ACS text
│   └── model_organism/             # Synthetic framework pointer (code lives in src/tools/model_organisms/)
│
├── docs/                           # Architecture, workflow, and reference docs
├── tests/                          # Test suite (pytest)
│   ├── unit/                       # Unit tests (no GPU required)
│   └── integration/                # Integration tests (GPU/cluster required)
├── .claude/                        # Claude Code skills and project config
│   └── skills/                     # Primary + utility skills (design-experiment, scaffold-experiment, …)
└── assets/                         # Static assets used by docs
```

## Data Organization

cruijff_kit keeps datasets out of the repo. Input data lives at `{ck_data_dir}` on the user's filesystem (configured in `claude.local.md`), organized as `{ck_data_dir}/{project}/`. For example, capitalization data lives at `{ck_data_dir}/capitalization/words_5L_80P_1000.json` and folktexts data at `{ck_data_dir}/folktexts/acs_income_*.json`.

This keeps the repo small and lets each user manage their own data governance — datasets are user-provisioned and not version-controlled.

## Core Workflows

### 1. Fine-tuning Workflow

**Entry point:** `src/tools/torchtune/setup_finetune.py`

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
- `src/tools/torchtune/setup_finetune.py` - Main orchestration script
  - Reads `setup_finetune.yaml` (user-friendly config)
  - Generates `finetune.yaml` (torchtune recipe config)
  - Generates `finetune.slurm` (SLURM batch script)
  - Handles path resolution, validation, defaults

- `src/tools/torchtune/templates/finetune_template.yaml` - Base template for torchtune configs

- `src/tools/torchtune/custom_recipes/` - Modified torchtune recipes
  - `lora_finetune_single_device_nightly.py` - Default for single-GPU runs; supports `run_val_every_n_steps` / `dataset_val`
  - `lora_finetune_distributed_stable.py` - Multi-GPU distributed training (no val support yet — see #474)
  - `lora_finetune_single_device_stable.py` - Older single-GPU recipe without val loop; retained as escape hatch for non-nightly torchtune installs
  - `custom_recipe_utils.py` - Shared utilities for recipes

**Custom features added to torchtune:**
- Adapter-only saves with self-loading offline (rewrite `adapter_config.json` base path after save; `port_cruijff_adapter` restores HF Hub name for export)
- Validation during training (requires nightly build)

### 2. Evaluation Workflow

> **Note:** The `scaffold-inspect` agent (invoked via `scaffold-experiment` skill) is the recommended way to set up evaluations. It writes `eval.yaml` and calls `setup_inspect.py` to render SLURM scripts from a template.

**Entry point:** `src/tools/inspect/setup_inspect.py` (reads `eval.yaml`, renders `eval_template.slurm`)

**Process** (per-cell layout, issue #498 — one cell directory per (task, epoch)):
```
eval/{cell_name}/eval.yaml → setup_inspect.py → eval/{cell_name}/cell.slurm
                                                              ↓
                                                        sbatch cell.slurm
                                                              ↓
                                                        task-specific inspect task
                                                              ↓
                                                        inspect-ai evaluation
```

**Key files:**
- `src/tools/inspect/setup_inspect.py` - Renders eval SLURM scripts from `eval_template.slurm`
  - Reads experiment-specific config from `eval.yaml`
  - Looks up GPU resources from `model_configs.py`
  - Template includes GPU monitoring, SLURM log management

- Experiment-specific inspect-ai task files (e.g., `blueprints/capitalization/inspect_task.py`)
  - Define evaluation prompts and scoring
  - Use inspect-ai framework

### 3. Heterogeneity Analysis (Post-Evaluation)

**Entry point:** `src/tools/inspect/heterogeneity/`

**Purpose:** Detect performance bias across demographic or experimental groups

**Process:**
```
Model predictions (CSV) → heterogeneity_report.py → analysis + visualizations
                                  OR
                          heterogeneity_eval.py (inspect-ai wrapper)
```

**Key files:**
- `src/tools/inspect/heterogeneity/heterogeneity_report.py` - Standalone analysis
  - Statistical tests (ANOVA for accuracy, variance for AUC)
  - Identifies outlier groups
  - Generates visualizations and JSON reports

- `src/tools/inspect/heterogeneity/heterogeneity_eval.py` - Inspect-ai integration
  - Wraps heterogeneity analysis as inspect-ai task
  - Returns standardized metrics

- `src/tools/inspect/heterogeneity/README.md` - Full documentation
  - Input format requirements
  - Usage examples (standalone and inspect-ai)
  - Statistical methods explained
  - Output interpretation guide

**When to use:**
- Evaluating fairness across demographic groups
- Understanding which subpopulations a model serves well/poorly
- Post-hoc analysis of any binary classification predictions with group labels

## Blueprints

Research task blueprints in `blueprints/`. Each blueprint follows the 5-slot shape (3 required + 2 optional):

- `README.md` — REQUIRED: task description + how to run
- `inspect_task.py` — REQUIRED: inspect-ai evaluation definition
- `generate_data.py` — REQUIRED: primary data producer
- `modifiers/` — OPTIONAL: data transforms (e.g., folktexts has ACS format converters)
- `baseline.py` — OPTIONAL: non-LLM comparison

Current blueprints:

- **capitalization**: Tests generalization by training on 5-letter words and evaluating on other lengths
- **folktexts**: Demographic prediction from ACS (census) text — binary classification tasks including income, employment, mobility, public coverage, and travel time
- **model_organism**: README pointer only; all code is generic and lives in `src/tools/model_organisms/`

### Model Organisms (`src/tools/model_organisms/`)

Composable framework for sequence-labeling tasks with known ground-truth rules. An experiment is specified by choosing an input type (`bits`, `digits`, `letters`), an output rule (`parity`, `first`, `last`, `majority`, `constant`, `coin`, …), a format (`spaced`, `dense`, `comma`, `tab`, `pipe`), and a design (`memorization`, `in_distribution`, `ood`). Single unified inspect-ai task evaluates any combination; data generation is invoked by `scaffold-experiment` via a `data.data_generation` block with `tool: model_organism`.

## Package Structure

cruijff_kit is installed as an editable package (`pip install -e .`), making utilities importable:

```python
from cruijff_kit.utils import run_names
```

**Selected packaged modules** (full list in `pyproject.toml`):
- `cruijff_kit.utils` - Shared utilities
- `cruijff_kit.tools.torchtune.custom_recipes` - Custom torchtune recipes
- `cruijff_kit.tools.torchtune.datasets` - Custom dataset classes (e.g., `chat_completion`)
- `cruijff_kit.tools.model_organisms` - Synthetic sequence-labeling primitives
- `cruijff_kit.tools.inspect` (+ `.scorers`) - Eval helpers and custom scorers
- `cruijff_kit.tabular_to_text_gen` (+ `.lib`, `.lib.perturbations`, `.lib.templates`) - Tabular→text conversion pipeline

**Not packaged but executed as scripts:**
- `src/tools/torchtune/setup_finetune.py`
- `src/tools/inspect/setup_inspect.py`
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

### Output Structure

After running setup_finetune.py:
```
{run_name}/
├── setup_finetune.yaml    # User config
├── finetune.yaml          # Generated torchtune config
└── finetune.slurm         # Generated SLURM script
```

After running finetune:
```
{run_name}/
└── artifacts/
    ├── epoch_0/
    │   ├── adapter_model.safetensors  # LoRA weights
    │   └── adapter_config.json
    ├── epoch_1/
    │   └── ...
    └── logs/
        └── wandb/
```

## Key Conventions

### experiment_summary.yaml Section Membership

A field's home in `experiment_summary.yaml` is **predictable, not memorized**. Decide it
with two questions, answered in order:

**1. Granularity — does the value vary, and at what level?**

| Granularity | Home |
|---|---|
| Same for the whole experiment | a top-level design section (`controls` / `data` / `evaluation` / `models` / `output`) |
| Differs across runs *by design* (the independent variable) | `variables` (the swept axis), echoed per-run in `runs[].parameters` |
| Differs per eval cell (task / epoch) | `evaluation.tasks[]` / `evaluation.matrix[]` |

**2. Role — what does the value describe?**

| Section | Holds | Test |
|---|---|---|
| `experiment` | identity / metadata of the experiment as a whole | "Does this identify or describe the experiment itself?" |
| `tools` | backend routing | "Does this select which external tool processes it?" |
| `variables` | the swept axes (independent variables) | "Does this vary across runs by design?" |
| `controls` | experiment-wide invariants that shape the task or training | "Held constant across all runs, regardless of which stage reads it?" |
| `models` | model resources consumed | "Is this a model the experiment runs on?" |
| `data` | facts about the dataset itself (paths, splits, generation spec) | "Is this a property of the data, not the task framing?" |
| `output` | external publication sinks | "Is this where results/artifacts get published?" |
| `runs` | per-run instantiations (identity, type, model, parameter *values*) | "Is this a per-run realization of the design?" |
| `evaluation` | knobs consumed *only* by the eval stage | "Does *only* evaluation read this?" |

**Two boundary rules** (these resolve the recurring confusions):

1. **`controls` is not "training-only."** It is the home for experiment-wide *invariants*,
   whether training, eval, or both consume them. `prompt`, `system_prompt`, and
   `dataset_type` live here because they are constant across runs and shared by train and
   eval — the consuming stage does not decide the home. (Note: single-sourcing `dataset_type`
   in `controls` also *constrains* it to one modality per experiment — a mixed base
   `text_completion` vs. instruct `chat_completion` sweep isn't expressible today.)
2. **A field both train and eval need lives in `controls`, never duplicated into
   `evaluation`.** `propagate.py` carries the `controls` value into the eval config
   (`eval.yaml`). One source, no hand-kept copy.

**Duplication vs. independence.** When a value seems to live in two places, classify it:

- **Must-match duplication** (same semantic value, kept equal only by convention) →
  *collapse to one home + propagate.* Example: `system_prompt` is single-sourced at
  `controls.system_prompt`.
- **Legitimately-independent per-stage instance** (the two values may validly differ) →
  *keep both, document the independence.* Example: `controls.seed` (training stochasticity)
  and `evaluation.seed` (eval sampling RNG) are independent; both default to 14.

**Three "system prompt" surfaces** (name them to avoid conflation): generation-time
`data.data_generation.context` (with `context_placement: system_prompt`, baked into the
dataset text); the runtime `controls.system_prompt` (training + eval); and the per-cell
`evaluation.tasks[].system_prompt` override. They operate at different stages and are
genuinely distinct.

### Configuration Defaults

- **LoRA alpha**: Automatically set to 2 × rank by `setup_finetune.py`
- **Run names**: Auto-generated positive adjective-noun pairs (e.g., "happy-narwhal") via `src/utils/run_names.py`
- **Output structure**: `{experiment_dir}/{run_name}/artifacts/epoch_N/`

### Checkpoint Management

- **Epoch indexing**: **IMPORTANT - Epochs are 0-indexed**
  - First epoch is saved as `epoch_0/`, not `epoch_1/`
  - Training for 1 epoch produces `epoch_0/`
  - Training for 2 epochs produces `epoch_0/` and `epoch_1/`
  - When referencing epochs in evaluation scripts, use the 0-indexed value
  - Example: After 1 epoch of training, evaluate using `epoch_0`

- **save_adapter_weights_only**: `'true'`/`'false'` (default `'true'`) - Save only the LoRA adapter (~MB), skip the merged base+LoRA checkpoint (~GB). The saved `adapter_config.json` has its `base_model_name_or_path` rewritten to the local base-model path so the dir is self-loading on offline compute. Use `python -m cruijff_kit.tools.torchtune.port_cruijff_adapter <epoch_dir>` to restore the HF Hub name when exporting.

### Custom Recipe Usage

cruijff_kit uses modified torchtune recipes for added features. To use a custom recipe:

```bash
python setup_finetune.py --custom_recipe cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_nightly
```

Available custom recipes:
- `lora_finetune_single_device_nightly.py` - Default for single-GPU; supports validation during training (requires torchtune nightly)
- `lora_finetune_distributed_stable.py` - Multi-GPU distributed training (no val support yet — see #474)
- `lora_finetune_single_device_stable.py` - Older single-GPU recipe; retained as escape hatch for non-nightly torchtune installs

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

**Location:** `src/tools/torchtune/custom_recipes/`

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

### Running Experiments

cruijff_kit supports two workflows:

#### Skills-Based Workflow (Recommended)

Use Claude Code skills to automate multi-run experiments. Skills generate all configurations directly from `experiment_summary.yaml`.

1. **Design:** Use `design-experiment` skill to create experiment plan (`experiment_summary.yaml`)
2. **Scaffold:** Use `scaffold-experiment` skill to generate all run directories and configs
3. **Execute:** Use `run-experiment` skill to run fine-tuning and evaluation
4. **Summarize:** Use `summarize-experiment` skill to generate the results summary — the required post-run step
5. **Explore (optional):** Use `explore-experiment` skill for detailed analysis and comparison — optional, and can be run any time after the experiment
6. **Archive:** Use `archive-experiment` skill to archive completed experiments (preserves configs/logs/results, deletes large checkpoints)

**Benefits:**
- Automated setup for multi-run experiments
- Consistent naming and organization
- Progress tracking and status updates
- Built-in safety (stagger delays prevent cache collisions)

**Implementation Note:** The workflow skills are orchestrators that delegate to specialized worker skills: `scaffold-experiment` calls `scaffold-torchtune` and `scaffold-inspect`, while `run-experiment` calls `run-torchtune` and `run-inspect`. Worker skills can also be invoked independently. See [SKILLS_ARCHITECTURE_SUMMARY.md](SKILLS_ARCHITECTURE_SUMMARY.md) for details.

#### Manual Workflow

For users who prefer direct control or don't have Claude Code access. See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) for a step-by-step walkthrough.

At a high level: hand-write `setup_finetune.yaml` in each run directory under `ck-projects/{project}/{experiment_name}/{run_name}/`, then run `python <cruijff_kit>/src/tools/torchtune/setup_finetune.py` in each to generate `finetune.yaml` + `finetune.slurm`, then `sbatch` each. For multi-run experiments, stagger submissions with `sleep 5` between them to avoid HuggingFace datasets cache race conditions.

### Adding a New Blueprint

1. Create directory under `blueprints/`
2. Add `README.md` with task description
3. Create `generate_data.py` (primary data producer)
4. Create `inspect_task.py` using the `create-inspect-task` skill
5. Optional: add `baseline.py` (non-LLM comparison) and/or `modifiers/` (data transforms)
6. Document the workflow in the blueprint README

### Tool Domains

Each subfolder of `src/tools/` is a domain package. Name a domain after the external
tool when one tool dominates it (`torchtune/`, `inspect/`); otherwise name it after the
pipeline stage it serves (`experiment/`, `slurm/`) or the framework it provides
(`model_organisms/`). Code owned by no single domain belongs in `src/utils/`, not a
tool folder (see below).

> **`inspect/` shadows the stdlib.** `src/tools/inspect/` shares its name with Python's
> standard-library `inspect` module (it wraps the external inspect-ai tool). No module
> in that subtree does a bare `import inspect`, and none should rely on one resolving to
> the stdlib — if that directory ever lands on `sys.path` as a top-level entry, the
> import could bind to the package instead. Keep stdlib-`inspect` use out of the subtree,
> or alias it explicitly. The hazard is also recorded in `src/tools/inspect/__init__.py`.

### Using Utilities

`src/utils/` holds generic, cross-cutting infrastructure owned by no single domain —
either imported across multiple tool domains (e.g. `logger`, `layout`) or generic
enough that any domain could use it, even if only one wires it up today (e.g.
`run_names`). Domain-specific scripts live in their domain package under `src/tools/`
instead.

- `layout.py` - Layout constants (e.g. `ARTIFACTS_DIR`)
- `run_names.py` - Generate random experiment names
- `logger.py` - Structured logging helpers

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
- **Shared storage** for models (configure path in `claude.local.md`)
- **User scratch space** for outputs (configure path in `claude.local.md`)

## Extension Points

### Supporting New Dataset Formats

1. Add conversion utility to `src/utils/`
2. Update `setup_finetune.py` dataset type handling
3. Consider adding template config

### Creating New Custom Recipes

1. Copy existing recipe from `src/tools/torchtune/custom_recipes/`
2. Import `custom_recipe_utils` for common functionality
3. Reference new recipe with `--custom_recipe` flag

## Key Dependencies

- **torchtune** - Fine-tuning framework (supports stable release and nightly)
- **inspect-ai** - Evaluation framework
- **transformers** - Model loading
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

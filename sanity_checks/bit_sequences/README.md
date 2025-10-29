# Bit Sequences Sanity Check

## Purpose

This sanity check runs fine-tuning on synthetic binary sequence datasets to test how well models learn binary classification tasks. Labels can be assigned either deterministically via **bit parity** (even/odd number of 1s) or probabilistically via a **Bernoulli distribution** with probability `p`.

The goal is to validate that:
1. Models can learn simple deterministic patterns (parity)
2. Models behave appropriately with probabilistic labels
3. The fine-tuning workflow handles binary classification correctly
4. Training/validation splits and metrics work as expected

## Dataset Types

**Parity-based (deterministic):**
- Input: Binary sequence (e.g., `01011`)
- Output: `1` if odd number of 1s, `0` if even number of 1s
- Optional: Inject noise by flipping labels with probability `p`

**Probabilistic:**
- Input: Binary sequence (e.g., `01011`)
- Output: `1` with probability `p`, `0` with probability `1-p`
- Labels are independent of input pattern

## Directory Structure

**Sanity check code** (in repository):
```
sanity_checks/bit_sequences/
├── README.md              # This file
└── generate.py            # Generates datasets with various configurations
```

**Sanity check data** (in repository, generated):
```
data/green/bit_sequences/
└── parity.json            # Generated data (not in git)
```

**Sanity check runs** (in scratch space):
```
/scratch/gpfs/MSALGANIK/niznik/ck-sanity-checks/
└── sanity_check_bit_sequences_YYYY-MM-DD/
    ├── parity_run/
    ├── probabilistic_run/
    ├── experiment_summary.md
    ├── design-experiment.log
    └── scaffold-experiment.log
```

**Outputs** (grouped by sanity check name):
```
/scratch/gpfs/MSALGANIK/niznik/ck-outputs/
└── sanity_check_bit_sequences_YYYY-MM-DD/
    ├── ck-out-parity_run/
    ├── ck-out-probabilistic_run/
    └── ...
```

## How to Run

### Step 0: Generate Data

From the repository root or sanity check directory:

**Parity dataset (deterministic, no noise):**
```bash
cd sanity_checks/bit_sequences
python generate.py --bit_length 15 --N 33000 --p 0 --bit_parity True --val_size 4000 --output parity.json
```

**Probabilistic dataset (p=0.5):**
```bash
python generate.py --bit_length 15 --N 33000 --p 0.5 --bit_parity False --val_size 4000 --output prob_p05.json
```

**Note:** The generator deduplicates at the sequence level - no sequence string appears in both train and validation. Use `bit_length` such that 2^bit_length >> val_size (e.g., 2^15 = 32,768 >> 4,000) to avoid removing all training data.

This creates a single JSON file in `data/green/bit_sequences/` with nested `train` and `validation` keys.

### Primary Method: Skills-Based Workflow (Recommended)

The skills-based workflow automates experiment planning, config generation, and job submission:

```bash
# 1. Navigate to sanity check directory
cd sanity_checks/bit_sequences

# 2. Design experiment (creates experiment plan)
# Claude Code will ask questions and create experiment_summary.md
/design-experiment

# 3. Generate all configs (creates run directories with configs and SLURM scripts)
/scaffold-experiment

# 4. Execute all runs (submits jobs and monitors progress)
/run-experiment
```

**Benefits:**
- Systematic planning and documentation
- Automated config generation for all runs
- Resource verification before execution
- Detailed logging of decisions
- Easier to replicate and modify

### Alternative: Manual Workflow

If you prefer manual control or need to customize individual runs:

#### Configuration File Approach

1. Create a `setup_finetune.yaml` in each run directory with your settings:

```yaml
# Example: parity_run/setup_finetune.yaml
my_wandb_project: 'bit_sequences_2025-10-28'
my_wandb_run_name: 'parity_run'
dataset_label: 'parity'          # Base name without extension
dataset_ext: '.json'             # Extension
train_on_input: 'true'           # Learn from both input and output
batch_size: 1
epochs: 10
log_every_n_steps: 1
run_val_every_n_steps: 50
save_adapter_weights_only: 'true'
input_formatting: ''
system_prompt: ''
custom_recipe: 'cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_val'
```

2. Generate configs and submit:

```bash
cd /path/to/run/directory
python ../../tools/torchtune/setup_finetune.py
sbatch finetune.slurm
```

## Data Generation Options

### Arguments

* `--bit_length`: Length of each binary sequence (e.g., 5)
* `--N`: Total number of examples (must be ≥ 2^bit_length)
* `--val_size`: Number of validation examples
* `--bit_parity`: `True` for parity-based labeling, `False` for probabilistic
* `--p`: Noise level (if parity) or Bernoulli probability (if probabilistic)
* `--seed`: Random seed (default: 42)
* `--output`: Output filename (e.g., parity.json, prob_p05.json)

### Data Format

**Current format** (used by this sanity check):
- Single JSON file with `train` and `validation` as top-level keys
- Splits determined by file structure (e.g., 9000 train / 1000 validation)

```json
{
  "train": [
    {"instruction": "", "input": "01011", "output": "1"},
    {"instruction": "", "input": "11010", "output": "0"}
  ],
  "validation": [
    {"instruction": "", "input": "00111", "output": "1"}
  ]
}
```

When using with `setup_finetune.yaml`, specify:
- `dataset_label: 'parity'` (base name without extension for parity.json)
- The validation split is automatically loaded from the same file

**Parameter changes** (compared to older versions):
- ✓ Use: `--dataset_label parity` (base name without extension)
- ✓ Use: `--dataset_ext '.json'`
- ✗ Removed: `--dataset_filename`, `--dataset_val_filename`, `--test_size`
- ✓ Use: `--val_size` instead of `--test_size`
- ✓ Use: `--output` to specify output filename

## Validation During Training

This sanity check uses the `lora_finetune_single_device_val` custom recipe, which:
- Validates periodically during training (`run_val_every_n_steps: 50`)
- Tracks both train and validation loss in real-time
- Helps detect overfitting during training
- Logs metrics to Weights & Biases

Both train and validation loss curves will appear in your W&B dashboard, allowing you to verify learning dynamics.

## Configuration Details

**Common parameters across runs:**
- Model: `Llama-3.2-1B-Instruct`
- Epochs: 10
- Batch size: 1
- LoRA rank: 8 (default)
- LoRA alpha: 16 (auto-set to 2 × rank)
- Log every: 1 step
- Validate every: 50 steps
- System prompt: "" (empty)
- Custom recipe: `cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_val`

**Experimental variables** (differ across runs):
- `bit_length`: Length of binary sequences
- `bit_parity`: Deterministic vs probabilistic labeling
- `p`: Noise/probability parameter

## Weights & Biases

### Project Naming

Use timestamped project names to prevent conflicts with previous runs:
- Example: `bit_sequences_2025-10-28`

### Syncing Results

If runs complete offline or need manual sync:

```bash
# Sync a specific run
wandb sync /path/to/ck-out-{run_name}/logs/wandb/latest-run

# Or from the output directory
cd /scratch/gpfs/MSALGANIK/niznik/ck-outputs/sanity_check_bit_sequences_YYYY-MM-DD/
for dir in ck-out-*/; do
  wandb sync "$dir/logs/wandb/latest-run"
done
```

## Expected Results

This sanity check focuses on **training dynamics** and binary classification learning. No inspect-ai evaluation is performed.

**Key comparisons to examine in W&B:**
1. **Parity tasks**: Model should achieve near-zero loss (pattern is learnable)
2. **Probabilistic tasks**: Loss should plateau at a level reflecting the inherent uncertainty
3. **Validation curves**: Should track training curves without significant overfitting
4. **Convergence speed**: Simpler patterns (shorter bit_length) should converge faster

## Notes

- Multiple runs can execute in parallel (estimated 40-80 minutes per run on one A100 GPU)
- This sanity check validates: data generation → fine-tuning → monitoring
- Designed as a learning tool for understanding binary classification with cruijff_kit
- No evaluation phase (W&B loss comparison is sufficient for this sanity check)

## Troubleshooting

**Data files missing:**
```bash
cd sanity_checks/bit_sequences
python generate.py --bit_length 15 --N 33000 --p 0 --bit_parity True --val_size 4000 --output parity.json
```

**Wrong directory structure:**
- Sanity check code: Lives in `sanity_checks/` in the repository
- Sanity check runs: Should be created in `/scratch/.../ck-sanity-checks/`
- Outputs: Will be in `/scratch/.../ck-outputs/{sanity_check_name}/`

**Parameters don't match:**
- This README reflects the current workflow (October 2025)
- If you see references to `--test_size` or separate `train.json`/`test.json` files, they're outdated
- Use `--val_size` and `--output` with nested train/validation structure
- Use `--dataset_label` and `--dataset_ext` instead of `--dataset_filename`

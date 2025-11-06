# Predictable or Not Sanity Check

## Purpose

This sanity check runs a simple fine-tuning task on small synthetic datasets to compare how training and validation loss differ between two training modes:

1. **Output only** (default): Model learns only from the output tokens
2. **Input and output**: Model learns from both input and output tokens (set `train_on_input: true`)

The goal is to understand how predictability in inputs and outputs affects learning dynamics under different training modes.

## Four Scenarios

Each scenario contains 1000 examples (900 train / 100 validation, 90/10 split):

1. **pp** - Predictable Input → Predictable Output
   - Input: Sequential pattern (e.g., `42,43,44,45,46`)
   - Output: Next number in sequence (e.g., `47`)

2. **pu** - Predictable Input → Unpredictable Output
   - Input: Sequential pattern (e.g., `42,43,44,45,46`)
   - Output: Random integer (1-1000)

3. **up** - Unpredictable Input → Predictable Output
   - Input: Random numbers (e.g., `1,1000,400,300,700`)
   - Output: Always `42`

4. **uu** - Unpredictable Input → Unpredictable Output
   - Input: Random numbers
   - Output: Random integer (1-1000)

## Complete Experiment Design

This sanity check tests **8 runs total**:
- 4 scenarios (pp, pu, up, uu)
- 2 training modes per scenario (output_only, input_and_output)

## Directory Structure

**Sanity check code** (in repository):
```
sanity_checks/predictable_or_not/
├── README.md                      # This file
├── generate_data.py               # Generates the 4 scenario datasets
└── predictable_or_not_inspect.py  # Inspect-ai evaluation task
```

**Sanity check data** (in repository, generated):
```
data/green/predictable_or_not/
├── pp.json               # Generated data (not in git)
├── pu.json
├── up.json
└── uu.json
```

**Sanity check runs** (in scratch space):
```
/scratch/gpfs/MSALGANIK/niznik/ck-sanity-checks/
└── sanity_check_predictable_or_not_YYYY-MM-DD/
    ├── pp_output_only/
    ├── pp_input_and_output/
    ├── pu_output_only/
    ├── pu_input_and_output/
    ├── up_output_only/
    ├── up_input_and_output/
    ├── uu_output_only/
    ├── uu_input_and_output/
    ├── experiment_summary.md
    ├── design-experiment.log
    └── scaffold-experiment.log
```

**Outputs** (grouped by sanity check name):
```
/scratch/gpfs/MSALGANIK/niznik/ck-outputs/
└── sanity_check_predictable_or_not_YYYY-MM-DD/
    ├── ck-out-pp_output_only/
    ├── ck-out-pp_input_and_output/
    └── ...
```

## How to Run

### Step 0: Generate Data

From the repository root or sanity check directory:

```bash
cd sanity_checks/predictable_or_not
python generate_data.py
```

This creates four JSON files in `data/green/predictable_or_not/` (`pp.json`, `pu.json`, `up.json`, `uu.json`), each with:
- 1000 total examples split into `train` (900) and `validation` (100) keys
- Format: `{"input": "...", "output": "..."}`
- No instruction field (empty `input_formatting` in config)

### Primary Method: Skills-Based Workflow (Recommended)

The skills-based workflow automates experiment planning, config generation, and job submission:

```bash
# 1. Navigate to sanity check directory
cd sanity_checks/predictable_or_not

# 2. Design experiment (creates experiment plan with all 8 runs)
# Claude Code will ask questions and create experiment_summary.md
/design-experiment

# 3. Generate all configs (creates 8 run directories with configs and SLURM scripts)
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
# Example: pp_output_only/setup_finetune.yaml
my_wandb_project: 'predictable_or_not_2025-10-27'
my_wandb_run_name: 'pp_output_only'
dataset_label: 'pp'              # Base name without extension
dataset_ext: '.json'             # Extension
train_on_input: 'false'          # or 'true' for input_and_output runs
batch_size: 1
epochs: 10
log_every_n_steps: 1
run_val_every_n_steps: 4
save_adapter_weights_only: 'true'
input_formatting: ''
system_prompt: ''
custom_recipe: 'cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_nightly'
```

2. Generate configs and submit:

```bash
cd /path/to/run/directory
python ../../tools/torchtune/setup_finetune.py
sbatch finetune.slurm
```

#### Submit All Runs at Once

From the experiment directory:

```bash
for dir in pp_output_only pp_input_and_output pu_output_only pu_input_and_output \
            up_output_only up_input_and_output uu_output_only uu_input_and_output; do
  (cd "$dir" && sbatch finetune.slurm)
done
```

## Data Format Details

**Current format** (used by this sanity check):
- Single JSON file with `train` and `validation` as top-level keys
- Splits determined by file structure (90/10 train/validation)

```json
{
  "train": [
    {"input": "1,2,3,4,5", "output": "6"},
    {"input": "2,3,4,5,6", "output": "7"}
  ],
  "validation": [
    {"input": "99,100,101,102,103", "output": "104"}
  ]
}
```

**Parameter changes** (compared to older versions):
- ✓ Use: `--dataset_label pp` (base name without extension)
- ✓ Use: `--dataset_ext '.json'`
- ✗ Removed: `--dataset_filename`, `--dataset_val_filename`, `--dataset_split_point`

The split logic moved from runtime parameters to file structure for better flexibility and consistency with torchtune patterns.

## Validation During Training

This sanity check uses the `lora_finetune_single_device_nightly` custom recipe, which:
- Validates every 4 training steps (`run_val_every_n_steps: 4`)
- Tracks both train and validation loss in real-time
- Helps detect overfitting during training
- Logs metrics to Weights & Biases

Both train and validation loss curves will appear in your W&B dashboard, allowing you to compare learning dynamics across scenarios and training modes.

## Configuration Details

**Common parameters across all runs:**
- Model: `Llama-3.2-1B-Instruct`
- Epochs: 10
- Batch size: 1
- LoRA rank: 8 (default)
- LoRA alpha: 16 (auto-set to 2 × rank)
- Log every: 1 step
- Validate every: 4 steps
- System prompt: "" (empty)
- Custom recipe: `cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_nightly`

**Experimental variables** (differ across runs):
- `train_on_input`: `false` (output_only) vs `true` (input_and_output)
- `dataset_label`: pp, pu, up, or uu

## Weights & Biases

### Project Naming

Use timestamped project names to prevent conflicts with previous runs:
- Example: `predictable_or_not_2025-10-27`

### Syncing Results

If runs complete offline or need manual sync:

```bash
# Sync a specific run
wandb sync /path/to/ck-out-{run_name}/logs/wandb/latest-run

# Or from the output directory
cd /scratch/gpfs/MSALGANIK/niznik/ck-outputs/sanity_check_predictable_or_not_YYYY-MM-DD/
for dir in ck-out-*/; do
  wandb sync "$dir/logs/wandb/latest-run"
done
```

## Evaluation Phase

After fine-tuning completes, evaluate the models using inspect-ai to measure accuracy on held-out validation data.

### Evaluation Task

The evaluation task is defined in `predictable_or_not_inspect.py` and automatically:
- Reads the fine-tuning configuration from `setup_finetune.yaml`
- Loads the same dataset used for training (e.g., pp.json for pp_output_only)
- Evaluates on the validation split (100 examples)
- Reports exact match and substring match accuracy

### Running Evaluation

**For a single run:**
```bash
cd /path/to/ck-sanity-checks/sanity_check_predictable_or_not_YYYY-MM-DD/pp_output_only
inspect eval ../../sanity_checks/predictable_or_not/predictable_or_not_inspect.py \
  --model hf/local \
  -M model_path=/path/to/ck-outputs/.../ck-out-pp_output_only/epoch_9 \
  -T config_dir=/path/to/ck-outputs/.../ck-out-pp_output_only/epoch_9
```

**For all runs:**
Use the skills-based workflow to automate evaluation across all 8 runs (implementation pending).

### Expected Evaluation Results

This sanity check validates both **training dynamics** (via W&B loss curves) and **final model performance** (via inspect-ai accuracy).

**Key comparisons:**

**Training dynamics (W&B):**
1. How does `train_on_input` affect learning speed and final loss?
2. Which scenarios show the biggest difference between output_only vs input_and_output?
3. Do validation loss curves show overfitting in any scenarios?
4. How does predictability of inputs vs outputs affect learning difficulty?

**Evaluation accuracy (inspect-ai):**
1. **pp (predictable → predictable)**: Should achieve ~100% accuracy (pattern is fully learnable)
2. **pu (predictable → unpredictable)**: Low accuracy expected (outputs are random)
3. **up (unpredictable → predictable)**: Should achieve ~100% accuracy (constant output "42")
4. **uu (unpredictable → unpredictable)**: Low accuracy expected (both random)
5. **train_on_input effect**: May see different accuracy between output_only vs input_and_output modes

## Notes

- All 8 runs can execute in parallel (estimated 40-80 minutes total on one A100 GPU)
- This sanity check validates the complete workflow: data generation → fine-tuning → monitoring → evaluation
- Designed as a learning tool for understanding cruijff_kit's fine-tuning and evaluation capabilities
- Evaluation phase validates model performance on held-out validation data beyond just loss curves

## Troubleshooting

**Data files missing:**
```bash
cd sanity_checks/predictable_or_not
python generate_data.py
```

**Wrong directory structure:**
- Sanity check code: Lives in `sanity_checks/` in the repository
- Sanity check runs: Should be created in `/scratch/.../ck-sanity-checks/`
- Outputs: Will be in `/scratch/.../ck-outputs/{sanity_check_name}/`

**Parameters don't match:**
- This README reflects the current workflow (October 2025)
- If you see references to `--dataset_filename` or `--dataset_split_point`, they're outdated
- Use `--dataset_label` and `--dataset_ext` instead

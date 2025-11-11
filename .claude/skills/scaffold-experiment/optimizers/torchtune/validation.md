# Parameter Validation

This module describes how to verify that generated finetune.yaml files contain correct parameter values matching the directory names.

## Purpose

**IMPORTANT:** After scaffolding, verify that the generated `finetune.yaml` files contain the correct parameter values matching the directory names. This catches bugs where setup_finetune.py might not properly substitute parameters.

## Parameter Field Mapping

Know where to find parameters in finetune.yaml:
- `lora_rank` → `model.lora_rank` (nested under model section)
- `lr` (learning rate) → `optimizer.lr` (nested under optimizer section, note: two-space indent)
- `batch_size` → `batch_size` (top level)
- `epochs` → `epochs` (top level)
- `my_wandb_run_name` → `my_wandb_run_name` (top level)
- `output_dir` → `output_dir` (top level, should include run name)

## Verification Script Pattern

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

## What to Verify

1. **Parameters varying across runs** (e.g., lora_rank, lr) match directory names
2. **Common parameters** (e.g., batch_size, epochs) match experiment configuration
3. **WandB run names** and output directories include correct run identifiers

## Handling Mismatches

**If mismatches found:**
- Report which runs have incorrect parameters
- Indicate which specific parameters are wrong
- Suggest checking if setup_finetune.py has all necessary arguments
- **Do NOT report success** - these runs would train with wrong hyperparameters!

## Notes

- The `lr` field in finetune.yaml has a **two-space indent** (`^  lr:`) - this is critical for grep to work correctly
- Adjust the verification script pattern based on which parameters vary in your experiment
- Always verify before submitting jobs to SLURM

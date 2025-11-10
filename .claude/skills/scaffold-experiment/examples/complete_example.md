# Complete Scaffolding Example

Concrete walkthrough of scaffolding an experiment from start to finish.

## Experiment Overview

**Goal:** Test how LoRA rank and learning rate affect fine-tuning performance

**Setup:**
- 8 fine-tuning runs (varying LoRA rank: 8, 16, 32, 64; learning rate: 1e-5, 5e-5)
- 1 evaluation task (capitalization accuracy)
- Evaluate last epoch (epoch 0) for each run

## Step 1: Locate and Verify

```bash
cd /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22
ls experiment_summary.md  # Verify exists
```

## Step 2: Scaffold Torchtune

### Create Run Directories

```bash
mkdir -p rank8_lr1e-5 rank8_lr5e-5 rank16_lr1e-5 rank16_lr5e-5 \
         rank32_lr1e-5 rank32_lr5e-5 rank64_lr1e-5 rank64_lr5e-5
```

### Generate Configs

For each run, create `setup_finetune.yaml` using the template structure from [templates/setup_finetune_template.yaml](../templates/setup_finetune_template.yaml), then execute:

```bash
module load anaconda3/2025.6
conda activate cruijff

for dir in rank*/; do
  (cd "$dir" && python /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/tools/torchtune/setup_finetune.py)
done
```

**Output:** Each run directory contains setup_finetune.yaml, finetune.yaml, finetune.slurm

### Validate Parameters

```bash
for dir in rank*/; do
  dir_clean=${dir%/}
  expected_rank=$(echo $dir_clean | grep -oP 'rank\K\d+')
  actual_rank=$(grep "lora_rank:" "$dir_clean/finetune.yaml" | awk '{print $2}')
  [ "$expected_rank" = "$actual_rank" ] && echo "✓ $dir_clean" || echo "✗ $dir_clean"
done
```

## Step 3: Scaffold Inspect-ai

### Verify Task Exists

```bash
inspect list /path/to/cap_task.py
# Output: capitalization
```

### Create Eval Directories and Scripts

```bash
for dir in rank*/; do
  mkdir -p "$dir/eval/logs"
done
```

Generate SLURM scripts for each run using the template from [templates/slurm_template.sh](../templates/slurm_template.sh), configured with:
- Model path: `{output_dir}/ck-out-{run_name}/epoch_0`
- Task: capitalization
- Config integration via `config_dir` parameter

## Step 4: Verify Structure

```bash
tree -L 3 /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/
```

**Expected:**
```
cap_4L_lora_lr_sweep_2025-10-22/
├── experiment_summary.md
├── scaffold.log
├── rank8_lr1e-5/
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   ├── finetune.slurm
│   └── eval/
│       ├── capitalization_epoch0.slurm
│       └── logs/
└── ... (7 more run directories)
```

## Next Steps

Execute via run-experiment skill or manually:

```bash
# Submit fine-tuning
for dir in rank*/; do (cd "$dir" && sbatch finetune.slurm); done

# After fine-tuning completes, submit evaluations
for dir in rank*/; do (cd "$dir/eval" && sbatch capitalization_epoch0.slurm); done
```

## Summary

✅ 8 fine-tuning runs configured and validated
✅ 8 evaluation scripts created
✅ Ready to execute workflow

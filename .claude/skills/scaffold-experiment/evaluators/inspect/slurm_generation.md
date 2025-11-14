# SLURM Script Generation

This module describes how to generate inspect.slurm evaluation scripts.

## Evaluation Naming Convention

**IMPORTANT: Epochs are 0-indexed**
- First epoch after training is `epoch_0`, not `epoch_1`
- Training for 1 epoch produces checkpoint at `epoch_0/`
- Training for 2 epochs produces `epoch_0/` and `epoch_1/`
- Evaluation script names must match: `{task_name}_epoch0.slurm`, not `epoch1`
- When experiment_summary.md says "evaluate last epoch after 1 epoch of training", use `epoch_0`

## Directory Organization

**For fine-tuned models:**
```
{experiment_dir}/{run_dir}/
├── finetune.slurm
├── finetune.yaml
├── setup_finetune.yaml
└── eval/
    ├── {task_name}_epoch0.slurm
    ├── {task_name}_epoch1.slurm
    └── logs/
```

**For base models (controls):**
```
{experiment_dir}/{run_dir}_base/
└── eval/
    ├── {task_name}_base.slurm
    └── logs/
```

## SLURM Script Template

Generate a SLURM script for each evaluation:

```bash
#!/bin/bash
#SBATCH --job-name=eval-{task_name}-{run_id}
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
{optional: #SBATCH --account={account}}
{optional: #SBATCH --constraint=gpu80}

# Load environment
module load anaconda3/2025.6
conda activate {conda_env}

# Set model and config paths
{if fine-tuned:}
MODEL_PATH="{output_dir_base}/ck-out-{run_name}/epoch_{N}"
CONFIG_PATH="{experiment_dir}/{run_dir}/setup_finetune.yaml"
{if base model:}
MODEL_PATH="{base_model_path}"
CONFIG_PATH="{experiment_dir}/{run_dir}/setup_finetune.yaml"

# Run inspect-ai evaluation
cd {experiment_dir}/{run_dir}/eval

{if fine-tuned with config_path:}
inspect eval {task_script_path}@{task_name} \\
  --model hf/local \\
  -M model_path="$MODEL_PATH" \\
  -T config_path="$CONFIG_PATH" \\
  --log-dir ./logs \\
  --log-level info

{if base model or direct dataset path:}
inspect eval {task_script_path}@{task_name} \\
  --model hf/local \\
  -M model_path="$MODEL_PATH" \\
  -T dataset_path="{eval_dataset_path}" \\
  -T system_prompt="{system_prompt}" \\
  --log-dir ./logs \\
  --log-level info

echo "Evaluation complete"
```

## Script Configuration

### SLURM Parameters

- **Time**: Default to 30 minutes (adjust based on experiment estimates if available)
- **GPUs**: 1 (evaluation is typically single-GPU)
- **Memory**: 32G (adjust based on model size if known)
- **Account/constraint**: Use from claude.local.md if specified

### Model Paths

- **Fine-tuned**: `{output_dir_base}/ck-out-{run_name}/epoch_{N}`
- **Base model**: Original model path from experiment_summary.md

### Task Parameters

- `config_path`: Path to setup_finetune.yaml (for both fine-tuned and base models)
- `dataset_path`: For base models or when explicit path needed
- `system_prompt`: Must match training configuration
- `temperature`: Typically 0.0 (may be task-specific)

### Output Location

- **Log directory**: `{run_dir}/eval/logs/`
- **SLURM output**: `{run_dir}/eval/slurm-{job_id}.out`

## Directory Structure Creation

Create eval directories as needed:

```bash
# For each run directory
mkdir -p {experiment_dir}/{run_dir}/eval
mkdir -p {experiment_dir}/{run_dir}/eval/logs

# Write SLURM script
cat > {experiment_dir}/{run_dir}/eval/{task_name}_epoch{N}.slurm << 'EOF'
{script content}
EOF

chmod +x {experiment_dir}/{run_dir}/eval/{task_name}_epoch{N}.slurm
```

## Important Notes

- Evaluation scripts should not be submitted until fine-tuning completes
- Model paths reference fine-tuning output directories that don't exist yet (created during training)
- System prompt consistency between training and evaluation is critical
- Evaluation logs will be written to `{run_dir}/eval/logs/` subdirectories

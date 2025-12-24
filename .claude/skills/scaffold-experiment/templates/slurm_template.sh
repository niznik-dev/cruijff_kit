#!/bin/bash
#SBATCH --job-name=eval-{task_name}-{run_id}
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_from_model_size}
#SBATCH --mem={mem_from_model_size}
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:{gpus_from_model_size}
# Optional: #SBATCH --account={account}
# Conditional: #SBATCH --constraint=gpu80 (required for 3B+ models)
#
# Model-aware resource allocation:
# | Model Size | Memory | GPUs | Constraint | CPUs |
# |------------|--------|------|------------|------|
# | 1B         | 32G    | 1    | -          | 4    |
# | 3B         | 64G    | 1    | gpu80      | 4    |
# | 8B         | 96G    | 1    | gpu80      | 8    |
# | 70B        | 256G   | 4    | gpu80      | 8    |

# Load environment
module load anaconda3/2025.6
conda activate {conda_env}

# CRITICAL: All paths MUST be absolute (start with /), never relative (../file)
MODEL_PATH="{model_path}"

# Dataset and prompt configuration
# (extracted from setup_finetune.yaml or experiment_summary.md at scaffolding time)
DATA_PATH="{data_path}"
PROMPT="{prompt}"
SYSTEM_PROMPT="{system_prompt}"

# Run inspect-ai evaluation
cd {experiment_dir}/{run_dir}/eval

inspect eval {task_script_path}@{task_name} \
  --model hf/local \
  -M model_path="$MODEL_PATH" \
  -T data_path="$DATA_PATH" \
  -T prompt="$PROMPT" \
  -T system_prompt="$SYSTEM_PROMPT" \
  --log-dir ./logs \
  --log-level info

echo "Evaluation complete"

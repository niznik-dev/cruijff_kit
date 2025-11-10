#!/bin/bash
#SBATCH --job-name=eval-{task_name}-{run_id}
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
# Optional: #SBATCH --account={account}
# Optional: #SBATCH --constraint=gpu80

# Load environment
module load anaconda3/2025.6
conda activate {conda_env}

# Set model and config paths
MODEL_PATH="{output_dir_base}/ck-out-{run_name}/epoch_{N}"
CONFIG_DIR="$MODEL_PATH"

# Run inspect-ai evaluation
cd {experiment_dir}/{run_dir}/eval

inspect eval {task_script_path}@{task_name} \
  --model hf/local \
  -M model_path="$MODEL_PATH" \
  -T config_dir="$CONFIG_DIR" \
  --log-dir ./logs \
  --log-level info

echo "Evaluation complete"

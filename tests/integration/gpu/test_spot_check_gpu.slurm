#!/bin/bash
#SBATCH --job-name=spot-check-test    # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                     # Memory allocation
#SBATCH --time=00:15:00               # Time limit (HH:MM:SS)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --output=spot_check_test_%j.out
##SBATCH --account=<ACT>
##SBATCH --partition=<PART>
##SBATCH --constraint=<CONST>

module purge
module load anaconda3/2025.6
conda activate cruijff

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

# Fill in paths before submitting:
#   --data  JSON file with "input"/"output" keys (e.g., {ck_data_dir}/folktexts/acs_income_verbose_50000_80P.json)
#   --base  Pretrained HF model directory (e.g., /path/to/pretrained-llms/Llama-3.2-3B-Instruct)
#   --ft    Fine-tuned checkpoint directory (e.g., /path/to/{run_name}/artifacts/epoch_2)
bash tests/integration/gpu/test_spot_check_gpu.sh \
    --data <DATA_PATH> \
    --base <BASE_MODEL_PATH> \
    --ft <FINETUNED_MODEL_PATH>

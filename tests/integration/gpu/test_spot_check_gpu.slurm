#!/bin/bash
#SBATCH --job-name=spot-check-test
#SBATCH --account=msalganik
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=0:15:00
#SBATCH --output=spot_check_test_%j.out

module load anaconda3/2025.6
conda activate cruijff

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

DATA=data/green/acs/acs_income_verbose_50000_80P.json
BASE=/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-3B-Instruct
FT=/scratch/gpfs/MSALGANIK/niznik/ck-outputs/acs_income_variable_order_2026-03-18/ck-out-Llama-3.2-3B-Instruct_original/epoch_2

bash tests/integration/gpu/test_spot_check_gpu.sh \
    --data "$DATA" \
    --base "$BASE" \
    --ft "$FT"

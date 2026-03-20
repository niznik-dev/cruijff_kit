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

cd /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit

DATA=/scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/data/green/acs/acs_income_verbose_50000_80P.json
BASE=/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-3B-Instruct
FT=/scratch/gpfs/MSALGANIK/niznik/ck-outputs/acs_income_variable_order_2026-03-18/ck-out-Llama-3.2-3B-Instruct_original/epoch_2

echo "========================================="
echo "Test 1: Fine-tuned 3B model"
echo "========================================="
python -m cruijff_kit.utils.spot_check \
    --model "$FT" \
    --data "$DATA" \
    --split validation \
    --n 10 \
    --max-new-tokens 5

echo ""
echo "========================================="
echo "Test 2: Base 3B model (no fine-tuning)"
echo "========================================="
python -m cruijff_kit.utils.spot_check \
    --model "$BASE" \
    --data "$DATA" \
    --split validation \
    --n 10 \
    --max-new-tokens 5

echo ""
echo "========================================="
echo "Test 3: Fine-tuned with --no-chat-template"
echo "========================================="
python -m cruijff_kit.utils.spot_check \
    --model "$FT" \
    --data "$DATA" \
    --split validation \
    --n 5 \
    --max-new-tokens 5 \
    --no-chat-template

echo ""
echo "========================================="
echo "Test 4: Fine-tuned with system prompt"
echo "========================================="
python -m cruijff_kit.utils.spot_check \
    --model "$FT" \
    --data "$DATA" \
    --split validation \
    --n 5 \
    --max-new-tokens 5 \
    --sysprompt "You are a helpful assistant that predicts income levels."

echo ""
echo "All spot_check GPU tests completed."

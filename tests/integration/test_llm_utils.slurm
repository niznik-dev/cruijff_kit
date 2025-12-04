#!/bin/bash
#SBATCH --job-name=test_llm_utils
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=nomig

# Integration tests for cruijff_kit.utils.llm_utils
# Requires GPU and access to base model

module purge
module load anaconda3/2025.6
conda activate cruijff

# Set PYTHONPATH to include the repo root
export PYTHONPATH="${SLURM_SUBMIT_DIR}/../.."

# Run pytest on the llm_utils integration tests
cd "${SLURM_SUBMIT_DIR}/../.."
pytest tests/integration/test_llm_utils.py -v --tb=short

echo "Tests complete"

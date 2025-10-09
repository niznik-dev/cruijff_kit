#!/bin/bash
#SBATCH --job-name=inspect # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G # Memory allocation
#SBATCH --time=00:15:00 # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin # Email when job starts
#SBATCH --mail-type=end # Email when job ends
#SBATCH --mail-user=<netid>@princeton.edu
#SBATCH --gres=gpu:1 # Request 1 GPU
##SBATCH --account=<ACT>
##SBATCH --partition=<PART>
##SBATCH --constraint=<CONST>

module purge
module load anaconda3/2024.10
conda activate ttenv

# NB - this will only work if finetune_filled.yaml still has the same output_dir and epochs fields from the run you want to inspect *and* will default to the final epoch; otherwise, edit these manually!
output_epoch=$(($(grep '^epochs:' ../../finetune_filled.yaml | awk '{print $2}') - 1))
output_dir=$(grep '^output_dir:' ../../finetune_filled.yaml | awk '{print $2}')epoch_${output_epoch}/
inspect eval eval_inspect.py --model hf/local -M model_path=${output_dir}
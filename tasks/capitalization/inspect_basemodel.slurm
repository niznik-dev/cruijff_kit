#!/bin/bash
#SBATCH --job-name=inspect # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G # Memory allocation
#SBATCH --time=00:15:00 # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin # Email when job starts
#SBATCH --mail-type=end # Email when job ends
#SBATCH --mail-user=mjs3@princeton.edu
#SBATCH --gres=gpu:1 # Request 1 GPU
##SBATCH --account=<ACT>
##SBATCH --partition=<PART>
##SBATCH --constraint=<CONST>

module purge
module load anaconda3/2024.10
conda activate ttenv

output_dir="/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-1B-Instruct"
inspect eval inspect.py --model hf/local -M model_path=${output_dir} -T config_dir=${output_dir}
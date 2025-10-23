#!/bin/bash
#SBATCH --job-name=save_hiddens # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=8 # 1/4 of CPU's so we don't overallocate jobs to a node
#SBATCH --mem=80G # Memory allocation
#SBATCH --time=48:00:00 # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin # Email when job starts
#SBATCH --mail-type=end # Email when job ends
#SBATCH --mail-user=<NETID>@princeton.edu
#SBATCH --gres=gpu:1 # Request 1 GPU
##SBATCH --account=<ACT>
##SBATCH --partition=<PART>
##SBATCH --constraint=<CONST>

module purge
module load anaconda3/2024.10
conda activate cruijff

python save_all_hiddens.py --config save_all_hiddens.yaml


#!/bin/bash
#SBATCH --job-name=save_hiddens # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=8 # 1/4 of CPU's so we don't overallocate jobs to a node
#SBATCH --mem=80G # Memory allocation
#SBATCH --time=48:00:00 # Time limit (HH:MM:SS)
#SBATCH --nodelist=ossc9424vm4



source $PROJECT/venvs/omni/bin/activate
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

python save_all_hiddens.py --config save_all_hiddens.yaml


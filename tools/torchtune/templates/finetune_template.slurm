#!/bin/bash
#SBATCH --job-name=<JOBNAME> # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=<MEM> # Memory allocation (RAM=VRAM rule)
#SBATCH --time=00:15:00 # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin # Email when job starts
#SBATCH --mail-type=end # Email when job ends
#SBATCH --mail-user=<NETID>@princeton.edu
#SBATCH --gres=gpu:1 # Request 1 GPU
##SBATCH --array=0-<ARRAY_MAX> # Uncomment if using job arrays
##SBATCH --account=<ACT>
##SBATCH --partition=<PART>
##SBATCH --constraint=<CONST>

module purge
module load anaconda3/2025.6
conda activate <CONDA_ENV>

mkdir -p <OUTPUT_DIR>logs/wandb  # wandb is picky about existing dirs
cp finetune.slurm setup_finetune.yaml <OUTPUT_DIR>/  # Copy config to output dir for reference

# Start GPU monitoring in background (filter to allocated GPU only)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 30 --id=$CUDA_VISIBLE_DEVICES > <OUTPUT_DIR>/gpu_metrics.csv 2>/dev/null &
GPU_MONITOR_PID=$!

tune run lora_finetune_single_device \
    --config finetune.yaml
TUNE_EXIT_CODE=$?

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null
wait $GPU_MONITOR_PID 2>/dev/null

[ $TUNE_EXIT_CODE == 0 ] && mv slurm-${SLURM_JOB_ID}.out <OUTPUT_DIR>/  # Move SLURM log to output dir if successful to avoid clutter
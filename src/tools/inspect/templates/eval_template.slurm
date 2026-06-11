#!/bin/bash
#SBATCH --job-name=<JOBNAME> # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=<MEM> # Memory allocation (RAM=VRAM rule)
#SBATCH --time=<TIME> # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin # Email when job starts
#SBATCH --mail-type=end # Email when job ends
#SBATCH --mail-user=<NETID>@princeton.edu
#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH --output=<GPU_METRICS_DIR>/slurm-%j.out
##SBATCH --account=<ACT>
##SBATCH --partition=<PART>
##SBATCH --constraint=<CONST>

module purge
module load anaconda3/2025.6
conda activate <CONDA_ENV>

# Start GPU monitoring in background
# SLURM cgroup isolation scopes nvidia-smi to the allocated GPU(s)
GPU_METRICS_DIR="<GPU_METRICS_DIR>"
mkdir -p "$GPU_METRICS_DIR"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv -l 30 > "${GPU_METRICS_DIR}/gpu_metrics.csv" 2>/dev/null &
GPU_MONITOR_PID=$!

# Prevent HF Hub lookups on offline compute nodes (inspect-ai >= 0.3.180)
export HF_HUB_OFFLINE=1

# Run inspect-ai evaluation
# Wrapped in /usr/bin/time -p as a format-stable wall-time anchor (issue #473):
# inspect-ai's own footer 'total time:' is the primary parse target, but
# /usr/bin/time -p output ('real <seconds>') is a redundant fallback that
# survives any upstream footer format change.
cd <EVAL_DIR>

/usr/bin/time -p inspect eval <TASK_SCRIPT> \
  --model <MODEL_HF_NAME> \
  -M model_path="<MODEL_PATH>" \
<DOSAMPLE_ARGS><DEVICE_ARGS><TASK_ARGS><METADATA_ARGS>  --max-connections <MAX_CONNECTIONS> \
  --seed <SEED> \
  --log-dir ./logs \
  --log-level info
INSPECT_EXIT_CODE=$?

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null
wait $GPU_MONITOR_PID 2>/dev/null

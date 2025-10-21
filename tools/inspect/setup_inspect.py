import argparse
import os
import shutil
import sys
from pathlib import Path

# Calculate paths relative to this script
script_dir = Path(__file__).parent

parser = argparse.ArgumentParser(description="Set up the run_inspect.py file for the current project.")
parser.add_argument("--base_model_dir", type=str, default="", help="Path to a base model directory")
parser.add_argument("--finetune_epoch_dir", type=str, required=True, help="Path to a finetuned model's target epoch (and its slurm parameters one folder up); if base_model_dir is provided, only the slurm parameters are used")
args = parser.parse_args()

# Decide where we'll load the model from
MODEL_PATH = args.base_model_dir if args.base_model_dir else args.finetune_epoch_dir
with open(f"{args.finetune_epoch_dir}/../finetune.slurm", "r") as f:
    slurm_script = f.read()

slurm_script = slurm_script.replace(" # Job name", "-i # Job name")

# Remove all lines after the conda environment is activated
slurm_script = slurm_script.split("mkdir -p")[0]

slurm_script += f"inspect eval run_inspect.py --model hf/local -M model_path={MODEL_PATH} -T config_dir={args.finetune_epoch_dir}"

with open("inspect.slurm", "w") as f:
    f.write(slurm_script)

print("Ready to inspect with `sbatch inspect.slurm` !")
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Calculate paths relative to this script
script_dir = Path(__file__).parent

# If possible, see if the input file comes from one of our task directories and if so grab the inspect.py from there
parser = argparse.ArgumentParser(description="Set up the inspect.py file for the current project.")
parser.add_argument("--base_model_dir", type=str, default="", help="Path to a base model directory")
parser.add_argument("--finetune_epoch_dir", type=str, required=True, help="Path to a finetuned model's target epoch (and its slurm parameters one folder up); if base_model_dir is provided, only the slurm parameters are used")
args = parser.parse_args()

# Decide where we'll load the model from
MODEL_PATH = args.base_model_dir if args.base_model_dir else args.finetune_epoch_dir

# Fix adapter_config.json if it's missing base_model_name_or_path
adapter_config_path = Path(args.finetune_epoch_dir) / "adapter_config.json"
if adapter_config_path.exists():
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)

    # Check if base_model_name_or_path is missing
    if 'base_model_name_or_path' not in adapter_config:
        # Try to infer from finetune.yaml
        finetune_yaml_path = Path(args.finetune_epoch_dir).parent / "finetune.yaml"
        if finetune_yaml_path.exists():
            import yaml
            with open(finetune_yaml_path, 'r') as f:
                finetune_config = yaml.safe_load(f)

            # Extract base model path from checkpointer config
            if 'checkpointer' in finetune_config and 'checkpoint_dir' in finetune_config['checkpointer']:
                base_model_path = finetune_config['checkpointer']['checkpoint_dir']

                # Resolve ${models_dir} variable if present
                if '${models_dir}' in base_model_path and 'models_dir' in finetune_config:
                    base_model_path = base_model_path.replace('${models_dir}', finetune_config['models_dir'])

                # Clean up double slashes and trailing slashes
                base_model_path = base_model_path.replace('//', '/').rstrip('/')

                adapter_config['base_model_name_or_path'] = base_model_path

                # Write updated config
                with open(adapter_config_path, 'w') as f:
                    json.dump(adapter_config, f, indent=2)
                print(f"✓ Fixed adapter_config.json: Added base_model_name_or_path = {base_model_path}")

with open(f"{args.finetune_epoch_dir}/../finetune.slurm", "r") as f:
    slurm_script = f.read()

slurm_script = slurm_script.replace(" # Job name", "-i # Job name")

# Remove all lines after the conda environment is activated
slurm_script = slurm_script.split("mkdir -p")[0]

# Check if eval.py exists in the parent directory, use it instead of inspect.py
experiment_dir = Path(args.finetune_epoch_dir).parent
eval_script = "eval.py" if (experiment_dir / "eval.py").exists() else "inspect.py"

slurm_script += f"inspect eval {eval_script} --model hf/local -M model_path={MODEL_PATH} -T config_dir={args.finetune_epoch_dir}"

with open("inspect.slurm", "w") as f:
    f.write(slurm_script)

print(f"✓ Created inspect.slurm using {eval_script}")
print("Ready to inspect with `sbatch inspect.slurm` !")
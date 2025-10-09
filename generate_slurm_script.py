import argparse
import json
import os
import yaml

from utils import run_names

RANDOM_MODEL_RUN_NAME = run_names.generate_model_run_name()[0]

# Skip these when writing the yaml file
SLURM_ONLY = ['time', 'gpus', 'conda_env', 'account', 'partition', 'constraint']

# Used for epochs_to_save to allow for all/none options
def parse_epochs(value):
    if value.lower() == 'none':
        return []
    elif value.lower() == 'all':
        return 'all'
    else:
        try:
            return [int(x.strip()) for x in value.split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid epochs format: {value}")


parser = argparse.ArgumentParser()

# ----- Config File -----
parser.add_argument("--generate_config", type=str, default="total_config.yaml", help="Path to YAML configuration file. Values from this file will be used as defaults, and can be overridden by CLI arguments.")

# ----- Required YAML Args Reused in Templating -----
parser.add_argument("--my_wandb_project", type=str, default="PredictingZygosity", help="Project for when results are synced to wandb")
parser.add_argument("--my_wandb_run_name", type=str, help="Name for when results are synced to wandb; if not provided, a random name will be generated")
parser.add_argument("--input_formatting", type=str, default="raw", help="Name of the folder where your input files are stored within input_dir; useful for multiple formatting styles (e.g. difference vs raw values). If same directory, set to empty string.")

parser.add_argument("--dataset_filename", type=str, default="tune_dataset", help="Name of the HF dataset folder or JSON file (should be in input_dir)")

parser.add_argument("--output_dir_base", type=str, default="/scratch/gpfs/MSALGANIK/$USER/", help="Full path to the output file folders (final output folder will be 'ck-out-' + my_wandb_name within this folder)")
parser.add_argument("--input_dir_base", type=str, default="/scratch/gpfs/MSALGANIK/$USER/zyg_in/", help="Full path to the input file folders")
parser.add_argument("--models_dir", type=str, default="/scratch/gpfs/MSALGANIK/pretrained-llms/", help="Full path to the model file folders")

# ----- Optional YAML Args -----
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
parser.add_argument("--save_adapter_weights_only", type=str, default="false", help="Whether to save only the adapter weights (true/false)")
parser.add_argument("--save_last_epoch_only", type=str, default="false", help="Whether to save only the last epoch (true/false)")
parser.add_argument("--stash_adapter_weights", type=str, default="false", help="Whether to stash adapter files in subdirectory to avoid confusing inspect-ai (true/false)")
parser.add_argument("--epochs_to_save", type=parse_epochs, default="all", help="Comma delimited epochs to save checkpoints at; can also be 'all' or 'none'.")
parser.add_argument("--max_steps_per_epoch", type=int, help="Maximum steps per epoch (useful for debugging)")
parser.add_argument("--log_every_n_steps", type=int, default=5, help="How often to log (in steps)")
parser.add_argument("--run_val_every_n_steps", type=int, default=0, help="How often to run validation (in steps)")
parser.add_argument("--system_prompt", type=str, default="", help="System prompt to use (if any)")
parser.add_argument("--train_on_input", type=str, default="false", help="Whether to train on the input data (true/false)")

# ------ Slurm Args -----
parser.add_argument("--time", type=str, default="00:15:00", help="Time to run the job (HH:MM:SS)")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("--conda_env", type=str, default="ttenv", help="Name of the conda environment to use")
parser.add_argument("--venv", type=str, default="", help="Path to the virtual environment to use (if not using conda)")
parser.add_argument("--modules", type=str, default="", help="Modules to load before running the script, separated by commas (e.g. '2024,Python/3.12.3')")

parser.add_argument("--account", type=str, help="Slurm account to use")
parser.add_argument("--partition", type=str, help="Slurm partition to use")
parser.add_argument("--constraint", type=str, help="Slurm constraint to use")

parser.add_argument("--custom_recipe", type=str, help="Full name of a custom recipe file in the repo's custom_recipes folder to use for fine-tuning")

args = parser.parse_args()

# Load config file if it exists and merge with CLI arguments
config_data = {}
if args.generate_config and os.path.exists(args.generate_config):
    with open(args.generate_config, "r") as f:
        config_data = yaml.safe_load(f) or {}

    # For each argument, use CLI value if provided, otherwise use config file value
    for key, value in config_data.items():
        # Only use config value if the argument wasn't explicitly provided on CLI
        # We check if it's still at its default value
        if hasattr(args, key):
            default_value = parser.get_default(key)
            current_value = getattr(args, key)
            # If current value equals default, use config file value
            if current_value == default_value:
                setattr(args, key, value)

model_run_name = args.my_wandb_run_name if args.my_wandb_run_name else RANDOM_MODEL_RUN_NAME
username = os.environ.get("USER")

# First edit the yaml template
with open("templates/finetune_template.yaml", "r") as f:
    config = yaml.safe_load(f)

for key, value in vars(args).items():
    if key in SLURM_ONLY:
        continue
    # Special cases first
    elif key == "my_wandb_run_name":
        config["my_wandb_run_name"] = model_run_name
    elif key == "input_dir_base":
        config["input_dir"] = value + args.input_formatting + ("/" if args.input_formatting else "")
    elif key == "output_dir_base":
        full_output_dir = value + "ck-out-" + model_run_name + "/"
        config["output_dir"] = full_output_dir
    elif key == "dataset_filename":
        config["dataset_filename"] = value
        if value.endswith('.json'):
            # Override to use JSON format instead of Parquet
            config["dataset"]["source"] = "json"
            config["dataset"]["data_files"] = config["dataset"].pop("data_dir")
            config["dataset"]["field"] = config["dataset"].pop("split")
            config["dataset_val"]["source"] = "json"
            config["dataset_val"]["data_files"] = config["dataset_val"].pop("data_dir")
            config["dataset_val"]["field"] = config["dataset_val"].pop("split")
    elif key == "system_prompt":
        if value:
            config["dataset"]["new_system_prompt"] = value
            config["dataset_val"]["new_system_prompt"] = value
    # TODO - change these to actual booleans in argparse?
    elif key == "save_adapter_weights_only":
        config["save_adapter_weights_only"] = (value == "true")
    elif key == "save_last_epoch_only":
        config["save_last_epoch_only"] = (value == "true")
    elif key == "stash_adapter_weights":
        config["stash_adapter_weights"] = (value == "true")
    elif key == "train_on_input":
        config["dataset"]["train_on_input"] = (value == "true")
    # The rest are straightforward
    else:
        config[key] = value

if config["run_val_every_n_steps"] == 0:
    # Remove all validation-related keys if not running validation
    config.pop("dataset_val", None)
    config.pop("run_val_every_n_steps", None)

for key in ['input_dir', 'output_dir', 'models_dir']:
    config[key] = config[key].replace("$USER", username)

with open("finetune_filled.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False)

# Now create the slurm script
with open("templates/finetune_template.slurm", "r") as f:
    slurm_script = f.read()

slurm_script = slurm_script.replace("<JOBNAME>", model_run_name)
# TODO - lookup reasonable memory/time values based on model choice (create a table somewhere)
slurm_script = slurm_script.replace("00:15:00", args.time)
slurm_script = slurm_script.replace("<NETID>", username)

if args.gpus > 1:
    slurm_script = slurm_script.replace("#SBATCH --cpus-per-task=1", "#SBATCH --cpus-per-task=" + str(args.gpus))
    slurm_script = slurm_script.replace("#SBATCH --gres=gpu:1", "#SBATCH --gres=gpu:" + str(args.gpus))
    slurm_script = slurm_script.replace("lora_finetune_single_device", "--nproc_per_node=" + str(args.gpus) + " lora_finetune_distributed")
if args.account:
    slurm_script = slurm_script.replace("##SBATCH --account=<ACT>", "#SBATCH --account=" + args.account)
if args.partition:
    slurm_script = slurm_script.replace("##SBATCH --partition=<PART>", "#SBATCH --partition=" + args.partition)
if args.constraint:
    slurm_script = slurm_script.replace("##SBATCH --constraint=<CONST>", "#SBATCH --constraint=" + args.constraint)
if args.custom_recipe:
    if args.gpus == 1:
        slurm_script = slurm_script.replace("lora_finetune_single_device", 'custom_recipes/' + args.custom_recipe)
    else:
        slurm_script = slurm_script.replace("lora_finetune_distributed", 'custom_recipes/' + args.custom_recipe)

slurm_script = slurm_script.replace("<CONDA_ENV>", args.conda_env)
if args.venv:
    slurm_script = slurm_script.replace(f"conda activate {args.conda_env}", f"source $PROJECT/venvs/{args.venv}/bin/activate")
if args.modules:
    slurm_script = "\n".join(
        line for line in slurm_script.splitlines()
        if "conda" not in line
    )
    module_string = ''
    for m in args.modules.split(','):
        module_string += f"\nmodule load {m.strip()}"
    slurm_script = slurm_script.replace("module purge", "module purge" + module_string)

slurm_script = slurm_script.replace("<OUTPUT_DIR>", full_output_dir)
slurm_script = slurm_script.replace("$USER", username)

with open("finetune_filled.slurm", "w") as f:
    f.write(slurm_script)
import argparse
import json
import os
import yaml
from pathlib import Path

from cruijff_kit.utils import run_names

# Calculate paths relative to this script
script_dir = Path(__file__).parent

# Skip these when writing the yaml file
SLURM_ONLY = ['time', 'gpus', 'conda_env', 'account', 'partition', 'constraint', 'mem']

# Model-specific configurations
# Keys are model directory names (e.g., "Llama-3.2-3B-Instruct")
# SLURM resources follow RAM=VRAM rule to ensure checkpoint saving doesn't OOM
MODEL_CONFIGS = {
    "Llama-3.2-1B-Instruct": {
        "component": "torchtune.models.llama3_2.lora_llama3_2_1b",
        "checkpoint_files": ["model.safetensors"],
        "model_type": "LLAMA3_2",
        "slurm": {
            "mem": "40G",
            "partition": "nomig",  # Avoid MIG partitions by default
            "constraint": None,
            "cpus": 4,
            "gpus": 1,
        },
    },
    "Llama-3.2-3B-Instruct": {
        "component": "torchtune.models.llama3_2.lora_llama3_2_3b",
        "checkpoint_files": {
            "filename_format": "model-{}-of-{}.safetensors",
            "max_filename": "00002",
        },
        "model_type": "LLAMA3_2",
        "slurm": {
            "mem": "80G",
            "partition": None,
            "constraint": "gpu80",
            "cpus": 4,
            "gpus": 1,
        },
    },
    "Llama-3.3-70B-Instruct": {
        "component": "torchtune.models.llama3_3.lora_llama3_3_70b",
        "checkpoint_files": {
            "filename_format": "model-{}-of-{}.safetensors",
            "max_filename": "00030",
        },
        "model_type": "LLAMA3",
        "slurm": {
            "mem": "320G",
            "partition": None,
            "constraint": "gpu80",
            "cpus": 16,
            "gpus": 4,
        },
    },
}

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


# Used for boolean arguments to accept flexible input
def parse_bool(value):
    """Parse boolean values from strings or booleans.

    Args:
        value: String or boolean to parse

    Returns:
        Boolean value

    Raises:
        argparse.ArgumentTypeError: If value cannot be parsed as boolean

    Accepts: true/false, True/False, 1/0, yes/no (case-insensitive)
    """
    if isinstance(value, bool):
        return value  # Already a boolean (e.g., from YAML config)

    value_lower = str(value).lower()
    if value_lower in ('true', '1', 'yes'):
        return True
    elif value_lower in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Boolean value expected. Got: '{value}'. "
            "Valid values: true/false, yes/no, 1/0 (case-insensitive)"
        )


def calculate_lora_alpha(lora_rank):
    """Calculate LoRA alpha from LoRA rank.

    Args:
        lora_rank: The LoRA rank value

    Returns:
        LoRA alpha value (always 2 * rank)
    """
    return lora_rank * 2


def validate_lr_scheduler(scheduler_name):
    """Validate learning rate scheduler name.

    Args:
        scheduler_name: Name of the learning rate scheduler (without prefix)

    Raises:
        ValueError: If scheduler_name is not in the list of valid schedulers
    """
    VALID_LR_SCHEDULERS = [
        'get_cosine_schedule_with_warmup',
        'get_linear_schedule_with_warmup',
        'get_constant_schedule_with_warmup',
        'get_exponential_schedule_with_warmup'
    ]

    if scheduler_name not in VALID_LR_SCHEDULERS:
        raise ValueError(
            f"Invalid lr_scheduler: '{scheduler_name}'. "
            f"Must be one of: {', '.join(VALID_LR_SCHEDULERS)}"
        )


def validate_dataset_type(dataset_type):
    """Validate dataset type.

    Args:
        dataset_type: Name of the dataset type (without prefix)

    Raises:
        ValueError: If dataset_type is not in the list of valid types
    """
    VALID_DATASET_TYPES = [
        'instruct_dataset',
        'chat_dataset',
        'text_completion_dataset'
    ]

    if dataset_type not in VALID_DATASET_TYPES:
        raise ValueError(
            f"Invalid dataset_type: '{dataset_type}'. "
            f"Must be one of: {', '.join(VALID_DATASET_TYPES)}"
        )


def construct_output_dir(output_dir_base, experiment_name, model_run_name):
    """Construct the full output directory path.

    Args:
        output_dir_base: Base directory for outputs
        experiment_name: Optional experiment name for grouping outputs
        model_run_name: Name of this specific model run

    Returns:
        Full output directory path with trailing slash
    """
    # Ensure output_dir_base ends with /
    if not output_dir_base.endswith('/'):
        output_dir_base += '/'

    # If experiment_name is provided, group outputs under that directory
    if experiment_name:
        return output_dir_base + experiment_name + "/ck-out-" + model_run_name + "/"
    else:
        # Backwards compatibility: outputs go directly to output_dir_base
        return output_dir_base + "ck-out-" + model_run_name + "/"


def configure_dataset_for_format(config, dataset_label, dataset_ext, dataset_type):
    """Configure dataset paths and structure based on file format and type.

    Args:
        config: The configuration dictionary to modify
        dataset_label: Name of the dataset file (without extension) or folder
        dataset_ext: Extension of the dataset file (e.g., '.json' or '.parquet')
        dataset_type: Type of dataset ('instruct_dataset', 'chat_dataset', etc.)

    Returns:
        Modified configuration dictionary
    """
    config["dataset_label"] = dataset_label

    if dataset_ext == '.parquet':
        # For parquet, add filenames inside the folder (dataset_label is the folder name)
        config["dataset"]["data_dir"] += '/train.parquet'
        if "dataset_val" in config:
            config["dataset_val"]["data_dir"] += '/validation.parquet'

    elif dataset_ext == '.json':
        # Change source and rename data_dir to data_files
        config["dataset"]["source"] = "json"
        config["dataset"]["data_files"] = config["dataset"].pop("data_dir")
        if "dataset_val" in config:
            config["dataset_val"]["source"] = "json"
            config["dataset_val"]["data_files"] = config["dataset_val"].pop("data_dir")

        if dataset_type == 'instruct_dataset':
            # For instruct, use a single file and change split to field
            config["dataset"]["data_files"] += '.json'
            config["dataset"]["field"] = config["dataset"].pop("split")
            if "dataset_val" in config:
                config["dataset_val"]["data_files"] += '.json'
                config["dataset_val"]["field"] = config["dataset_val"].pop("split")
        else:
            # For chat, remove split and add filenames inside the folder
            config["dataset"]["data_files"] += '/train.json'
            config["dataset"].pop("split")
            if "dataset_val" in config:
                config["dataset_val"]["data_files"] += '/validation.json'
                config["dataset_val"].pop("split")

    return config


def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser()

    # ----- Config File -----
    parser.add_argument("--config_file", type=str, default="setup_finetune.yaml", help="Path to YAML configuration file. Values from this file will be used as defaults, and can be overridden by CLI arguments.")

    # ----- Required YAML Args Reused in Templating -----
    parser.add_argument("--my_wandb_project", type=str, default="PredictingZygosity", help="Project for when results are synced to wandb")
    parser.add_argument("--my_wandb_run_name", type=str, help="Name for when results are synced to wandb; if not provided, a random name will be generated")
    parser.add_argument("--input_formatting", type=str, default="raw", help="Name of the folder where your input files are stored within input_dir; useful for multiple formatting styles (e.g. difference vs raw values). If same directory, set to empty string.")

    parser.add_argument("--dataset_label", type=str, default="tune_dataset", help="Name of the dataset file (without extension) or folder (either should be in input_dir)")
    parser.add_argument("--dataset_ext", type=str, default="", help="Extension of the dataset file (e.g. .json or .parquet)")

    parser.add_argument("--experiment_name", type=str, default="", help="Name of the experiment/sanity_check (used to group outputs in ck-outputs/{experiment_name}/). If not provided, outputs go directly to output_dir_base.")
    parser.add_argument("--output_dir_base", type=str, default="/scratch/gpfs/MSALGANIK/$USER/", help="Full path to the output file folders (final output folder will be 'ck-out-' + my_wandb_name within this folder)")
    parser.add_argument("--input_dir_base", type=str, default="/scratch/gpfs/MSALGANIK/$USER/zyg_in/", help="Full path to the input file folders")
    parser.add_argument("--models_dir", type=str, default="/scratch/gpfs/MSALGANIK/pretrained-llms/", help="Full path to the model file folders")

    # ----- Optional YAML Args -----
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--save_adapter_weights_only", type=parse_bool, default=False, help="Whether to save only the adapter weights (true/false)")
    parser.add_argument("--save_last_epoch_only", type=parse_bool, default=False, help="Whether to save only the last epoch (true/false)")
    parser.add_argument("--stash_adapter_weights", type=parse_bool, default=False, help="Whether to stash adapter files in subdirectory to avoid confusing inspect-ai (true/false)")
    parser.add_argument("--epochs_to_save", type=parse_epochs, default="all", help="Comma delimited epochs to save checkpoints at; can also be 'all' or 'none'.")
    parser.add_argument("--max_steps_per_epoch", type=int, help="Maximum steps per epoch (useful for debugging)")
    parser.add_argument("--log_every_n_steps", type=int, default=5, help="How often to log (in steps)")
    parser.add_argument("--run_val_every_n_steps", type=int, default=0, help="How often to run validation (in steps)")
    parser.add_argument("--system_prompt", type=str, default="", help="System prompt to use (if any)")
    parser.add_argument("--train_on_input", type=parse_bool, default=False, help="Whether to train on the input data (true/false)")

    # ------ Model/Training Args -----
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank (alpha will be auto-calculated as 2*rank)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--lr_scheduler", type=str, default="get_cosine_schedule_with_warmup", help="Learning rate scheduler function name (without 'torchtune.training.lr_schedulers.' prefix)")
    parser.add_argument("--dataset_type", type=str, default="instruct_dataset", help="Dataset type function name (without 'torchtune.datasets.' prefix)")

    # ------ Slurm Args -----
    parser.add_argument("--time", type=str, default="00:15:00", help="Time to run the job (HH:MM:SS)")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--conda_env", type=str, default="cruijff", help="Name of the conda environment to use")
    parser.add_argument("--venv", type=str, default="", help="Path to the virtual environment to use (if not using conda)")
    parser.add_argument("--modules", type=str, default="", help="Modules to load before running the script, separated by commas (e.g. '2024,Python/3.12.3')")

    parser.add_argument("--account", type=str, help="Slurm account to use")
    parser.add_argument("--partition", type=str, help="Slurm partition to use")
    parser.add_argument("--mem", type=str, help="Slurm memory allocation (e.g., '40G', '16G')")
    parser.add_argument("--constraint", type=str, help="Slurm constraint to use")

    parser.add_argument("--custom_recipe", type=str, help="Full name of a custom recipe file in the repo's custom_recipes folder to use for fine-tuning")

    # ------ Model Selection -----
    parser.add_argument("--torchtune_model_name", type=str, default="Llama-3.2-1B-Instruct",
                        help="Model name as listed by 'tune ls' (e.g., 'Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Llama-3.3-70B-Instruct')")
    parser.add_argument("--model_checkpoint", type=str, default=None,
                        help="Model directory name within models_dir (defaults to torchtune_model_name if not provided)")

    return parser


def main():
    """Main function to set up fine-tuning configuration and SLURM script."""
    parser = create_parser()
    args = parser.parse_args()

    RANDOM_MODEL_RUN_NAME = run_names.generate_model_run_name()[0]

    # Load config file if it exists and merge with CLI arguments
    config_data = {}
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, "r") as f:
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

    # Validate lr_scheduler and dataset_type (after config file has been loaded and merged)
    validate_lr_scheduler(args.lr_scheduler)
    validate_dataset_type(args.dataset_type)

    model_run_name = args.my_wandb_run_name if args.my_wandb_run_name else RANDOM_MODEL_RUN_NAME
    username = os.environ.get("USER")

    # Get model config early (needed for both YAML and SLURM generation)
    if args.torchtune_model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: '{args.torchtune_model_name}'. "
            f"Supported models: {', '.join(MODEL_CONFIGS.keys())}"
        )
    model_config = MODEL_CONFIGS[args.torchtune_model_name]

    # First edit the yaml template
    with open(f"{script_dir}/templates/finetune_template.yaml", "r") as f:
        config = yaml.safe_load(f)

    for key, value in vars(args).items():
        if key in SLURM_ONLY:
            continue
        # Model config - apply torchtune-specific settings
        elif key == "torchtune_model_name":
            # Model directory: use model_checkpoint if provided, otherwise torchtune_model_name
            model_dir = args.model_checkpoint if args.model_checkpoint else value

            # Set model component
            config["model"]["_component_"] = model_config["component"]

            # Set tokenizer path
            config["tokenizer"]["path"] = f"${{models_dir}}/{model_dir}/original/tokenizer.model"

            # Set checkpointer config
            config["checkpointer"]["checkpoint_dir"] = f"${{models_dir}}/{model_dir}/"
            config["checkpointer"]["model_type"] = model_config["model_type"]
            config["checkpointer"]["checkpoint_files"] = model_config["checkpoint_files"]
        elif key == "model_checkpoint":
            continue  # Handled in torchtune_model_name
        # Special cases first
        elif key == "my_wandb_run_name":
            config["my_wandb_run_name"] = model_run_name
        elif key == "input_dir_base":
            config["input_dir"] = value + args.input_formatting + ("/" if args.input_formatting else "")
        elif key == "output_dir_base":
            full_output_dir = construct_output_dir(value, args.experiment_name, model_run_name)
            config["output_dir"] = full_output_dir
        elif key == "experiment_name":
            pass  # Handled in output_dir_base
        elif key == "dataset_label":
            config = configure_dataset_for_format(config, value, args.dataset_ext, args.dataset_type)
        elif key == "dataset_ext":
            pass  # Handled in dataset_label
        elif key == "system_prompt":
            if value:
                config["dataset"]["new_system_prompt"] = value
                config["dataset_val"]["new_system_prompt"] = value
        elif key == "train_on_input":
            # Special case: nested in dataset config
            config["dataset"]["train_on_input"] = value
        elif key == "lora_rank":
            # Set both rank and alpha (alpha = 2 * rank)
            config["model"]["lora_rank"] = value
            config["model"]["lora_alpha"] = calculate_lora_alpha(value)
        elif key == "lr":
            config["optimizer"]["lr"] = value
        elif key == "num_warmup_steps":
            config["lr_scheduler"]["num_warmup_steps"] = value
        elif key == "lr_scheduler":
            # Construct full component path
            config["lr_scheduler"]["_component_"] = f"torchtune.training.lr_schedulers.{value}"
        elif key == "dataset_type":
            config["dataset"]["_component_"] = f"torchtune.datasets.{value}"
            if value == 'chat_dataset':
                config["dataset"]["conversation_column"] = "messages"
                config["dataset"]["conversation_style"] = "openai"
            if "dataset_val" in config:
                config["dataset_val"]["_component_"] = f"torchtune.datasets.{value}"
                if value == 'chat_dataset':
                    config["dataset_val"]["conversation_column"] = "messages"
                    config["dataset_val"]["conversation_style"] = "openai"
        # The rest are straightforward
        else:
            config[key] = value

    if config["run_val_every_n_steps"] == 0:
        # Remove all validation-related keys if not running validation
        config.pop("dataset_val", None)
        config.pop("run_val_every_n_steps", None)

    for key in ['input_dir', 'output_dir', 'models_dir']:
        config[key] = config[key].replace("$USER", username)

    with open("finetune.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    # Now create the slurm script
    with open(f"{script_dir}/templates/finetune_template.slurm", "r") as f:
        slurm_script = f.read()

    slurm_script = slurm_script.replace("<JOBNAME>", model_run_name)
    slurm_script = slurm_script.replace("00:15:00", args.time)
    slurm_script = slurm_script.replace("<NETID>", username)

    # Apply model-aware SLURM resources (RAM=VRAM rule)
    # User-specified mem overrides model defaults (for MIG support)
    slurm_config = model_config.get("slurm", {})
    mem = args.mem if args.mem else slurm_config.get("mem", "32G")
    slurm_script = slurm_script.replace("<MEM>", mem)

    # GPUs: CLI overrides model config default
    # Check if user explicitly set --gpus (compare to parser default of 1)
    model_gpus = slurm_config.get("gpus", 1)
    gpus = args.gpus if args.gpus != 1 else model_gpus

    # CPUs: use model config value
    cpus = slurm_config.get("cpus", 4)

    # Multi-GPU setup: update SLURM and use distributed training
    if gpus > 1:
        slurm_script = slurm_script.replace("#SBATCH --gres=gpu:1", "#SBATCH --gres=gpu:" + str(gpus))
        slurm_script = slurm_script.replace("lora_finetune_single_device", "--nproc_per_node=" + str(gpus) + " lora_finetune_distributed")
    slurm_script = slurm_script.replace("#SBATCH --cpus-per-task=1", "#SBATCH --cpus-per-task=" + str(cpus))

    if args.account:
        slurm_script = slurm_script.replace("##SBATCH --account=<ACT>", "#SBATCH --account=" + args.account)

    # Partition: CLI/yaml overrides model config
    # Use 'is not None' check because empty string is valid (MIG support)
    partition = args.partition if args.partition is not None else slurm_config.get("partition")
    if partition is not None and partition != "":
        slurm_script = slurm_script.replace("##SBATCH --partition=<PART>", "#SBATCH --partition=" + partition)

    # Constraint: CLI overrides model config
    constraint = args.constraint if args.constraint else slurm_config.get("constraint")
    if constraint:
        slurm_script = slurm_script.replace("##SBATCH --constraint=<CONST>", "#SBATCH --constraint=" + constraint)
    if args.custom_recipe:
        if gpus == 1:
            slurm_script = slurm_script.replace("lora_finetune_single_device", args.custom_recipe + '.__main__')
        else:
            slurm_script = slurm_script.replace("lora_finetune_distributed", args.custom_recipe + '.__main__')

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

    with open("finetune.slurm", "w") as f:
        f.write(slurm_script)


if __name__ == "__main__":
    main()
import argparse
import json
import math
import os
import warnings
import yaml
from pathlib import Path

from cruijff_kit.utils import run_names
from cruijff_kit.tools.torchtune import config_recipe_loader
from cruijff_kit.tools.torchtune.model_configs import (
    MODEL_CONFIGS,
    configure_tokenizer
)

# Calculate paths relative to this script
script_dir = Path(__file__).parent

# Skip these when writing the yaml file
SLURM_ONLY = ['time', 'gpus', 'conda_env', 'account', 'partition', 'constraint', 'mem']
# Meta-arguments that are not torchtune config parameters
META_ARGS = ['training_samples']

# Maps torchtune recipe config paths to setup_finetune.py argument names
RECIPE_PARAM_MAPPING = {
    'model.lora_rank': 'lora_rank',
    'model.lora_dropout': 'lora_dropout',
    'optimizer.lr': 'lr',
    'optimizer.weight_decay': 'weight_decay',
    'batch_size': 'batch_size',
    'epochs': 'epochs',
    'gradient_accumulation_steps': 'gradient_accumulation_steps',
    'lr_scheduler.num_warmup_steps': 'num_warmup_steps',
    'tokenizer.max_seq_len': 'max_seq_len',
}


def extract_flat_params(recipe_config: dict, mapping: dict) -> dict:
    """Extract parameters from nested recipe config using mapping.

    Args:
        recipe_config: Nested dictionary from torchtune recipe YAML
        mapping: Dict mapping recipe paths (e.g., 'model.lora_rank') to arg names

    Returns:
        Flat dictionary of {arg_name: value} for parameters found in recipe
    """
    flat = {}
    for recipe_path, arg_name in mapping.items():
        parts = recipe_path.split('.')
        value = recipe_config
        try:
            for part in parts:
                value = value[part]
            flat[arg_name] = value
        except (KeyError, TypeError):
            continue  # Parameter not in recipe
    return flat

def compute_training_steps(training_samples, batch_size, gradient_accumulation_steps, epochs):
    """Compute total training steps and steps per epoch.

    Args:
        training_samples: Number of training samples in the dataset
        batch_size: Per-device batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        epochs: Number of training epochs

    Returns:
        Dict with 'steps_per_epoch', 'total_steps', and 'effective_batch_size'
    """
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(training_samples / effective_batch_size)
    total_steps = steps_per_epoch * epochs
    return {
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'effective_batch_size': effective_batch_size,
    }


def warn_on_low_steps(step_info, num_warmup_steps):
    """Print step summary and emit warnings if training steps are dangerously low.

    Args:
        step_info: Dict from compute_training_steps()
        num_warmup_steps: Number of warmup steps configured for the LR scheduler
    """
    total = step_info['total_steps']
    print(f"Training step summary: {step_info['steps_per_epoch']} steps/epoch, "
          f"{total} total steps (effective batch size: {step_info['effective_batch_size']})")

    if total < num_warmup_steps:
        warnings.warn(
            f"Total training steps ({total}) < warmup steps ({num_warmup_steps}). "
            f"Warmup will never complete — the learning rate will never reach its target value.",
            stacklevel=2,
        )
    min_recommended = 3 * num_warmup_steps
    if total < min_recommended:
        warnings.warn(
            f"Total training steps ({total}) < {min_recommended} "
            f"(3x warmup steps of {num_warmup_steps}). "
            f"The model will spend most of training in warmup with little time at full learning rate. "
            f"Consider reducing batch size or gradient accumulation, or increasing epochs.",
            stacklevel=2,
        )


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
        dataset_type: Name of the dataset type

    Raises:
        ValueError: If dataset_type is not in the list of valid types
    """
    VALID_DATASET_TYPES = [
        'chat_completion',  # Instruct models - uses HF apply_chat_template for train/eval parity
        'text_completion',  # Base models - simple concatenation with HF tokenizer
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
        dataset_type: Type of dataset ('chat_completion', 'instruct_dataset', etc.)

    Returns:
        Modified configuration dictionary
    """
    config["dataset_label"] = dataset_label

    if dataset_type in ('chat_completion', 'text_completion'):
        # Uses data_files directly (already in template)
        # Just need to ensure the path includes the dataset_label
        # Template has: "${input_dir}/${dataset_label}.json"
        # This gets substituted by the config values, so no changes needed here
        pass

    elif dataset_ext == '.parquet':
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
    parser.add_argument("--prompt", type=str, default="{input}\n", help="Prompt template (use {input} placeholder)")
    parser.add_argument("--input_key", type=str, default="input", help="JSON key for input field")
    parser.add_argument("--output_key", type=str, default="output", help="JSON key for output field")
    parser.add_argument("--system_prompt", type=str, default="", help="System prompt (legacy, for instruct_dataset)")
    parser.add_argument("--train_on_input", type=parse_bool, default=False, help="Whether to train on the input data (true/false)")

    # ------ Model/Training Args -----
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank (alpha will be auto-calculated as 2*rank)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--lr_scheduler", type=str, default="get_cosine_schedule_with_warmup", help="Learning rate scheduler function name (without 'torchtune.training.lr_schedulers.' prefix)")
    parser.add_argument("--dataset_type", type=str, default="chat_completion", help="Dataset type: 'chat_completion' (default, uses HF chat templates for train/eval parity) or legacy 'instruct_dataset'/'chat_dataset'")
    parser.add_argument("--packed", type=parse_bool, default=False, help="Whether to use packed sequences (true/false). Should be False for custom datasets.")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length for tokenizer")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="Dropout for LoRA layers")

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

    # ------ Training Step Guard -----
    parser.add_argument("--training_samples", type=int, default=None,
                        help="Number of training samples. When provided, computes and reports total training steps and warns if dangerously low.")

    parser.add_argument("--custom_recipe", type=str, help="Full name of a custom recipe file in the repo's custom_recipes folder to use for fine-tuning")
    parser.add_argument("--base_recipe", type=str, default=None,
                        help="Torchtune recipe name to use as base config (e.g., 'llama3_2/1B_lora_single_device'). Recipe defaults are used for parameters not explicitly set.")

    # ------ Model Selection -----
    parser.add_argument("--torchtune_model_name", type=str, default="Llama-3.2-1B-Instruct",
                        help="Model name as listed by 'tune ls' (e.g., 'Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Llama-3.1-8B-Instruct', 'Llama-3.3-70B-Instruct')")
    parser.add_argument("--model_checkpoint", type=str, default=None,
                        help="Model directory name within models_dir (defaults to torchtune_model_name if not provided)")

    return parser


def main():
    """Main function to set up fine-tuning configuration and SLURM script."""
    parser = create_parser()
    args = parser.parse_args()

    RANDOM_MODEL_RUN_NAME = run_names.generate_model_run_name()[0]

    # Load config file if it exists
    config_data = {}
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, "r") as f:
            config_data = yaml.safe_load(f) or {}

    # Load torchtune recipe hyperparameter defaults if base_recipe is specified
    recipe_defaults = {}
    recipe_config = None
    base_recipe = args.base_recipe or config_data.get('base_recipe')
    if base_recipe:
        try:
            recipe_config = config_recipe_loader.get_recipe_config(base_recipe)
            recipe_defaults = extract_flat_params(recipe_config, RECIPE_PARAM_MAPPING)
        except config_recipe_loader.RecipeConfigError as e:
            raise SystemExit(f"ERROR: Could not load recipe '{base_recipe}': {e}")

    # Apply default hyperparameters (only if at argparse default AND not overridden in config_file)
    # Precedence: CLI > config_file > recipe > argparse_default
    for key, value in recipe_defaults.items():
        if hasattr(args, key) and key not in config_data:
            default_value = parser.get_default(key)
            current_value = getattr(args, key)
            if current_value == default_value:
                setattr(args, key, value)

    # Apply config file values (higher priority than recipe)
    for key, value in config_data.items():
        # Only use config value if the argument wasn't explicitly provided on CLI
        # We check if it's still at its default value
        if hasattr(args, key):
            default_value = parser.get_default(key)
            current_value = getattr(args, key)
            # If current value equals default, use config file value
            if current_value == default_value:
                setattr(args, key, value)

    # Validate lr_scheduler (after config file has been loaded and merged)
    validate_lr_scheduler(args.lr_scheduler)

    # Training step guard: compute and warn if --training_samples provided
    if args.training_samples is not None:
        step_info = compute_training_steps(
            training_samples=args.training_samples,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epochs=args.epochs,
        )
        warn_on_low_steps(step_info, args.num_warmup_steps)

    model_run_name = args.my_wandb_run_name if args.my_wandb_run_name else RANDOM_MODEL_RUN_NAME
    username = os.environ.get("USER")

    # Get model config early (needed for both YAML and SLURM generation)
    if args.torchtune_model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: '{args.torchtune_model_name}'. "
            f"Supported models: {', '.join(MODEL_CONFIGS.keys())}"
        )
    model_config = MODEL_CONFIGS[args.torchtune_model_name]

    # Default dataset_type from MODEL_CONFIGS if not explicitly set by user or config file
    # Priority: CLI arg > config file > MODEL_CONFIGS > argparse default
    if args.dataset_type == parser.get_default("dataset_type"):
        # User didn't override via CLI or config file, use model's default
        args.dataset_type = model_config.get("dataset_type", "chat_completion")
    validate_dataset_type(args.dataset_type)

    # First edit the yaml template
    with open(f"{script_dir}/templates/finetune_template.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Derive model_dir early (needed for chat_completion's model_path)
    # Extract basename in case model_checkpoint is an absolute path
    raw_checkpoint = args.model_checkpoint or args.torchtune_model_name
    model_dir = os.path.basename(raw_checkpoint.rstrip('/'))

    for key, value in vars(args).items():
        if key in SLURM_ONLY or key in META_ARGS:
            continue
        # Model config - apply torchtune-specific settings
        elif key == "torchtune_model_name":
            # Model directory: use model_checkpoint if provided, otherwise torchtune_model_name
            # Extract basename in case it's an absolute path
            raw_checkpoint = args.model_checkpoint or value
            model_dir = os.path.basename(raw_checkpoint.rstrip('/'))

            # Set model component
            config["model"]["_component_"] = model_config["component"]

            # Set tokenizer path based on model family
            configure_tokenizer(config, model_config, model_dir, args.torchtune_model_name)

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
        elif key in ("prompt", "input_key", "output_key"):
            pass  # Handled in dataset_type
        elif key == "system_prompt":
            # chat_completion handles system_prompt in dataset_type switch
            # Only apply new_system_prompt for legacy instruct_dataset
            if value and args.dataset_type != 'chat_completion':
                config["dataset"]["new_system_prompt"] = value
                if "dataset_val" in config:
                    config["dataset_val"]["new_system_prompt"] = value
        elif key == "train_on_input":
            # Handled in dataset_type switch for chat_completion
            pass
        elif key == "lora_rank":
            # Set both rank and alpha (alpha = 2 * rank)
            config["model"]["lora_rank"] = value
            config["model"]["lora_alpha"] = calculate_lora_alpha(value)
        elif key == "lora_dropout":
            config["model"]["lora_dropout"] = value
        elif key == "lr":
            config["optimizer"]["lr"] = value
        elif key == "weight_decay":
            config["optimizer"]["weight_decay"] = value
        elif key == "gradient_accumulation_steps":
            config["gradient_accumulation_steps"] = value
        elif key == "base_recipe":
            pass  # Not a config parameter, used for loading recipe defaults
        elif key == "num_warmup_steps":
            config["lr_scheduler"]["num_warmup_steps"] = value
        elif key == "lr_scheduler":
            # Construct full component path
            config["lr_scheduler"]["_component_"] = f"torchtune.training.lr_schedulers.{value}"
        elif key == "dataset_type":
            if value == 'chat_completion':
                # Template already has chat_completion config
                # Just override the user-specified values and set model_path
                config["dataset"]["model_path"] = f"${{models_dir}}/{model_dir}"
                config["dataset"]["prompt"] = args.prompt
                config["dataset"]["system_prompt"] = args.system_prompt
                config["dataset"]["input_key"] = args.input_key
                config["dataset"]["output_key"] = args.output_key
                config["dataset"]["train_on_input"] = args.train_on_input
                if "dataset_val" in config:
                    config["dataset_val"]["model_path"] = f"${{models_dir}}/{model_dir}"
                    config["dataset_val"]["prompt"] = args.prompt
                    config["dataset_val"]["system_prompt"] = args.system_prompt
                    config["dataset_val"]["input_key"] = args.input_key
                    config["dataset_val"]["output_key"] = args.output_key
                    config["dataset_val"]["train_on_input"] = args.train_on_input
            elif value == 'text_completion':
                # Base model dataset - simple concatenation, no chat template
                config["dataset"]["_component_"] = "cruijff_kit.tools.torchtune.datasets.text_completion.text_completion_dataset"
                config["dataset"]["model_path"] = f"${{models_dir}}/{model_dir}"
                config["dataset"]["prompt"] = args.prompt
                config["dataset"]["input_key"] = args.input_key
                config["dataset"]["output_key"] = args.output_key
                config["dataset"]["train_on_input"] = args.train_on_input
                # Remove system_prompt - not used for base models
                config["dataset"].pop("system_prompt", None)
                if "dataset_val" in config:
                    config["dataset_val"]["_component_"] = "cruijff_kit.tools.torchtune.datasets.text_completion.text_completion_dataset"
                    config["dataset_val"]["model_path"] = f"${{models_dir}}/{model_dir}"
                    config["dataset_val"]["prompt"] = args.prompt
                    config["dataset_val"]["input_key"] = args.input_key
                    config["dataset_val"]["output_key"] = args.output_key
                    config["dataset_val"]["train_on_input"] = args.train_on_input
                    config["dataset_val"].pop("system_prompt", None)
            else:
                # Legacy dataset types - need to reconfigure from template
                config["dataset"]["_component_"] = f"torchtune.datasets.{value}"
                # Remove chat_completion specific keys
                for key_to_remove in ["prompt", "input_key", "output_key", "model_path", "system_prompt"]:
                    config["dataset"].pop(key_to_remove, None)
                # Add back keys needed for legacy types
                config["dataset"]["data_dir"] = config["dataset"].pop("data_files").replace(".json", "")
                config["dataset"]["split"] = config["dataset"].pop("field")
                if value == 'chat_dataset':
                    config["dataset"]["conversation_column"] = "messages"
                    config["dataset"]["conversation_style"] = "openai"
                if "dataset_val" in config:
                    config["dataset_val"]["_component_"] = f"torchtune.datasets.{value}"
                    for key_to_remove in ["prompt", "input_key", "output_key", "model_path", "system_prompt"]:
                        config["dataset_val"].pop(key_to_remove, None)
                    config["dataset_val"]["data_dir"] = config["dataset_val"].pop("data_files").replace(".json", "")
                    config["dataset_val"]["split"] = config["dataset_val"].pop("field")
                    if value == 'chat_dataset':
                        config["dataset_val"]["conversation_column"] = "messages"
                        config["dataset_val"]["conversation_style"] = "openai"
        elif key == "packed":
            # Add packed to dataset config (recipes read this to select collate_fn)
            config["dataset"]["packed"] = value
            if "dataset_val" in config:
                config["dataset_val"]["packed"] = value
        elif key == "max_seq_len":
            # Update tokenizer's max_seq_len
            config["tokenizer"]["max_seq_len"] = value
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

    # Partition: set by caller (scaffold agent reads claude.local.md)
    # Use 'is not None' check because empty string is valid (MIG support)
    partition = args.partition
    if partition is not None and partition != "":
        slurm_script = slurm_script.replace("##SBATCH --partition=<PART>", "#SBATCH --partition=" + partition)

    # Constraint: set by caller (scaffold agent reads claude.local.md)
    constraint = args.constraint
    if constraint:
        slurm_script = slurm_script.replace("##SBATCH --constraint=<CONST>", "#SBATCH --constraint=" + constraint)
    if args.custom_recipe:
        custom_recipe = args.custom_recipe
        # Auto-switch between single_device and distributed based on GPU count (patch for custom recipes)
        if gpus > 1 and "single_device" in custom_recipe:
            custom_recipe = custom_recipe.replace("single_device", "distributed")
        elif gpus == 1 and "distributed" in custom_recipe:
            custom_recipe = custom_recipe.replace("distributed", "single_device")

        if gpus == 1:
            slurm_script = slurm_script.replace("lora_finetune_single_device", custom_recipe + '.__main__')
        else:
            slurm_script = slurm_script.replace("lora_finetune_distributed", custom_recipe + '.__main__')

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
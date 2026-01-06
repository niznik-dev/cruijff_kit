#!/usr/bin/env python3
"""Generate finetune.yaml from template and parameters (template-based approach).

This script implements the template-based workflow for generating finetune.yaml:
1. Load finetune_template.yaml (cruijff_kit's base template)
2. Apply parameter substitutions with special case handling
3. Configure dataset based on format (json/parquet) and type
4. Strip SLURM-only parameters
5. Output complete finetune.yaml

This is the template-based approach (alternative to recipe-based with merge_recipe_params.py).
Use this when you want cruijff_kit's default template rather than torchtune recipes.

Usage:
    # With params file
    python generate_finetune_yaml.py --params params.yaml --output finetune.yaml

    # With direct CLI args
    python generate_finetune_yaml.py \\
        --lora-rank 8 \\
        --lr 1e-4 \\
        --my-wandb-run-name my_run \\
        --output finetune.yaml
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Import from local modules
from config_utils import (
    SLURM_ONLY_PARAMS,
    parse_epochs,
    parse_bool,
    calculate_lora_alpha,
    construct_output_dir,
    expand_user_in_paths,
    strip_slurm_params,
    validate_lr_scheduler,
    validate_dataset_type,
)
from dataset_config import build_dataset_pair
from cruijff_kit.utils import run_names

# Calculate paths relative to this script
script_dir = Path(__file__).parent


def apply_template_parameters(config: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Apply parameters to template config with special case handling.

    This function handles the template-based workflow where we start with
    cruijff_kit's finetune_template.yaml and substitute/modify values.

    Args:
        config: Template configuration loaded from YAML
        parameters: Flat dictionary of parameters to apply

    Returns:
        Modified configuration ready for finetune.yaml
    """
    # Extract commonly used values
    model_run_name = parameters.get("my_wandb_run_name", "")
    output_dir_base = parameters.get("output_dir_base", "")
    experiment_name = parameters.get("experiment_name", "")

    # --- Direct Mappings (simple key-value) ---
    direct_mappings = [
        "my_wandb_project",
        "my_wandb_run_name",
        "batch_size",
        "epochs",
        "log_every_n_steps",
        "run_val_every_n_steps",
        "max_steps_per_epoch",
        "save_adapter_weights_only",
        "save_last_epoch_only",
        "stash_adapter_weights",
        "epochs_to_save",
    ]

    for key in direct_mappings:
        if key in parameters and parameters[key] is not None:
            config[key] = parameters[key]

    # --- Nested Mappings ---

    # Learning rate → optimizer.lr
    if "lr" in parameters and "optimizer" in config:
        config["optimizer"]["lr"] = parameters["lr"]

    # LoRA rank → model.lora_rank AND model.lora_alpha
    if "lora_rank" in parameters and "model" in config:
        config["model"]["lora_rank"] = parameters["lora_rank"]
        config["model"]["lora_alpha"] = calculate_lora_alpha(parameters["lora_rank"])

    # Warmup steps → lr_scheduler.num_warmup_steps
    if "num_warmup_steps" in parameters and "lr_scheduler" in config:
        config["lr_scheduler"]["num_warmup_steps"] = parameters["num_warmup_steps"]

    # LR scheduler component
    if "lr_scheduler" in parameters and isinstance(parameters["lr_scheduler"], str):
        scheduler = parameters["lr_scheduler"]
        if not scheduler.startswith("torchtune.") and "lr_scheduler" in config:
            config["lr_scheduler"]["_component_"] = f"torchtune.training.lr_schedulers.{scheduler}"

    # --- Output Directory ---
    if output_dir_base and model_run_name:
        config["output_dir"] = construct_output_dir(
            output_dir_base, experiment_name, model_run_name
        )

    # --- Input Directory ---
    input_dir_base = parameters.get("input_dir_base", "")
    input_formatting = parameters.get("input_formatting", "")
    if input_dir_base:
        config["input_dir"] = input_dir_base + input_formatting + ("/" if input_formatting else "")

    # --- Models Directory ---
    if "models_dir" in parameters:
        config["models_dir"] = parameters["models_dir"]

    # --- Dataset Configuration ---
    data_path = None
    if "input_dir" in config and "dataset_label" in parameters:
        data_path = config["input_dir"] + parameters["dataset_label"]
    elif "dataset_label" in parameters and input_dir_base:
        data_path = input_dir_base + input_formatting + ("/" if input_formatting else "") + parameters["dataset_label"]

    if data_path:
        # Determine format
        dataset_ext = parameters.get("dataset_ext", ".json")
        if dataset_ext == ".json":
            data_format = "json"
        elif dataset_ext in [".parquet", "/"]:
            data_format = "parquet"
        else:
            data_format = "json"  # default

        # Determine type
        dataset_type = parameters.get("dataset_type", "instruct_dataset")

        # Determine if validation is needed
        run_val = parameters.get("run_val_every_n_steps", config.get("run_val_every_n_steps", 0))
        include_validation = run_val > 0

        # System prompt
        system_prompt = parameters.get("system_prompt", "")

        # Train on input
        train_on_input = parameters.get("train_on_input", False)

        # Build dataset configs
        dataset_configs = build_dataset_pair(
            data_path=data_path,
            data_format=data_format,
            dataset_type=dataset_type,
            train_on_input=train_on_input,
            system_prompt=system_prompt if system_prompt else None,
            include_validation=include_validation,
        )

        config["dataset"] = dataset_configs["dataset"]
        if "dataset_val" in dataset_configs:
            config["dataset_val"] = dataset_configs["dataset_val"]
        elif "dataset_val" in config:
            del config["dataset_val"]

    # --- Validation config removal if not needed ---
    if config.get("run_val_every_n_steps", 0) == 0:
        config.pop("dataset_val", None)
        config.pop("run_val_every_n_steps", None)

    # --- Path Expansion ---
    config = expand_user_in_paths(config)

    return config


def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate finetune.yaml from template and parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/Output
    parser.add_argument(
        "--params",
        type=str,
        help="Path to YAML file with all parameters"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="finetune.yaml",
        help="Output path for finetune.yaml (default: finetune.yaml)"
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Path to template YAML (default: templates/finetune_template.yaml)"
    )

    # ----- Core Parameters -----
    parser.add_argument("--my-wandb-project", type=str, default="PredictingZygosity")
    parser.add_argument("--my-wandb-run-name", type=str,
                        help="Run name for wandb (auto-generated if not provided)")
    parser.add_argument("--input-formatting", type=str, default="raw")
    parser.add_argument("--dataset-label", type=str, default="tune_dataset")
    parser.add_argument("--dataset-ext", type=str, default="")
    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--output-dir-base", type=str, default="/scratch/gpfs/MSALGANIK/$USER/")
    parser.add_argument("--input-dir-base", type=str, default="/scratch/gpfs/MSALGANIK/$USER/zyg_in/")
    parser.add_argument("--models-dir", type=str, default="/scratch/gpfs/MSALGANIK/pretrained-llms/")

    # ----- Training Parameters -----
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-adapter-weights-only", type=parse_bool, default=False)
    parser.add_argument("--save-last-epoch-only", type=parse_bool, default=False)
    parser.add_argument("--stash-adapter-weights", type=parse_bool, default=False)
    parser.add_argument("--epochs-to-save", type=parse_epochs, default="all")
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--log-every-n-steps", type=int, default=5)
    parser.add_argument("--run-val-every-n-steps", type=int, default=0)
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--train-on-input", type=parse_bool, default=False)

    # ----- Model/Optimizer Parameters -----
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-warmup-steps", type=int, default=100)
    parser.add_argument("--lr-scheduler", type=str, default="get_cosine_schedule_with_warmup")
    parser.add_argument("--dataset-type", type=str, default="instruct_dataset")

    return parser


def load_params_from_file(filepath: str) -> dict:
    """Load parameters from YAML file."""
    with open(filepath, 'r') as f:
        params = yaml.safe_load(f) or {}
    return params


def generate_yaml_from_template(
    template_path: str,
    parameters: dict,
    output_path: str
) -> None:
    """Generate finetune.yaml from template and parameters.

    Args:
        template_path: Path to finetune_template.yaml
        parameters: Dictionary of parameters (with underscored keys)
        output_path: Path to write finetune.yaml
    """
    # Load template
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate lr_scheduler and dataset_type
    if 'lr_scheduler' in parameters:
        validate_lr_scheduler(parameters['lr_scheduler'])
    if 'dataset_type' in parameters:
        validate_dataset_type(parameters['dataset_type'])

    # Generate random run name if not provided
    if 'my_wandb_run_name' not in parameters or not parameters['my_wandb_run_name']:
        parameters['my_wandb_run_name'] = run_names.generate_model_run_name()[0]

    # Apply parameters to template
    config = apply_template_parameters(config, parameters)

    # Strip SLURM-only parameters
    config = strip_slurm_params(config)

    # Write output
    with open(output_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Generated finetune.yaml: {output_path}")
    print(f"  Run name: {config.get('my_wandb_run_name', 'N/A')}")
    print(f"  Output directory: {config.get('output_dir', 'N/A')}")
    if 'model' in config:
        print(f"  LoRA rank: {config['model'].get('lora_rank', 'N/A')}")
        print(f"  LoRA alpha: {config['model'].get('lora_alpha', 'N/A')}")
    if 'optimizer' in config:
        print(f"  Learning rate: {config['optimizer'].get('lr', 'N/A')}")


def main():
    """Main entry point for CLI usage."""
    parser = create_parser()
    args = parser.parse_args()

    # Load parameters from file if provided
    parameters = {}
    if args.params:
        parameters = load_params_from_file(args.params)
        print(f"Loaded parameters from {args.params}")

    # Get parser defaults for comparison
    parser_defaults = {}
    for action in parser._actions:
        if action.dest != 'help' and hasattr(action, 'default'):
            parser_defaults[action.dest] = action.default

    cli_params = vars(args)

    # Convert hyphenated CLI args to underscored keys
    normalized_cli_params = {}
    for key, value in cli_params.items():
        normalized_key = key.replace('-', '_')
        normalized_cli_params[normalized_key] = value

    for key, value in normalized_cli_params.items():
        if key in ['params', 'output', 'template']:
            continue  # Skip meta-parameters

        # Override file param if CLI param was explicitly provided (not default)
        orig_key = key.replace('_', '-')
        if orig_key in parser_defaults:
            if value != parser_defaults.get(orig_key):
                parameters[key] = value
        else:
            if value is not None:
                parameters[key] = value

    # If no params file, use all CLI params
    if not args.params:
        parameters = {k: v for k, v in normalized_cli_params.items()
                      if k not in ['params', 'output', 'template']}

    # Determine template path
    if args.template:
        template_path = args.template
    else:
        template_path = script_dir / "templates" / "finetune_template.yaml"

    # Generate YAML
    try:
        generate_yaml_from_template(
            template_path=str(template_path),
            parameters=parameters,
            output_path=args.output
        )
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid parameter value: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating YAML: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

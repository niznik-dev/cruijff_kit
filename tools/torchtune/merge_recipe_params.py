#!/usr/bin/env python3
"""Merge torchtune recipe defaults with experiment parameters.

This script implements the recipe-based workflow for generating finetune.yaml:
1. Extract defaults from a torchtune recipe (via tune cp)
2. Merge with precedence: recipe defaults → controls → run parameters
3. Replace the dataset section with local file configuration
4. Apply derived calculations (e.g., lora_alpha = 2 * lora_rank)
5. Output complete finetune.yaml

Usage:
    python merge_recipe_params.py \\
        --recipe llama3_2/1B_lora_single_device \\
        --controls controls.yaml \\
        --run-params run_params.yaml \\
        --additional-params additional_params.yaml \\
        --output finetune.yaml
"""

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Import from local modules
from recipe_config_loader import (
    get_recipe_config,
    merge_parameters,
    format_merge_tracking,
    RecipeNotFoundError,
    RecipeExtractionError
)
from config_utils import (
    calculate_lora_alpha,
    expand_user_in_paths,
    strip_slurm_params,
    construct_output_dir,
    SLURM_ONLY_PARAMS
)
from dataset_config import build_dataset_pair

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def apply_recipe_overrides(
    config: Dict[str, Any],
    additional_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply additional parameters and derived calculations to recipe config.

    This handles:
    - Dataset section replacement (from HF defaults to local files)
    - Output directory construction
    - WandB configuration
    - LoRA alpha calculation (if lora_rank was changed)
    - Path expansion ($USER)

    Args:
        config: Merged recipe configuration
        additional_params: Non-recipe parameters (paths, wandb, dataset info, etc.)

    Returns:
        Complete configuration ready for finetune.yaml
    """
    if additional_params is None:
        additional_params = {}

    # --- Dataset Configuration ---
    # Replace recipe's HF dataset with local file configuration
    data_path = additional_params.get('data_path')
    data_format = additional_params.get('data_format', 'json')
    dataset_type = additional_params.get('dataset_type', 'instruct_dataset')
    train_on_input = additional_params.get('train_on_input', False)
    system_prompt = additional_params.get('system_prompt', '')
    include_validation = additional_params.get('validation_during_training', False)

    if data_path:
        dataset_configs = build_dataset_pair(
            data_path=data_path,
            data_format=data_format,
            dataset_type=dataset_type,
            train_on_input=train_on_input,
            system_prompt=system_prompt if system_prompt else None,
            include_validation=include_validation,
        )
        # Replace dataset sections entirely
        config['dataset'] = dataset_configs['dataset']
        if 'dataset_val' in dataset_configs:
            config['dataset_val'] = dataset_configs['dataset_val']
            # Set validation frequency if not already set
            if 'run_val_every_n_steps' not in config:
                config['run_val_every_n_steps'] = 50
        else:
            # Remove validation config if not needed
            config.pop('dataset_val', None)
            config.pop('run_val_every_n_steps', None)

    # --- Output Directory ---
    output_dir_base = additional_params.get('output_dir_base', '')
    experiment_name = additional_params.get('experiment_name', '')
    run_name = additional_params.get('my_wandb_run_name', '')

    if output_dir_base and run_name:
        config['output_dir'] = construct_output_dir(
            output_dir_base, experiment_name, run_name
        )

    # Update checkpointer output_dir if it exists
    if 'checkpointer' in config and 'output_dir' in config:
        config['checkpointer']['output_dir'] = config['output_dir']

    # --- Model Directory ---
    models_dir = additional_params.get('models_dir', '')
    if models_dir:
        config['models_dir'] = models_dir
        # Update tokenizer path if it uses models_dir reference
        if 'tokenizer' in config and 'path' in config['tokenizer']:
            # Only update if it's a relative reference
            token_path = config['tokenizer']['path']
            if '${models_dir}' in token_path:
                config['tokenizer']['path'] = token_path.replace(
                    '${models_dir}', models_dir
                )

        # Update checkpointer checkpoint_dir
        if 'checkpointer' in config and 'checkpoint_dir' in config['checkpointer']:
            ckpt_dir = config['checkpointer']['checkpoint_dir']
            if '${models_dir}' in ckpt_dir:
                config['checkpointer']['checkpoint_dir'] = ckpt_dir.replace(
                    '${models_dir}', models_dir
                )

    # --- WandB Configuration ---
    if 'my_wandb_run_name' in additional_params:
        config['my_wandb_run_name'] = additional_params['my_wandb_run_name']
    if 'my_wandb_project' in additional_params:
        config['my_wandb_project'] = additional_params['my_wandb_project']

    # Update metric_logger if it exists
    if 'metric_logger' in config:
        if run_name:
            config['metric_logger']['name'] = run_name
        if 'my_wandb_project' in additional_params:
            config['metric_logger']['project'] = additional_params['my_wandb_project']
        if 'output_dir' in config:
            config['metric_logger']['log_dir'] = config['output_dir'] + 'logs'

    # --- LoRA Alpha Calculation ---
    # If lora_rank was overridden, recalculate lora_alpha
    if 'model' in config and 'lora_rank' in config['model']:
        config['model']['lora_alpha'] = calculate_lora_alpha(config['model']['lora_rank'])

    # --- Path Expansion ---
    config = expand_user_in_paths(config)

    return config


def merge_for_run(
    recipe_name: str,
    controls: Dict[str, Any],
    run_parameters: Dict[str, Any],
    additional_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Merge recipe defaults with controls and run parameters for a single run.

    Args:
        recipe_name: Name of torchtune recipe (e.g., "llama3_2/1B_lora_single_device")
        controls: Experiment-level control parameters
        run_parameters: Run-specific parameters (override controls)
        additional_params: Non-recipe parameters (paths, wandb, dataset info, SLURM)

    Returns:
        Complete finetune.yaml configuration dictionary

    Raises:
        RecipeNotFoundError: If recipe doesn't exist
        RecipeExtractionError: If recipe extraction fails
    """
    # Get recipe defaults
    logger.info(f"Extracting defaults from recipe: {recipe_name}")
    try:
        recipe_defaults = get_recipe_config(recipe_name)
        logger.info(f"Loaded recipe with {len(recipe_defaults)} top-level keys")
    except (RecipeNotFoundError, RecipeExtractionError) as e:
        logger.error(f"Failed to load recipe: {e}")
        raise

    # Merge with precedence: recipe → controls → run parameters
    merged_config, tracking = merge_parameters(
        recipe_defaults,
        controls,
        run_parameters
    )

    # Log merge tracking
    logger.debug(format_merge_tracking(tracking))

    # Apply additional overrides (dataset, paths, wandb, etc.)
    merged_config = apply_recipe_overrides(merged_config, additional_params)

    # Strip SLURM-only parameters
    merged_config = strip_slurm_params(merged_config)

    logger.info(f"Generated complete finetune.yaml config with {len(merged_config)} top-level keys")
    return merged_config


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Merge torchtune recipe defaults with experiment parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input arguments
    parser.add_argument(
        '--recipe',
        type=str,
        required=True,
        help='Torchtune recipe name (e.g., "llama3_2/1B_lora_single_device")'
    )
    parser.add_argument(
        '--controls',
        type=str,
        help='Path to YAML file with experiment controls'
    )
    parser.add_argument(
        '--run-params',
        type=str,
        help='Path to YAML file with run-specific parameters'
    )
    parser.add_argument(
        '--additional-params',
        type=str,
        help='Path to YAML file with non-recipe parameters (paths, wandb, dataset, SLURM)'
    )

    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for merged YAML (default: print to stdout)'
    )
    parser.add_argument(
        '--format',
        choices=['yaml', 'json'],
        default='yaml',
        help='Output format (default: yaml)'
    )

    # Logging arguments
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all logging except errors'
    )

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load input files
    try:
        controls = {}
        if args.controls:
            with open(args.controls, 'r') as f:
                controls = yaml.safe_load(f) or {}
            logger.info(f"Loaded controls from {args.controls}")

        run_params = {}
        if args.run_params:
            with open(args.run_params, 'r') as f:
                run_params = yaml.safe_load(f) or {}
            logger.info(f"Loaded run parameters from {args.run_params}")

        additional_params = {}
        if args.additional_params:
            with open(args.additional_params, 'r') as f:
                additional_params = yaml.safe_load(f) or {}
            logger.info(f"Loaded additional parameters from {args.additional_params}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        sys.exit(1)

    # Merge parameters
    try:
        merged = merge_for_run(
            recipe_name=args.recipe,
            controls=controls,
            run_parameters=run_params,
            additional_params=additional_params
        )
    except (RecipeNotFoundError, RecipeExtractionError) as e:
        logger.error(f"Merge failed: {e}")
        sys.exit(1)

    # Format output
    if args.format == 'yaml':
        output_str = yaml.dump(merged, default_flow_style=False, sort_keys=False)
    else:
        output_str = json.dumps(merged, indent=2)

    # Write or print output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_str)
        logger.info(f"Wrote merged config to {args.output}")
    else:
        if not args.quiet:
            print()
        print(output_str)


if __name__ == '__main__':
    main()

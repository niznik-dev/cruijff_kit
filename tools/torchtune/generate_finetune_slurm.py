#!/usr/bin/env python3
"""Generate SLURM submission script for torchtune fine-tuning.

This script generates finetune.slurm by:
1. Reading finetune.yaml to extract job_name and output_dir
2. Loading SLURM parameters from input (YAML file or CLI args)
3. Reading finetune_template.slurm
4. Applying SLURM-specific substitutions
5. Writing finetune.slurm

Works with both recipe-based and template-based finetune.yaml files.

Usage:
    # With SLURM params file
    python generate_finetune_slurm.py \\
        --config finetune.yaml \\
        --slurm-params slurm.yaml \\
        --output finetune.slurm

    # With direct CLI args
    python generate_finetune_slurm.py \\
        --config finetune.yaml \\
        --gpus 2 \\
        --time 01:00:00 \\
        --conda-env cruijff \\
        --output finetune.slurm
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Calculate paths relative to this script
script_dir = Path(__file__).parent


def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate SLURM submission script for torchtune fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to finetune.yaml (to extract job_name and output_dir)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="finetune.slurm",
        help="Output path for SLURM script (default: finetune.slurm)"
    )

    # SLURM parameters source
    parser.add_argument(
        "--slurm-params",
        type=str,
        help="Path to YAML file with SLURM parameters"
    )

    # Individual SLURM parameters (can override slurm-params file)
    parser.add_argument("--time", type=str, default="00:15:00",
                       help="Time limit (HH:MM:SS) (default: 00:15:00)")
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use (default: 1)")
    parser.add_argument("--conda-env", type=str, default="cruijff",
                       help="Name of conda environment (default: cruijff)")
    parser.add_argument("--venv", type=str, default="",
                       help="Path to virtual environment (if not using conda)")
    parser.add_argument("--modules", type=str, default="",
                       help="Modules to load, comma-separated (e.g., '2024,Python/3.12.3')")
    parser.add_argument("--account", type=str,
                       help="SLURM account to use")
    parser.add_argument("--partition", type=str,
                       help="SLURM partition to use")
    parser.add_argument("--constraint", type=str,
                       help="SLURM constraint to use")
    parser.add_argument("--custom-recipe", type=str,
                       help="Full name of custom recipe module")

    return parser


def load_slurm_params_from_file(filepath: str) -> dict:
    """Load SLURM parameters from YAML file.

    Args:
        filepath: Path to YAML file with SLURM parameters

    Returns:
        Dictionary of SLURM parameters
    """
    with open(filepath, 'r') as f:
        params = yaml.safe_load(f) or {}
    return params


def generate_slurm_script(
    config_path: str,
    slurm_params: dict,
    template_path: str,
    output_path: str
) -> None:
    """Generate SLURM submission script from finetune.yaml and parameters.

    Args:
        config_path: Path to finetune.yaml
        slurm_params: Dictionary of SLURM parameters
        template_path: Path to finetune_template.slurm
        output_path: Path to write finetune.slurm
    """
    # Read finetune.yaml to get job_name and output_dir
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    job_name = config.get('my_wandb_run_name', 'finetune_job')
    output_dir = config.get('output_dir', './')

    # Get username
    username = os.environ.get("USER", "unknown")

    # Extract SLURM parameters with defaults
    time = slurm_params.get('time', '00:15:00')
    gpus = slurm_params.get('gpus', 1)
    conda_env = slurm_params.get('conda_env', 'cruijff')
    venv = slurm_params.get('venv', '')
    modules = slurm_params.get('modules', '')
    account = slurm_params.get('account')
    partition = slurm_params.get('partition')
    constraint = slurm_params.get('constraint')
    custom_recipe = slurm_params.get('custom_recipe')

    # Read SLURM template
    with open(template_path, 'r') as f:
        slurm_script = f.read()

    # Apply substitutions (same logic as setup_finetune.py lines 339-377)

    # Basic substitutions
    slurm_script = slurm_script.replace("<JOBNAME>", job_name)
    slurm_script = slurm_script.replace("<NETID>", username)
    slurm_script = slurm_script.replace("00:15:00", time)

    # GPU configuration
    if gpus > 1:
        # Multi-GPU: adjust CPU count and GPU count, use distributed recipe
        slurm_script = slurm_script.replace(
            "#SBATCH --cpus-per-task=1",
            f"#SBATCH --cpus-per-task={gpus}"
        )
        slurm_script = slurm_script.replace(
            "#SBATCH --gres=gpu:1",
            f"#SBATCH --gres=gpu:{gpus}"
        )
        slurm_script = slurm_script.replace(
            "lora_finetune_single_device",
            f"--nproc_per_node={gpus} lora_finetune_distributed"
        )

    # Optional SLURM directives (uncomment and set if provided)
    if account:
        slurm_script = slurm_script.replace(
            "##SBATCH --account=<ACT>",
            f"#SBATCH --account={account}"
        )
    if partition:
        slurm_script = slurm_script.replace(
            "##SBATCH --partition=<PART>",
            f"#SBATCH --partition={partition}"
        )
    if constraint:
        slurm_script = slurm_script.replace(
            "##SBATCH --constraint=<CONST>",
            f"#SBATCH --constraint={constraint}"
        )

    # Custom recipe
    if custom_recipe:
        if gpus == 1:
            slurm_script = slurm_script.replace(
                "lora_finetune_single_device",
                f"{custom_recipe}.__main__"
            )
        else:
            slurm_script = slurm_script.replace(
                "lora_finetune_distributed",
                f"{custom_recipe}.__main__"
            )

    # Environment activation
    slurm_script = slurm_script.replace("<CONDA_ENV>", conda_env)

    if venv:
        # Replace conda activation with venv activation
        slurm_script = slurm_script.replace(
            f"conda activate {conda_env}",
            f"source $PROJECT/venvs/{venv}/bin/activate"
        )
    elif modules:
        # Remove conda lines and add module load commands
        slurm_script = "\n".join(
            line for line in slurm_script.splitlines()
            if "conda" not in line
        )
        module_string = ''
        for m in modules.split(','):
            module_string += f"\nmodule load {m.strip()}"
        slurm_script = slurm_script.replace(
            "module purge",
            "module purge" + module_string
        )

    # Output directory and $USER expansion
    slurm_script = slurm_script.replace("<OUTPUT_DIR>", output_dir)
    slurm_script = slurm_script.replace("$USER", username)

    # Write SLURM script
    with open(output_path, 'w') as f:
        f.write(slurm_script)

    print(f"Generated SLURM script: {output_path}")
    print(f"  Job name: {job_name}")
    print(f"  Output directory: {output_dir}")
    print(f"  GPUs: {gpus}")
    print(f"  Time limit: {time}")


def main():
    """Main entry point for CLI usage."""
    parser = create_parser()
    args = parser.parse_args()

    # Load SLURM parameters from file if provided
    slurm_params = {}
    if args.slurm_params:
        slurm_params = load_slurm_params_from_file(args.slurm_params)

    # Override with CLI arguments (CLI takes precedence)
    cli_params = {
        'time': args.time,
        'gpus': args.gpus,
        'conda_env': args.conda_env,
        'venv': args.venv,
        'modules': args.modules,
    }

    # Add optional parameters if provided on CLI
    if args.account:
        cli_params['account'] = args.account
    if args.partition:
        cli_params['partition'] = args.partition
    if args.constraint:
        cli_params['constraint'] = args.constraint
    if args.custom_recipe:
        cli_params['custom_recipe'] = args.custom_recipe

    # Merge: file params + CLI params (CLI overrides file)
    # Only override if CLI value is not the default
    parser_defaults = {
        'time': '00:15:00',
        'gpus': 1,
        'conda_env': 'cruijff',
        'venv': '',
        'modules': ''
    }

    for key, value in cli_params.items():
        # Override file param if CLI param was explicitly provided (not default)
        if key in parser_defaults:
            if value != parser_defaults[key]:
                slurm_params[key] = value
        else:
            # Optional params (account, partition, etc.) - always use if provided
            if value:
                slurm_params[key] = value

    # If no slurm-params file and using all defaults, use CLI params directly
    if not args.slurm_params:
        slurm_params = cli_params

    # Template path
    template_path = script_dir / "templates" / "finetune_template.slurm"

    # Generate SLURM script
    try:
        generate_slurm_script(
            config_path=args.config,
            slurm_params=slurm_params,
            template_path=str(template_path),
            output_path=args.output
        )
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating SLURM script: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

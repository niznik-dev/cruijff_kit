#!/usr/bin/env python3
"""
Generate SLURM batch scripts for fine-tuning experiments.

This script automatically creates finetune.slurm scripts for all runs in an
experiment directory by reading their finetune.yaml configurations and
determining appropriate resource allocations.
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def parse_local_config(cruijff_root: Path) -> Dict[str, str]:
    """Parse user-specific settings from claude.local.md."""
    local_config_path = cruijff_root / "claude.local.md"

    # Default values
    config = {
        'email': 'user@princeton.edu',
        'account': 'myaccount',
        'partition': 'gpu',
        'constraint': 'gpu80',
        'conda_env': 'ttenv',
    }

    if not local_config_path.exists():
        print(f"Warning: {local_config_path} not found. Using default values.")
        return config

    # Parse the markdown file for key settings
    with open(local_config_path) as f:
        content = f.read()

        # Simple parsing - look for specific patterns
        for line in content.split('\n'):
            if '**Username**:' in line or 'Username' in line and '`' in line:
                # Extract username from patterns like: - **Username**: `mjs3`
                if '`' in line:
                    username = line.split('`')[1]
                    config['email'] = f"{username}@princeton.edu"
            elif '**Account**:' in line:
                if '`' in line:
                    config['account'] = line.split('`')[1]
            elif '**Partition**:' in line:
                if '`' in line:
                    config['partition'] = line.split('`')[1]
            elif '**Constraint**:' in line:
                if '`' in line:
                    config['constraint'] = line.split('`')[1]
            elif '**Default conda environment**:' in line:
                if '`' in line:
                    config['conda_env'] = line.split('`')[1]

    return config


def parse_runs_plan(experiment_dir: Path) -> Dict[str, Dict[str, str]]:
    """Parse runs_plan.md to extract resource estimates for each run.

    Returns a dictionary mapping run names to their resource requirements.
    Example: {'Llama-3.2-1B-Instruct_5L_rank4': {'time': '10 min', 'mem': '32G'}}
    """
    runs_plan_path = experiment_dir / "runs_plan.md"
    resource_map = {}

    if not runs_plan_path.exists():
        return resource_map

    with open(runs_plan_path) as f:
        content = f.read()
        lines = content.split('\n')

        # Look for the table with run information
        in_table = False
        for line in lines:
            # Detect table start (contains Run Name header)
            if '| Run Name |' in line and '| Model |' in line:
                in_table = True
                continue

            # Skip table separator line
            if in_table and line.startswith('|---'):
                continue

            # End table when we hit a blank line or new section
            if in_table and (not line.strip() or line.startswith('#')):
                in_table = False
                continue

            # Parse table rows
            if in_table and line.startswith('|'):
                parts = [p.strip() for p in line.split('|')]
                # parts[0] is empty (before first |)
                # parts[1] is Run Name
                # parts[2] is Model
                # parts[3] is LoRA Rank
                # parts[4] is Train Dataset
                # parts[5] is Batch Size
                # parts[6] is Est. Time

                if len(parts) >= 7:
                    run_name = parts[1]
                    model = parts[2]
                    est_time = parts[6]

                    # Skip header row
                    if run_name == 'Run Name':
                        continue

                    # Convert time estimate to SLURM format
                    # "10 min" -> "00:15:00" (add buffer)
                    # "20 min" -> "00:30:00" (add buffer)
                    time_slurm = '00:20:00'  # default
                    if 'min' in est_time:
                        try:
                            minutes = int(est_time.split()[0])
                            # Add 50% buffer for safety
                            minutes_buffered = int(minutes * 1.5)
                            hours = minutes_buffered // 60
                            mins = minutes_buffered % 60
                            time_slurm = f"{hours:02d}:{mins:02d}:00"
                        except:
                            pass

                    # Determine memory based on model size
                    mem = '32G'
                    if '1B' in model or '1b' in model:
                        mem = '32G'
                    elif '3B' in model or '3b' in model:
                        mem = '64G'
                    elif '7B' in model or '8B' in model:
                        mem = '128G'
                    elif '70B' in model:
                        mem = '256G'

                    resource_map[run_name] = {
                        'time': time_slurm,
                        'mem': mem,
                    }

    return resource_map


def determine_resources(config: Dict[str, Any], run_name: str = "", resource_map: Dict[str, Dict[str, str]] = None) -> Dict[str, str]:
    """Determine SLURM resource requirements.

    Priority:
    1. Use resource_map from runs_plan.md if available
    2. Fall back to model-size-based heuristics
    """
    # Default values
    resources = {
        'mem': '32G',
        'time': '00:20:00',
        'cpus': '1',
        'gpus': '1',
    }

    # First, try to use resource map from runs_plan.md
    if resource_map and run_name in resource_map:
        plan_resources = resource_map[run_name]
        resources['mem'] = plan_resources.get('mem', resources['mem'])
        resources['time'] = plan_resources.get('time', resources['time'])
        return resources

    # Fall back to model-size-based heuristics
    model_component = config.get('model', {}).get('_component_', '')
    model_lower = model_component.lower()

    if '1b' in model_lower:
        resources['mem'] = '32G'
        resources['time'] = '00:20:00'
    elif '3b' in model_lower:
        resources['mem'] = '64G'
        resources['time'] = '00:30:00'
    elif '7b' in model_lower or '8b' in model_lower:
        resources['mem'] = '128G'
        resources['time'] = '01:00:00'
    elif '70b' in model_lower:
        resources['mem'] = '256G'
        resources['time'] = '04:00:00'
        resources['gpus'] = '2'

    return resources


def generate_job_name(run_dir: Path) -> str:
    """Generate a concise job name from the run directory name.

    Examples:
        Llama-3.2-1B-Instruct_5L_rank4 -> 1B_5L_r4
        Llama-3.2-3B-Instruct_9L_rank64 -> 3B_9L_r64
    """
    name = run_dir.name

    # Extract key components
    parts = []

    # Model size
    if '1B' in name:
        parts.append('1B')
    elif '3B' in name:
        parts.append('3B')
    elif '7B' in name or '8B' in name:
        parts.append('8B')
    elif '70B' in name:
        parts.append('70B')

    # Dataset identifier (e.g., 5L, 9L, 13L)
    if '_5L_' in name or name.endswith('_5L'):
        parts.append('5L')
    elif '_9L_' in name or name.endswith('_9L'):
        parts.append('9L')
    elif '_13L_' in name or name.endswith('_13L'):
        parts.append('13L')
    elif '_7L_' in name or name.endswith('_7L'):
        parts.append('7L')
    elif '_8L_' in name or name.endswith('_8L'):
        parts.append('8L')

    # Rank
    if 'rank4' in name:
        parts.append('r4')
    elif 'rank8' in name:
        parts.append('r8')
    elif 'rank16' in name:
        parts.append('r16')
    elif 'rank64' in name:
        parts.append('r64')

    # If we couldn't parse it, just truncate
    if not parts:
        return name[:15]

    return '_'.join(parts)


def detect_recipe(config: Dict[str, Any], run_dir: Path) -> str:
    """Detect which torchtune recipe to use.

    Checks for custom recipes or falls back to standard recipes.
    """
    # Check if setup_finetune.yaml exists and specifies a custom recipe
    setup_file = run_dir / "setup_finetune.yaml"
    if setup_file.exists():
        with open(setup_file) as f:
            setup_config = yaml.safe_load(f)
            if setup_config and 'custom_recipe' in setup_config:
                return setup_config['custom_recipe']

    # Default to custom recipe with validation
    return "cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_val"


def generate_slurm_script(
    run_dir: Path,
    user_config: Dict[str, str],
    resource_map: Optional[Dict[str, Dict[str, str]]] = None,
    template_path: Optional[Path] = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> bool:
    """Generate a SLURM script for a single run directory.

    Returns True if script was generated, False if skipped.
    """
    yaml_file = run_dir / "finetune.yaml"
    slurm_file = run_dir / "finetune.slurm"

    # Check if finetune.yaml exists
    if not yaml_file.exists():
        print(f"  Skipping {run_dir.name}: no finetune.yaml found")
        return False

    # Check if slurm file already exists
    if slurm_file.exists() and not overwrite:
        print(f"  Skipping {run_dir.name}: finetune.slurm already exists (use --overwrite to replace)")
        return False

    # Read the finetune.yaml
    try:
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"  Error reading {yaml_file}: {e}")
        return False

    # Determine resources
    resources = determine_resources(config, run_name=run_dir.name, resource_map=resource_map)
    job_name = generate_job_name(run_dir)
    recipe = detect_recipe(config, run_dir)

    # Generate the SLURM script content
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={resources['cpus']}
#SBATCH --mem={resources['mem']}
#SBATCH --time={resources['time']}
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user={user_config['email']}
#SBATCH --gres=gpu:{resources['gpus']}
#SBATCH --account={user_config['account']}
#SBATCH --partition={user_config['partition']}
#SBATCH --constraint={user_config['constraint']}

module purge
module load anaconda3/2025.6
conda activate {user_config['conda_env']}

OUTPUT_DIR={run_dir.absolute()}/

mkdir -p ${{OUTPUT_DIR}}logs/wandb  # wandb is picky about existing dirs

# Run the fine-tuning with custom recipe
tune run {recipe} \\
    --config finetune.yaml

# Move SLURM log to output dir if successful (only if not already there)
[ $? == 0 ] && [ ! -f ${{OUTPUT_DIR}}/slurm-${{SLURM_JOB_ID}}.out ] && mv slurm-${{SLURM_JOB_ID}}.out ${{OUTPUT_DIR}}/
"""

    # Print what we're doing
    action = "Would generate" if dry_run else "Generating"
    status = "OVERWRITE" if slurm_file.exists() else "NEW"
    print(f"  [{status}] {run_dir.name}: {job_name} ({resources['mem']}, {resources['time']})")

    if dry_run:
        return True

    # Write the file
    with open(slurm_file, 'w') as f:
        f.write(slurm_content)

    # Make it executable
    slurm_file.chmod(0o755)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLURM scripts for fine-tuning experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate scripts for all runs in an experiment
  python tools/slurm/generate_slurm_scripts.py /path/to/experiment/

  # Dry run to preview what would be generated
  python tools/slurm/generate_slurm_scripts.py /path/to/experiment/ --dry-run

  # Overwrite existing scripts
  python tools/slurm/generate_slurm_scripts.py /path/to/experiment/ --overwrite

  # Use custom template
  python tools/slurm/generate_slurm_scripts.py /path/to/experiment/ \\
      --template tools/torchtune/templates/custom_template.slurm
        """
    )
    parser.add_argument(
        'experiment_dir',
        type=Path,
        help='Path to experiment directory containing run subdirectories'
    )
    parser.add_argument(
        '--template',
        type=Path,
        help='Path to SLURM template file (optional, uses built-in template by default)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be generated without writing files'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing finetune.slurm files'
    )
    parser.add_argument(
        '--cruijff-root',
        type=Path,
        default=Path(__file__).parent.parent.parent,
        help='Path to cruijff_kit root directory (for finding claude.local.md)'
    )

    args = parser.parse_args()

    # Validate experiment directory
    if not args.experiment_dir.exists():
        print(f"Error: Experiment directory not found: {args.experiment_dir}")
        sys.exit(1)

    if not args.experiment_dir.is_dir():
        print(f"Error: Not a directory: {args.experiment_dir}")
        sys.exit(1)

    # Parse user configuration
    print(f"Reading user configuration from {args.cruijff_root / 'claude.local.md'}...")
    user_config = parse_local_config(args.cruijff_root)
    print(f"  Account: {user_config['account']}")
    print(f"  Partition: {user_config['partition']}")
    print(f"  Email: {user_config['email']}")
    print(f"  Conda env: {user_config['conda_env']}")
    print()

    # Parse resource estimates from runs_plan.md if available
    print(f"Looking for runs_plan.md in {args.experiment_dir}...")
    resource_map = parse_runs_plan(args.experiment_dir)
    if resource_map:
        print(f"  Found resource estimates for {len(resource_map)} runs in runs_plan.md")
    else:
        print(f"  No runs_plan.md found, will use model-based resource heuristics")
    print()

    # Find all subdirectories with finetune.yaml
    print(f"Scanning {args.experiment_dir} for run directories...")
    run_dirs = []
    for item in sorted(args.experiment_dir.iterdir()):
        if item.is_dir() and (item / "finetune.yaml").exists():
            run_dirs.append(item)

    if not run_dirs:
        print("No run directories found (directories must contain finetune.yaml)")
        sys.exit(1)

    print(f"Found {len(run_dirs)} run directories")
    print()

    # Generate scripts for each run
    if args.dry_run:
        print("DRY RUN MODE - No files will be written")
        print()

    generated = 0
    for run_dir in run_dirs:
        if generate_slurm_script(
            run_dir,
            user_config,
            resource_map=resource_map,
            template_path=args.template,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        ):
            generated += 1

    print()
    if args.dry_run:
        print(f"Would generate {generated} SLURM scripts")
    else:
        print(f"Successfully generated {generated} SLURM scripts")

    return 0


if __name__ == '__main__':
    sys.exit(main())

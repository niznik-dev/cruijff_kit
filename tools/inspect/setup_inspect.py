"""Generate eval SLURM scripts from eval_template.slurm.

Reads experiment-specific config from eval_config.yaml (task script, model
paths, metadata, task args) and combines it with SLURM infrastructure args
from the CLI and model_configs.py to render a complete SLURM script.

Usage:
    python tools/inspect/setup_inspect.py \
        --config eval_config.yaml \
        --model_name Llama-3.2-1B-Instruct \
        --time 0:10:00 \
        --account msalganik
"""

import argparse
import os

import yaml
from pathlib import Path

from cruijff_kit.tools.torchtune.model_configs import MODEL_CONFIGS

# Template lives next to this script
TEMPLATE_PATH = Path(__file__).parent / "templates" / "eval_template.slurm"

# Keys in eval_config.yaml that become -T (task) args in the inspect command
TASK_ARG_KEYS = ["data_path", "config_path", "vis_label", "use_chat_template"]

# Keys in eval_config.yaml that become --metadata args in the inspect command
METADATA_ARG_KEYS = ["epoch", "finetuned", "source_model"]


def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate an eval SLURM script from eval_config.yaml and the eval template."
    )

    # --- Config file (has all experiment-specific values) ---
    parser.add_argument("--config", type=str, required=True,
                        help="Path to eval_config.yaml")

    # --- Model (for SLURM resource lookup) ---
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name key in MODEL_CONFIGS (e.g. 'Llama-3.2-1B-Instruct')")

    # --- SLURM overrides (infrastructure, not experiment-specific) ---
    parser.add_argument("--time", type=str, default="0:10:00",
                        help="SLURM time limit (HH:MM:SS)")
    parser.add_argument("--account", type=str, default=None,
                        help="SLURM account")
    parser.add_argument("--mem", type=str, default=None,
                        help="SLURM memory (overrides model_configs default)")
    parser.add_argument("--partition", type=str, default=None,
                        help="SLURM partition (overrides model_configs default)")
    parser.add_argument("--constraint", type=str, default=None,
                        help="SLURM constraint (overrides model_configs default)")
    parser.add_argument("--conda_env", type=str, default="cruijff",
                        help="Conda environment name")

    # --- Output ---
    parser.add_argument("--output_slurm", type=str, default=None,
                        help="Output filename (default: {task_name}_epoch{epoch}.slurm)")

    return parser


def load_eval_config(config_path):
    """Load and validate eval_config.yaml.

    Returns the config dict with config_path and eval_dir auto-derived.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # Auto-derive eval_dir and config_path from the config file location
    config["eval_dir"] = str(config_path.parent)
    config["config_path"] = str(config_path)

    # Validate required keys
    required = ["task_script", "task_name", "model_path", "model_hf_name", "output_dir"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"eval_config.yaml missing required keys: {', '.join(missing)}")

    return config


def build_task_args(config):
    """Build -T key=value lines for the inspect eval command.

    Reads TASK_ARG_KEYS from config. config_path is always included
    (auto-derived from --config location).

    Returns a string of lines like '  -T data_path="..." \\\n'
    """
    lines = []
    for key in TASK_ARG_KEYS:
        value = config.get(key)
        if value is not None:
            lines.append(f'  -T {key}="{value}" \\')
    return "\n".join(lines) + "\n" if lines else ""


def build_metadata_args(config):
    """Build --metadata key=value lines for the inspect eval command.

    Reads METADATA_ARG_KEYS from config.

    Returns a string of lines like '  --metadata epoch=0 \\\n'
    """
    lines = []
    for key in METADATA_ARG_KEYS:
        value = config.get(key)
        if value is not None:
            lines.append(f'  --metadata {key}="{value}" \\')
    return "\n".join(lines) + "\n" if lines else ""


def render_template(cli_args, config):
    """Read the eval template and perform placeholder replacements.

    Args:
        cli_args: Parsed CLI arguments (SLURM overrides, model_name)
        config: Dict from load_eval_config (experiment-specific values)

    Returns the rendered SLURM script as a string.
    """
    # Validate model name
    if cli_args.model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: '{cli_args.model_name}'. "
            f"Supported models: {', '.join(MODEL_CONFIGS.keys())}"
        )

    model_config = MODEL_CONFIGS[cli_args.model_name]
    slurm_config = model_config.get("slurm", {})

    # Resolve SLURM resources: CLI overrides > model_configs defaults
    mem = cli_args.mem if cli_args.mem else slurm_config.get("mem", "32G")
    cpus = slurm_config.get("cpus", 4)

    # Ensure output_dir ends with /
    output_dir = config["output_dir"]
    if not output_dir.endswith("/"):
        output_dir += "/"

    username = os.environ.get("USER", "unknown")
    task_name = config["task_name"]
    epoch = config.get("epoch")

    # Build task and metadata arg blocks
    task_args = build_task_args(config)
    metadata_args = build_metadata_args(config)

    # Job name: eval-{task_name}-ep{epoch} or eval-{task_name}
    if epoch is not None:
        jobname = f"eval-{task_name}-ep{epoch}"
    else:
        jobname = f"eval-{task_name}"

    # Read template
    with open(TEMPLATE_PATH, "r") as f:
        script = f.read()

    # Placeholder replacements
    script = script.replace("<JOBNAME>", jobname)
    script = script.replace("<MEM>", mem)
    script = script.replace("<TIME>", cli_args.time)
    script = script.replace("<NETID>", username)
    script = script.replace("<CONDA_ENV>", cli_args.conda_env)
    script = script.replace("<OUTPUT_DIR>", output_dir)
    script = script.replace("<EVAL_DIR>", config["eval_dir"])
    script = script.replace("<TASK_SCRIPT>", config["task_script"])
    script = script.replace("<MODEL_HF_NAME>", config["model_hf_name"])
    script = script.replace("<MODEL_PATH>", config["model_path"])
    script = script.replace("<TASK_ARGS>", task_args)
    script = script.replace("<METADATA_ARGS>", metadata_args)

    # CPUs from model config
    script = script.replace("#SBATCH --cpus-per-task=1",
                            f"#SBATCH --cpus-per-task={cpus}")

    # Activate ##SBATCH lines when values provided
    # Account: CLI only (not a model property)
    if cli_args.account:
        script = script.replace("##SBATCH --account=<ACT>",
                                f"#SBATCH --account={cli_args.account}")

    # Partition: CLI overrides model config
    partition = cli_args.partition if cli_args.partition is not None else slurm_config.get("partition")
    if partition is not None and partition != "":
        script = script.replace("##SBATCH --partition=<PART>",
                                f"#SBATCH --partition={partition}")

    # Constraint: CLI overrides model config
    constraint = cli_args.constraint if cli_args.constraint else slurm_config.get("constraint")
    if constraint:
        script = script.replace("##SBATCH --constraint=<CONST>",
                                f"#SBATCH --constraint={constraint}")

    return script


def main():
    """Generate an eval SLURM script from eval_config.yaml and the template."""
    parser = create_parser()
    cli_args = parser.parse_args()

    config = load_eval_config(cli_args.config)
    script = render_template(cli_args, config)

    task_name = config["task_name"]
    epoch = config.get("epoch")

    # Determine output filename
    if cli_args.output_slurm:
        output_path = cli_args.output_slurm
    elif epoch is not None:
        output_path = f"{task_name}_epoch{epoch}.slurm"
    else:
        output_path = f"{task_name}.slurm"

    with open(output_path, "w") as f:
        f.write(script)

    print(f"Wrote eval SLURM script: {output_path}")


if __name__ == "__main__":
    main()

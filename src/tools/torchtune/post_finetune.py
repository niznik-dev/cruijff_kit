"""Post-finetune adapter housekeeping.

Reads the run's finetune.yaml to determine the output_dir, base checkpoint
dir, and adapter-save mode; then walks every `epoch_N` subdirectory and
applies the matching step:

- adapter-only saves (`save_adapter_weights_only: True`): rewrite
  `adapter_config.json`'s `base_model_name_or_path` to the absolute local
  base path so the dir is self-loading on offline compute.
- merged saves (`save_adapter_weights_only: False`): stash adapter files
  into an `adapter_weights/` subdir so transformers' PEFT auto-detection
  doesn't shadow the merged checkpoint.

Invoke as a SLURM post-train step (from the dir containing finetune.yaml):
    python -m cruijff_kit.tools.torchtune.post_finetune --config finetune.yaml
"""

import argparse
import re
from pathlib import Path

from omegaconf import OmegaConf
from torchtune import utils

from cruijff_kit.tools.torchtune.adapter_utils import (
    rewrite_adapter_config_base_path,
    stash_adapter_files,
)

log = utils.get_logger("DEBUG")

EPOCH_RE = re.compile(r"^epoch_(\d+)$")


def read_run_config(cfg_path) -> tuple[Path, str, bool]:
    """Read finetune.yaml via OmegaConf so ${...} interpolations resolve.

    Load with OmegaConf, not yaml.safe_load: finetune.yaml uses ${...}
    interpolation (e.g. checkpoint_dir: ${models_directory}/...) that torchtune
    resolves at training time. Reading the raw yaml leaves the literal
    "${models_directory}" string, which abspath() then roots under the cwd. Accessing
    individual nodes resolves only the interpolations they need.

    Returns (output_dir, base_model_path, save_adapter_weights_only).
    """
    cfg = OmegaConf.load(cfg_path)
    return (
        Path(str(cfg.output_dir)),
        str(cfg.checkpointer.checkpoint_dir),
        bool(cfg.get("save_adapter_weights_only", False)),
    )


def discover_epoch_dirs(output_dir: Path) -> list[int]:
    epochs = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        match = EPOCH_RE.match(child.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="finetune.yaml",
        help="Path to the finetune.yaml the recipe used (defaults to cwd/finetune.yaml)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    output_dir, base_model_path, save_adapter_weights_only = read_run_config(cfg_path)

    if not output_dir.is_dir():
        raise SystemExit(f"output_dir from {cfg_path} not found: {output_dir}")

    epochs = discover_epoch_dirs(output_dir)
    if not epochs:
        log.warning(f"No epoch_N subdirs in {output_dir}; nothing to do.")
        return

    for epoch in epochs:
        if save_adapter_weights_only:
            rewrite_adapter_config_base_path(
                str(output_dir), epoch, base_model_path, log
            )
        else:
            stash_adapter_files(str(output_dir), epoch, log)


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for custom recipes.

!--- cruijff_kit patch ---!
"""

import json
import os
import shutil
from pathlib import Path

from torchtune import utils

log = utils.get_logger("DEBUG")


# Sentinel returned inside check_adapter_base_path()'s message so callers can
# pattern-match for this specific failure mode in addition to surfacing the
# whole human-readable string.
STALE_BASE_PATH_TAG = "STALE_LOCAL_BASE_PATH"


def check_adapter_base_path(adapter_dir) -> str | None:
    """Verify the adapter dir's base_model_name_or_path is loadable here.

    Returns None if the adapter dir is fine, or if the dir has no
    adapter_config.json (i.e. it's a base model or merged checkpoint — not our
    concern). Returns a human-readable problem description if
    base_model_name_or_path is a local absolute path that no longer exists on
    disk.

    HF Hub-style names (e.g. "meta-llama/Llama-3.2-1B-Instruct") are NOT
    checked here — they resolve via HF cache or hub, not the local filesystem.
    If the user is on offline compute without a populated cache, transformers
    will error at load time; that's a separate failure mode.
    """
    adapter_dir = Path(adapter_dir)
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return None

    cfg = json.loads(cfg_path.read_text())
    base = cfg.get("base_model_name_or_path")
    if not base:
        return None

    # Local absolute path → check the filesystem.
    if base.startswith(os.sep) or (len(base) > 1 and base[1] == ":"):
        if not Path(base).exists():
            return (
                f"{STALE_BASE_PATH_TAG}: adapter at {adapter_dir} expects base "
                f"model at {base}, but that path does not exist. The base "
                "model has likely moved. Re-point with `python -m "
                "cruijff_kit.tools.torchtune.port_cruijff_adapter "
                f"{adapter_dir} --repo-id <new_path_or_hf_repo_id>`, or "
                "restore the base model to its original location."
            )

    # HF Hub name (org/name shape) — leave to transformers/HF cache to resolve.
    return None


def rewrite_adapter_config_base_path(
    output_dir: str, epoch: int, base_model_path: str, logger=None
) -> None:
    """Point adapter_config.json's base_model_name_or_path at the local base model.

    Torchtune writes the HF Hub repo name (e.g. 'meta-llama/Llama-3.2-1B-Instruct').
    On offline compute nodes (HF_HUB_OFFLINE=1) transformers can't resolve that, so
    we rewrite it to the absolute path of the local base model. Once rewritten,
    transformers' native PEFT auto-detection (AutoModelForCausalLM.from_pretrained
    on the adapter dir) loads the base + adapter without us emitting a merged
    checkpoint — saving ~base-model-size disk per epoch.

    The original HF Hub repo name is preserved in original_repo_id.json (which
    torchtune already writes); the port_cruijff_adapter utility uses that to
    restore portability when exporting a checkpoint to another machine.
    """
    if logger is None:
        logger = log

    checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch}")
    cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")

    if not os.path.exists(cfg_path):
        logger.warning(
            f"adapter_config.json not found at {cfg_path}; skipping base-path rewrite."
        )
        return

    abs_base = os.path.abspath(base_model_path).rstrip("/")
    with open(cfg_path) as f:
        cfg_data = json.load(f)

    original = cfg_data.get("base_model_name_or_path")
    cfg_data["base_model_name_or_path"] = abs_base
    with open(cfg_path, "w") as f:
        json.dump(cfg_data, f, indent=2)

    logger.info(
        f"Rewrote adapter_config.json base_model_name_or_path: {original} -> {abs_base}"
    )


def stash_adapter_files(output_dir: str, epoch: int, logger=None) -> None:
    """Move adapter files into an `adapter_weights/` subdir of the epoch dir.

    Used when `save_adapter_weights_only=False` (i.e. torchtune wrote both the
    merged base+LoRA checkpoint AND the adapter files side-by-side). With both
    present at the top level, transformers' native PEFT auto-detection wins:
    `AutoModelForCausalLM.from_pretrained(dir)` loads base + adapter and the
    merged `model.safetensors` sits ignored. Moving the adapter files out of
    the way lets the merged checkpoint load as intended.

    The stashed adapter dir is left in its portable PEFT form
    (base_model_name_or_path still the HF Hub repo name as torchtune wrote it)
    — anyone who wants to use the adapter directly can load it from
    `<epoch_dir>/adapter_weights/`.
    """
    if logger is None:
        logger = log

    checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch}")
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    adapter_stash_dir = os.path.join(checkpoint_dir, "adapter_weights")
    os.makedirs(adapter_stash_dir, exist_ok=True)

    adapter_files = [
        "adapter_config.json",
        "adapter_model.pt",
        "adapter_model.safetensors",
    ]
    stashed = 0
    for filename in adapter_files:
        src = os.path.join(checkpoint_dir, filename)
        if os.path.exists(src):
            shutil.move(src, os.path.join(adapter_stash_dir, filename))
            logger.info(f"Stashed {filename} to adapter_weights/ subdirectory")
            stashed += 1

    if stashed == 0:
        logger.info(f"No adapter files found to stash in {checkpoint_dir}")


def validate_epochs_to_save(epochs_to_save, total_epochs: int) -> list[int]:
    """Validate and normalize the cruijff_kit `epochs_to_save` config value.

    Without this guard, a misformatted value (out-of-range index, empty list,
    wrong type) silently produces a run with zero checkpoints — the recipe just
    logs "Skipping checkpoint save" every epoch.

    Accepts the string 'all', or any iterable of ints. Returns a Python list of
    valid epoch indices. Raises ValueError on bad input.
    """
    if isinstance(epochs_to_save, str):
        if epochs_to_save == "all":
            return list(range(total_epochs))
        raise ValueError(
            f"epochs_to_save string value must be 'all', got: {epochs_to_save!r}"
        )

    try:
        epochs_list = list(epochs_to_save)
    except TypeError:
        raise ValueError(
            f"epochs_to_save must be a list of ints or 'all', got "
            f"{type(epochs_to_save).__name__}: {epochs_to_save!r}"
        )

    if not epochs_list:
        raise ValueError(
            "epochs_to_save resolved to an empty list — no checkpoints would be saved. "
            "Set epochs_to_save: 'all' or provide at least one valid epoch index."
        )

    bad = [
        e
        for e in epochs_list
        if isinstance(e, bool) or not isinstance(e, int) or not (0 <= e < total_epochs)
    ]
    if bad:
        raise ValueError(
            f"epochs_to_save contains values outside [0, {total_epochs}) or of wrong type: "
            f"{bad}. total_epochs={total_epochs}."
        )

    return epochs_list

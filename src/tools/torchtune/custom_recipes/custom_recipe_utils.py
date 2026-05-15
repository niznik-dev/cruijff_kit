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

from torchtune import utils

log = utils.get_logger("DEBUG")


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
    torchtune already writes); the uncruijff_adapter utility uses that to restore
    portability when exporting a checkpoint to another machine.
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

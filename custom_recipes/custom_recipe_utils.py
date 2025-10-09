# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for custom recipes.

!--- cruijff-kit patch ---!
"""

import os
import shutil

from torchtune import utils

log = utils.get_logger("DEBUG")


def stash_adapter_files(output_dir: str, epoch: int, logger=None) -> None:
    """
    Move adapter files from the merged model checkpoint directory to a subdirectory
    to prevent inspect-ai from being confused about whether this is a merged model.

    This stashes:
    - adapter_config.json
    - adapter_model.pt
    - adapter_model.safetensors

    into an 'adapter_weights' subdirectory within the checkpoint directory.

    Args:
        output_dir: Base output directory containing checkpoint subdirectories
        epoch: Epoch number for the checkpoint directory
        logger: Optional logger to use (defaults to module-level log)
    """
    if logger is None:
        logger = log

    # Get the checkpoint directory for this epoch
    checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch}")

    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return

    # Create adapter_weights subdirectory
    adapter_stash_dir = os.path.join(checkpoint_dir, "adapter_weights")
    os.makedirs(adapter_stash_dir, exist_ok=True)

    # List of adapter files to stash
    adapter_files = [
        "adapter_config.json",
        "adapter_model.pt",
        "adapter_model.safetensors"
    ]

    stashed_count = 0
    for filename in adapter_files:
        src_path = os.path.join(checkpoint_dir, filename)
        if os.path.exists(src_path):
            dst_path = os.path.join(adapter_stash_dir, filename)
            shutil.move(src_path, dst_path)
            logger.info(f"Stashed {filename} to adapter_weights/ subdirectory")
            stashed_count += 1

    if stashed_count == 0:
        logger.info(f"No adapter files found to stash in {checkpoint_dir}")

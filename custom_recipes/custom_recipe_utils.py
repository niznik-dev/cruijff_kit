# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for custom recipes.

!--- cruijff-kit patch ---!
"""

import json
import os
import tempfile
from typing import Optional

from omegaconf import DictConfig
from torchtune import utils

log = utils.get_logger("DEBUG")


def filter_dataset_by_split(cfg: DictConfig, logger=None) -> Optional[str]:
    """
    Filter JSON dataset by split field if split_key and split_value are provided.
    Returns path to temp file if filtering occurred, None otherwise.
    Only works for JSON files - skips filtering for other sources.

    Args:
        cfg: Dataset configuration containing split_key, split_value, data_files, and source
        logger: Optional logger to use (defaults to module-level log)

    Returns:
        Path to temporary filtered JSON file, or None if no filtering was performed
    """
    if logger is None:
        logger = log

    split_key = cfg.get("split_key")
    split_value = cfg.get("split_value")
    data_files = cfg.get("data_files")
    source = cfg.get("source")

    # Only filter if we have the required params and it's a JSON source
    if split_key is None or split_value is None or data_files is None:
        return None

    if source != "json":
        logger.info(f"Skipping split filtering for non-JSON source: {source}")
        return None

    # Load the JSON file
    with open(data_files, 'r') as f:
        data = json.load(f)

    # Filter by split field
    original_len = len(data)
    filtered_data = [item for item in data if item.get(split_key) == split_value]
    filtered_len = len(filtered_data)

    logger.info(f"Filtered dataset by '{split_key}' field: {original_len} -> {filtered_len} examples ({split_value})")

    # Write to temp file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.json', prefix=f'filtered_{split_value}_')
    with os.fdopen(temp_fd, 'w') as f:
        json.dump(filtered_data, f)

    return temp_path

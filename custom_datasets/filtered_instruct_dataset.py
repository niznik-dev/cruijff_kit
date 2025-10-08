# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset
from torchtune.datasets import instruct_dataset


def filtered_instruct_dataset(
    tokenizer,
    source: str,
    split_key: str = None,
    split_value: str = None,
    **kwargs,
):
    """
    Wrapper around torchtune's instruct_dataset that filters by a split field.

    This allows loading a single JSON file with a column for splits and filtering
    to only specific split values (e.g., 'train' or 'validation').

    Args:
        tokenizer: The tokenizer to use for tokenization
        source: Source of the dataset (e.g., "json" or HuggingFace dataset name)
        split_key: Name of the field/column to filter on (e.g., "split")
        split_value: Value to filter for (e.g., "train" or "validation")
        **kwargs: Additional arguments passed to instruct_dataset and load_dataset

    Returns:
        The filtered and processed dataset

    Example YAML config:
        dataset:
          _component_: custom_datasets.filtered_instruct_dataset.filtered_instruct_dataset
          source: "json"
          data_files: "${input_dir}/${dataset_filename}"
          split_key: "split"
          split_value: "train"
          packed: True
    """
    # If no filtering requested, just use the regular instruct_dataset
    if split_key is None or split_value is None:
        return instruct_dataset(tokenizer=tokenizer, source=source, **kwargs)

    # Load the raw dataset first
    # Extract load_dataset specific kwargs
    data_files = kwargs.pop('data_files', None)
    split = kwargs.pop('split', 'train')

    # Load raw HuggingFace dataset
    ds = load_dataset(source, data_files=data_files, split=split)

    # Filter by the split field
    original_len = len(ds)
    ds = ds.filter(lambda x: x.get(split_key) == split_value)
    filtered_len = len(ds)

    print(f"Filtered dataset by '{split_key}' field: {original_len} -> {filtered_len} examples ({split_value})")

    # Now pass the filtered dataset to instruct_dataset
    # We need to pass it as a pre-loaded dataset, not reload from source
    return instruct_dataset(
        tokenizer=tokenizer,
        source=ds,  # Pass the filtered dataset directly
        **kwargs
    )

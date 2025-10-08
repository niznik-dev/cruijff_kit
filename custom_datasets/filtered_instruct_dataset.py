# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset
from torchtune.datasets._sft import SFTDataset
from torchtune.data import InputOutputToMessages


def filtered_instruct_dataset(
    tokenizer,
    source: str,
    split_key: str = None,
    split_value: str = None,
    **kwargs,
):
    """
    Dataset that filters by a split field before applying instruction formatting.

    This allows loading a single JSON file with a column for splits and filtering
    to only specific split values (e.g., 'train' or 'validation').

    Args:
        tokenizer: The tokenizer to use for tokenization
        source: Source of the dataset (e.g., "json" or HuggingFace dataset name)
        split_key: Name of the field/column to filter on (e.g., "split")
        split_value: Value to filter for (e.g., "train" or "validation")
        **kwargs: Additional arguments passed to SFTDataset and load_dataset

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
    # Extract dataset-specific kwargs
    column_map = kwargs.pop('column_map', None)
    new_system_prompt = kwargs.pop('new_system_prompt', None)
    packed = kwargs.pop('packed', False)
    split = kwargs.pop('split', 'train')
    train_on_input = kwargs.pop('train_on_input', False)

    # Load the raw dataset (kwargs contains load_dataset args like data_files)
    ds = load_dataset(source, split=split, **kwargs)

    # Filter by the split field if specified
    if split_key is not None and split_value is not None:
        original_len = len(ds)
        ds = ds.filter(lambda x: x.get(split_key) == split_value)
        filtered_len = len(ds)
        print(f"Filtered dataset by '{split_key}' field: {original_len} -> {filtered_len} examples ({split_value})")

    # Create message transform (same as instruct_dataset does)
    message_transform = InputOutputToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )

    # Create SFTDataset with a dummy source to prevent it from loading
    # We'll override _data immediately after
    sft_ds = SFTDataset.__new__(SFTDataset)
    sft_ds._data = ds
    sft_ds._message_transform = message_transform
    sft_ds._model_transform = tokenizer
    sft_ds.packed = packed

    return sft_ds

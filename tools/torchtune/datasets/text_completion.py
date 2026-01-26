"""
TextCompletionDataset: Fine-tuning dataset for base/foundation models.

Uses simple text concatenation (no chat template) for models without
instruction-following capabilities. Ensures exact tokenization parity
with evaluation by using HuggingFace tokenizer directly.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class TextCompletionConfig:
    """Configuration for text completion dataset."""

    model_path: str  # Path to HuggingFace model (for loading tokenizer)
    prompt: str = "{input}"  # Format string to wrap input
    input_key: str = "input"
    output_key: str = "output"
    max_length: Optional[int] = None
    train_on_input: bool = False  # False = output only, True = full sequence


class TextCompletionDataset(Dataset):
    """
    Fine-tuning dataset for base/foundation models.

    Uses simple text concatenation: prompt.format(**row) + output
    No chat template is applied, matching evaluation for base models.

    Returns {"tokens": tensor, "labels": tensor} where labels has -100
    for prompt tokens (not trained) and token IDs for output response.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,  # Ignored - we load our own HF tokenizer
        *,
        cfg: TextCompletionConfig,
    ):
        self.rows = rows
        self.cfg = cfg

        # Load HuggingFace tokenizer (ignore passed torchtune tokenizer)
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(cfg.model_path)

        # Ensure we have a pad token
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]

        # 1. Format prompt with input
        formatted_prompt = self.cfg.prompt.format(**row)
        output = row[self.cfg.output_key]

        # 2. Tokenize prompt only (for masking calculation)
        prompt_ids = self.tok.encode(formatted_prompt, add_special_tokens=True)

        # 3. Tokenize full sequence: prompt + output
        full_text = formatted_prompt + output
        full_ids = self.tok.encode(full_text, add_special_tokens=True)

        # 4. Determine prompt length for masking
        # Note: Due to tokenization, prompt_ids may not be exact prefix of full_ids,
        # but the lengths should be usable for masking
        prompt_len = len(prompt_ids)

        # 5. Create labels
        if self.cfg.train_on_input:
            labels = full_ids.copy()
        else:
            # Mask prompt tokens, train on output only
            labels = [-100] * prompt_len + full_ids[prompt_len:]

        # 6. Optional truncation
        if self.cfg.max_length is not None:
            full_ids = full_ids[: self.cfg.max_length]
            labels = labels[: self.cfg.max_length]

        return {
            "tokens": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def text_completion_dataset(
    tokenizer,  # Ignored - we load our own HF tokenizer
    *,
    source: str,
    data_files: str,
    model_path: str,  # Required: path to HF model for tokenizer
    prompt: str = "{input}",
    input_key: str = "input",
    output_key: str = "output",
    max_length: Optional[int] = None,
    train_on_input: bool = False,
    split: str = "train",
    packed: bool = False,  # Accepted but not used - recipe reads this for collate_fn selection
) -> TextCompletionDataset:
    """
    Factory function for torchtune YAML instantiation.

    Usage in finetune.yaml:
        dataset:
          _component_: cruijff_kit.tools.torchtune.datasets.text_completion.text_completion_dataset
          source: json
          data_files: /path/to/data.json
          model_path: /path/to/Llama-3.2-1B
          split: train
          prompt: "{input}"
          input_key: input
          output_key: output
          train_on_input: false

    Note: The `tokenizer` argument is ignored. This dataset loads its own
    HuggingFace tokenizer from model_path to ensure consistent tokenization
    between training and evaluation (which also uses HF tokenizer).

    Note: No system_prompt parameter - base models don't use chat format.
    The prompt template should include any prefixes needed.
    """
    if source != "json":
        raise ValueError(f"Only 'json' source supported, got: {source}")

    with open(data_files) as f:
        data = json.load(f)

    # Handle both flat list and {"train": [...], "validation": [...]} formats
    if isinstance(data, dict) and split in data:
        rows = data[split]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unexpected JSON structure: expected list or dict with '{split}' key")

    cfg = TextCompletionConfig(
        model_path=model_path,
        prompt=prompt,
        input_key=input_key,
        output_key=output_key,
        max_length=max_length,
        train_on_input=train_on_input,
    )

    return TextCompletionDataset(rows, tokenizer, cfg=cfg)

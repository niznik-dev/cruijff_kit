"""
ChatCompletionDataset: Fine-tuning dataset using HuggingFace chat templates.

Uses HuggingFace's apply_chat_template() to ensure exact tokenization parity
with inspect-ai evaluation.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class ChatCompletionConfig:
    """Configuration for chat completion dataset."""

    model_path: str  # Path to HuggingFace model (for loading tokenizer)
    prompt: str = (
        "{input}"  # Format string to wrap input before placing in user message
    )
    system_prompt: str = ""  # Optional system message
    input_key: str = "input"
    output_key: str = "output"
    max_length: Optional[int] = None
    train_on_input: bool = False  # False = output only, True = full sequence


class ChatCompletionDataset(Dataset):
    """
    Fine-tuning dataset that applies HuggingFace chat templates.

    Ensures training format matches inspect-ai evaluation format by using
    the exact same apply_chat_template() code path.

    Returns {"tokens": tensor, "labels": tensor} where labels has -100
    for prompt tokens (not trained) and token IDs for assistant response.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,  # Ignored - we load our own HF tokenizer
        *,
        cfg: ChatCompletionConfig,
    ):
        self.rows = rows
        self.cfg = cfg

        # Load HuggingFace tokenizer (ignore passed torchtune tokenizer)
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(cfg.model_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]

        # 1. Wrap input with prompt template
        wrapped_input = self.cfg.prompt.format(**row)
        output = row[self.cfg.output_key]

        # 2. Build chat messages (without assistant response)
        messages = []
        if self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.append({"role": "user", "content": wrapped_input})

        # 3. Tokenize prompt (with generation prompt to get assistant header)
        prompt_ids = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )

        # 4. Add assistant response and tokenize full sequence
        messages.append({"role": "assistant", "content": output})
        full_ids = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
        )

        # 5. Verify prompt is prefix of full (sanity check)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Prompt tokenization mismatch at idx {idx}. "
                f"Prompt length: {len(prompt_ids)}, Full length: {len(full_ids)}. "
                "This suggests apply_chat_template behaves differently with/without assistant message."
            )

        # 6. Create labels
        if self.cfg.train_on_input:
            labels = full_ids.copy()
        else:
            # Mask prompt tokens, train on assistant response only
            prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + full_ids[prompt_len:]

        # 7. Optional truncation
        if self.cfg.max_length is not None:
            full_ids = full_ids[: self.cfg.max_length]
            labels = labels[: self.cfg.max_length]

        return {
            "tokens": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def chat_completion_dataset(
    tokenizer,  # Ignored - we load our own HF tokenizer
    *,
    source: str,
    data_files: str,
    model_path: str,  # Required: path to HF model for tokenizer
    prompt: str = "{input}",
    system_prompt: str = "",
    input_key: str = "input",
    output_key: str = "output",
    max_length: Optional[int] = None,
    train_on_input: bool = False,
    split: str = "train",
    packed: bool = False,  # Accepted but not used - recipe reads this for collate_fn selection
) -> ChatCompletionDataset:
    """
    Factory function for torchtune YAML instantiation.

    Usage in finetune.yaml:
        dataset:
          _component_: cruijff_kit.tools.torchtune.datasets.chat_completion.chat_completion_dataset
          source: json
          data_files: /path/to/data.json
          model_path: /path/to/Llama-3.2-1B-Instruct
          split: train
          prompt: "{input}"
          system_prompt: ""
          input_key: input
          output_key: output
          train_on_input: false

    Note: The `tokenizer` argument is ignored. This dataset loads its own
    HuggingFace tokenizer from model_path to ensure apply_chat_template()
    is available and matches inspect-ai's tokenization.
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
        raise ValueError(
            f"Unexpected JSON structure: expected list or dict with '{split}' key"
        )

    cfg = ChatCompletionConfig(
        model_path=model_path,
        prompt=prompt,
        system_prompt=system_prompt,
        input_key=input_key,
        output_key=output_key,
        max_length=max_length,
        train_on_input=train_on_input,
    )

    return ChatCompletionDataset(rows, tokenizer, cfg=cfg)

"""
ConditionalCompletionDataset: Template-agnostic fine-tuning dataset.

Bypasses chat templates to give explicit control over tokenization,
ensuring training/inference parity.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

PromptSpec = Union[str, Callable[[Dict[str, Any]], str]]


@dataclass
class ConditionalCompletionConfig:
    """Configuration for conditional completion dataset."""

    prompt: PromptSpec  # format string or callable
    input_key: str = "input"
    output_key: str = "output"
    add_bos: bool = True
    add_eos: bool = True
    max_length: Optional[int] = None
    train_on_input: bool = False  # False = target only, True = full sequence
    separator: str = ""  # between prompt and target


class ConditionalCompletionDataset(Dataset):
    """
    Builds: prompt(example) + separator + output
    Labels: -100 for prompt tokens, real IDs for output tokens.
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer,
        cfg: ConditionalCompletionConfig,
    ):
        self.rows = rows
        self.tok = tokenizer
        self.cfg = cfg

        # torchtune tokenizers use *_id, HF uses *_token_id
        self.bos_id = getattr(tokenizer, "bos_id", None) or getattr(tokenizer, "bos_token_id", None)
        self.eos_id = getattr(tokenizer, "eos_id", None) or getattr(tokenizer, "eos_token_id", None)
        self.pad_id = getattr(tokenizer, "pad_id", None) or getattr(tokenizer, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = self.eos_id

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, text: str) -> List[int]:
        """Encode text without special tokens. Handles both torchtune and HF tokenizers."""
        if hasattr(self.tok, "encode"):
            # torchtune tokenizer
            return self.tok.encode(text, add_bos=False, add_eos=False)
        else:
            # HuggingFace tokenizer
            return self.tok(text, add_special_tokens=False)["input_ids"]

    def _render_prompt(self, row: Dict[str, Any]) -> str:
        if callable(self.cfg.prompt):
            return self.cfg.prompt(row)
        return self.cfg.prompt.format(**row)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        out = row[self.cfg.output_key]

        prompt = self._render_prompt(row)
        prefix = prompt + self.cfg.separator
        full_text = prefix + out

        # Tokenize without special tokens so boundary is exact
        prefix_ids = self._encode(prefix)
        full_ids = self._encode(full_text)

        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"Prefix tokenization mismatch at idx {idx}. "
                "Check prompt rendering or tokenizer behavior."
            )

        bos = [self.bos_id] if self.cfg.add_bos and self.bos_id is not None else []
        eos = [self.eos_id] if self.cfg.add_eos and self.eos_id is not None else []

        input_ids = bos + full_ids + eos

        if self.cfg.train_on_input:
            labels = input_ids.copy()
        else:
            # target_only: mask input tokens with -100
            prefix_len = len(bos) + len(prefix_ids)
            labels = [-100] * prefix_len + full_ids[len(prefix_ids) :] + eos

        if self.cfg.max_length is not None:
            input_ids = input_ids[: self.cfg.max_length]
            labels = labels[: self.cfg.max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()

        return {
            "tokens": input_ids,
            "labels": labels,
        }


def pad_collate(batch: List[Dict[str, torch.Tensor]], pad_token_id: int):
    """Collate function that pads sequences to max length in batch."""
    max_len = max(x["input_ids"].shape[0] for x in batch)

    def pad1(t: torch.Tensor, pad_val: int) -> torch.Tensor:
        n = max_len - t.shape[0]
        if n <= 0:
            return t
        return torch.cat([t, torch.full((n,), pad_val, dtype=t.dtype)], dim=0)

    input_ids = torch.stack([pad1(b["input_ids"], pad_token_id) for b in batch])
    labels = torch.stack([pad1(b["labels"], -100) for b in batch])
    attention_mask = torch.stack([pad1(b["attention_mask"], 0) for b in batch])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def conditional_completion_dataset(
    tokenizer,
    source: str,
    data_files: str,
    prompt: str,
    input_key: str = "input",
    output_key: str = "output",
    add_bos: bool = True,
    add_eos: bool = True,
    max_length: Optional[int] = None,
    train_on_input: bool = False,
    separator: str = "",
    field: str = "train",
) -> ConditionalCompletionDataset:
    """
    Factory function for torchtune YAML instantiation.

    Usage in finetune.yaml:
        dataset:
          _component_: cruijff_kit.tools.torchtune.datasets.conditional_completion.conditional_completion_dataset
          source: json
          data_files: /path/to/data.json
          field: train
          prompt: "Capitalize: {input}\n"
          input_key: input
          output_key: output
          train_on_input: false
    """
    if source != "json":
        raise ValueError(f"Only 'json' source supported, got: {source}")

    with open(data_files) as f:
        data = json.load(f)

    # Handle both flat list and {"train": [...], "validation": [...]} formats
    if isinstance(data, dict) and field in data:
        rows = data[field]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unexpected JSON structure: expected list or dict with '{field}' key")

    cfg = ConditionalCompletionConfig(
        prompt=prompt,
        input_key=input_key,
        output_key=output_key,
        add_bos=add_bos,
        add_eos=add_eos,
        max_length=max_length,
        train_on_input=train_on_input,
        separator=separator,
    )

    return ConditionalCompletionDataset(rows, tokenizer, cfg)

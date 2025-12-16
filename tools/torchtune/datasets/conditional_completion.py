"""
ConditionalCompletionDataset: Template-agnostic fine-tuning dataset.

Bypasses chat templates to give explicit control over tokenization,
ensuring training/inference parity.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from torch.utils.data import Dataset

PromptSpec = Union[str, Callable[[dict[str, Any]], str]]


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
        rows: list[dict[str, Any]],
        tokenizer,  # Supports both torchtune ModelTokenizer and HuggingFace tokenizers
        *,
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

    def _encode(self, text: str) -> list[int]:
        """Encode text without special tokens. Handles both torchtune and HF tokenizers."""
        if hasattr(self.tok, "encode"):
            # torchtune tokenizer
            return self.tok.encode(text, add_bos=False, add_eos=False)
        else:
            # HuggingFace tokenizer
            return self.tok(text, add_special_tokens=False)["input_ids"]

    def _render_prompt(self, row: dict[str, Any]) -> str:
        if callable(self.cfg.prompt):
            return self.cfg.prompt(row)
        return self.cfg.prompt.format(**row)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
            # Mask prompt tokens so loss is only computed on output + EOS.
            #
            # Example with prompt="Capitalize: {input}\n", input="hello", output="HELLO":
            #   prefix = "Capitalize: hello\n"
            #   full_text = "Capitalize: hello\nHELLO"
            #
            # After tokenization (hypothetical token IDs):
            #   bos = [1]
            #   prefix_ids = [10, 20, 30, 40]       # "Capitalize: hello\n"
            #   full_ids = [10, 20, 30, 40, 50, 60] # "Capitalize: hello\nHELLO"
            #   eos = [2]
            #
            # Result:
            #   input_ids: [1,    10,   20,   30,   40,   50, 60, 2]
            #   labels:    [-100, -100, -100, -100, -100, 50, 60, 2]
            #              ←───── ignored by loss ─────→ ←─trained─→
            #
            # The -100 is PyTorch CrossEntropyLoss's ignore_index, so we only
            # compute loss on the output tokens and EOS.
            prefix_len = len(bos) + len(prefix_ids)
            labels = [-100] * prefix_len + full_ids[len(prefix_ids) :] + eos

        if self.cfg.max_length is not None:
            input_ids = input_ids[: self.cfg.max_length]
            labels = labels[: self.cfg.max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # Note: torchtune's SFTDataset returns 'tokens', 'mask', and 'labels'.
        # We skip 'mask' because we compute labels directly rather than using
        # a mask to derive them. Training loops only need 'tokens' and 'labels'.
        return {
            "tokens": input_ids,
            "labels": labels,
        }



def conditional_completion_dataset(
    tokenizer,
    *,
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
    split: str = "train",
) -> ConditionalCompletionDataset:
    """
    Factory function for torchtune YAML instantiation.

    Usage in finetune.yaml:
        dataset:
          _component_: cruijff_kit.tools.torchtune.datasets.conditional_completion.conditional_completion_dataset
          source: json
          data_files: /path/to/data.json
          split: train
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
    if isinstance(data, dict) and split in data:
        rows = data[split]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unexpected JSON structure: expected list or dict with '{split}' key")

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

    return ConditionalCompletionDataset(rows, tokenizer, cfg=cfg)

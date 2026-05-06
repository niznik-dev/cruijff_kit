#!/usr/bin/env python
"""
Quick spot-check inference on fine-tuned models.

Uses the same model-loading pattern as inspect-ai's HF backend so that
"what you see in spot_check" matches "what inspect would do."

Usage:
    python -m cruijff_kit.utils.spot_check \
        --model /path/to/checkpoint \
        --data /path/to/eval_data.json \
        --n 5

    # With split (for nested JSON with train/validation keys)
    python -m cruijff_kit.utils.spot_check \
        --model /path/to/model \
        --data /path/to/data.json \
        --split validation \
        --n 5

Or from Python:
    from cruijff_kit.utils.spot_check import spot_check
    spot_check(model_path, data_path, n=5, split="validation")
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_data(data_path: str, n: int, split: str = None):
    """
    Load prompts and targets from JSON file.

    Supports two formats:
    1. Flat list: [{"input": ..., "output": ...}, ...]
    2. Nested with splits: {"train": [...], "validation": [...]}

    Args:
        data_path: Path to JSON file
        n: Number of samples to load
        split: Optional split name (e.g., "train", "validation")

    Returns:
        tuple: (prompts, targets) lists
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    # Handle nested format with splits
    if split:
        if split not in data:
            raise ValueError(
                f"Split '{split}' not found. Available: {list(data.keys())}"
            )
        data = data[split]
    elif isinstance(data, dict) and "train" in data:
        # Auto-detect nested format, default to validation
        print("  (Auto-detected nested format, using 'validation' split)")
        data = data.get("validation", data.get("train"))

    # Extract prompts and targets
    data = data[:n]
    prompts = [ex["input"] for ex in data]
    targets = [ex["output"] for ex in data]

    return prompts, targets


def _load_model(model_path: str):
    """
    Load model and tokenizer using the same pattern as inspect-ai's HF backend.

    Returns:
        tuple: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model.eval()

    return tokenizer, model


def _generate_one(
    model,
    tokenizer,
    prompt: str,
    sysprompt: str = "",
    use_chat_template: bool = True,
    max_new_tokens: int = 10,
) -> str:
    """
    Generate a response for a single prompt, mirroring inspect-ai's HF flow.

    Returns:
        The generated text (input tokens stripped).
    """
    if use_chat_template:
        messages = []
        if sysprompt:
            messages.append({"role": "system", "content": sysprompt})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Strip input tokens, decode only generated portion
    generated_ids = output_ids[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def spot_check(
    model_path: str,
    data_path: str,
    n: int = 5,
    split: str = None,
    max_new_tokens: int = 10,
    sysprompt: str = "",
    use_chat_template: bool = True,
    show_full_prompt: bool = False,
):
    """
    Run quick inference on n samples and display results.

    Args:
        model_path: Path to HF model or checkpoint directory
        data_path: Path to JSON data file with 'input' and 'output' keys
        n: Number of samples to evaluate
        split: Data split to use (e.g., "train", "validation"). Auto-detects if None.
        max_new_tokens: Max tokens to generate
        sysprompt: System prompt
        use_chat_template: Whether to use chat template
        show_full_prompt: If True, show full prompt text (can be long)

    Returns:
        List of dicts with prompt, expected, and generated outputs
    """
    print(f"\n{'=' * 60}")
    print("SPOT CHECK")
    print(f"{'=' * 60}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Samples: {n}")
    print(f"{'=' * 60}\n")

    # Load model
    print("Loading model...")
    tokenizer, model = _load_model(model_path)
    print(f"Model loaded on {model.device}\n")

    # Load data
    prompts, targets = load_data(data_path, n=n, split=split)

    # Generate one at a time
    print("Generating responses...")
    outputs = []
    for i, prompt in enumerate(prompts):
        generated = _generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            sysprompt=sysprompt,
            use_chat_template=use_chat_template,
            max_new_tokens=max_new_tokens,
        )
        outputs.append(generated)
        print(f"  [{i + 1}/{n}] done")

    # Display results
    results = []
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}\n")

    for i, (prompt, expected, generated) in enumerate(zip(prompts, targets, outputs)):
        result = {
            "prompt": prompt,
            "expected": expected,
            "generated": generated.strip(),
            "match": expected.strip() == generated.strip(),
        }
        results.append(result)

        print(f"[{i + 1}/{n}]")
        if show_full_prompt:
            print(f"  Prompt:    {prompt}")
        else:
            # Show truncated prompt
            prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
            print(f"  Prompt:    {prompt_preview}")
        print(f"  Expected:  '{expected}'")
        print(f"  Generated: '{generated.strip()}'")
        print(f"  Match:     {'✓' if result['match'] else '✗'}")
        print()

    # Summary
    matches = sum(1 for r in results if r["match"])
    print(f"{'=' * 60}")
    print(f"SUMMARY: {matches}/{n} exact matches ({100 * matches / n:.1f}%)")
    print(f"{'=' * 60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Quick spot-check inference on fine-tuned models"
    )
    parser.add_argument(
        "--model", required=True, help="Path to HF model or checkpoint directory"
    )
    parser.add_argument("--data", required=True, help="Path to JSON data file")
    parser.add_argument(
        "--split", default=None, help="Data split (e.g., 'train', 'validation')"
    )
    parser.add_argument(
        "--n", type=int, default=5, help="Number of samples (default: 5)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10,
        help="Max tokens to generate (default: 10)",
    )
    parser.add_argument("--sysprompt", default="", help="System prompt")
    parser.add_argument(
        "--no-chat-template", action="store_true", help="Disable chat template"
    )
    parser.add_argument(
        "--show-full-prompt", action="store_true", help="Show full prompt text"
    )

    args = parser.parse_args()

    spot_check(
        model_path=args.model,
        data_path=args.data,
        n=args.n,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        sysprompt=args.sysprompt,
        use_chat_template=not args.no_chat_template,
        show_full_prompt=args.show_full_prompt,
    )


if __name__ == "__main__":
    main()

"""
Synthetic Twins Zygosity Prediction - Inspect-AI Evaluation Task

Evaluates model performance on predicting twin zygosity (monozygotic vs dizygotic)
from trait values.

Usage:
    # Evaluate fine-tuned model (reads from setup_finetune.yaml)
    inspect eval experiments/synthetic_twins/twins_task.py --model hf/local \\
        -M model_path=/path/to/model \\
        -T config_dir=/path/to/epoch_0

    # Standalone evaluation with direct dataset path
    inspect eval experiments/synthetic_twins/twins_task.py --model hf/local \\
        -M model_path=/path/to/model \\
        -T dataset_path=/path/to/twin_zygosity.json

    # Evaluate on validation split instead of test
    inspect eval experiments/synthetic_twins/twins_task.py --model hf/local \\
        -M model_path=/path/to/model \\
        -T config_dir=/path/to/epoch_0 \\
        -T split=validation
"""

from __future__ import annotations
from typing import Optional, Sequence
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.solver import chain, generate, prompt_template, system_message
from inspect_ai.scorer import match, includes
import yaml


@task
def twins_zygosity(
    config_dir: Optional[str] = None,
    dataset_path: Optional[str] = None,
    system_prompt: str = "",
    temperature: float = 1e-7,
    split: str = "test",
    max_tokens: int = 10
) -> Task:
    """
    Evaluation task for twin zygosity prediction.

    Expected output format: "0" (dizygotic) or "1" (monozygotic)

    Args:
        config_dir: Path to epoch directory (contains ../setup_finetune.yaml).
                   If provided, reads dataset path and system prompt from config.
        dataset_path: Direct path to dataset JSON file. Used if config_dir not provided.
        system_prompt: System message for the model. Overrides config if both provided.
        temperature: Generation temperature (default: 1e-7 for near-deterministic output).
        split: Which data split to use - "train", "validation", or "test" (default: "test").
        max_tokens: Maximum tokens to generate (default: 10, sufficient for binary output).

    Returns:
        Task: Configured inspect-ai task

    Raises:
        ValueError: If neither config_dir nor dataset_path is provided
        FileNotFoundError: If specified files don't exist
    """

    # Determine configuration source
    if config_dir:
        # Mode 1: Read from fine-tuning configuration
        config_path = Path(config_dir).parent / "setup_finetune.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Make sure config_dir points to an epoch directory with ../setup_finetune.yaml"
            )

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        try:
            # Extract dataset path from config
            dataset_ext = config['dataset_ext']
            if dataset_ext == '.json':
                data_path = config['input_dir_base'] + config['dataset_label'] + '.json'
            else:
                # For parquet or other formats
                data_path = config['input_dir_base'] + config['dataset_label'] + dataset_ext

            # Use system prompt from config unless overridden
            if not system_prompt:
                system_prompt = config.get('system_prompt', '')

        except KeyError as e:
            raise KeyError(f"Missing required key in setup_finetune.yaml: {e}")

    elif dataset_path:
        # Mode 2: Direct dataset path
        data_path = dataset_path

        if not Path(data_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

    else:
        raise ValueError(
            "Must provide either config_dir or dataset_path.\n"
            "Examples:\n"
            "  -T config_dir=/path/to/epoch_0\n"
            "  -T dataset_path=/path/to/twin_zygosity.json"
        )

    # Handle system prompt if it was parsed as a list (CLI quirk)
    if isinstance(system_prompt, Sequence) and not isinstance(system_prompt, str):
        system_prompt = ",".join(system_prompt)

    # Define record to sample conversion
    def record_to_sample(record):
        return Sample(
            input=record["input"],
            target=record["output"]
        )

    # Load dataset
    # Assumes JSON format with structure: {"train": [...], "validation": [...], "test": [...]}
    dataset = hf_dataset(
        path="json",
        data_files=data_path,
        field=split,  # Access the specified split (e.g., "test")
        split="train",  # Top-level split (HuggingFace quirk - always use "train" here)
        sample_fields=record_to_sample
    )

    # Build solver chain
    solver_chain = chain(
        system_message(system_prompt),
        prompt_template("{prompt}"),
        generate(
            temperature=temperature,
            max_tokens=max_tokens
        )
    )

    # Configure scorers for binary classification
    # Using both exact match and includes for robustness
    scorers = [
        match(location="exact", ignore_case=False),  # Exact match: "0" or "1"
        includes(ignore_case=False)  # Substring match (for verbose responses)
    ]

    return Task(
        dataset=dataset,
        solver=solver_chain,
        scorer=scorers,
    )

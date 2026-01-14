"""
Eval task for ACS income prediction with chat_completion dataset.

Reads prompt and system_prompt from setup_finetune.yaml to ensure train/eval parity.

Usage:
    inspect eval inspect_task_acs_income.py --model hf/local \
        -M model_path=/path/to/checkpoint \
        -T config_path=/path/to/setup_finetune.yaml \
        -T data_path=/path/to/acs_income_condensed_6000_83P.json
"""

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.solver import chain, generate, system_message
from inspect_ai.scorer import match, includes


@task
def acs_income(
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 5,
) -> Task:
    """
    Eval task for ACS income prediction (>$50k) trained with chat_completion.

    Args:
        data_path: Path to JSON file with {"train": [...], "validation": [...], "test": [...]}
        config_path: Path to setup_finetune.yaml (reads prompt/system_prompt from it)
        split: Which split to evaluate on (default: test)
        temperature: Generation temperature
        max_tokens: Max tokens to generate
    """
    # Read prompt config from YAML
    prompt_str = "{input}"
    system_prompt = ""

    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        prompt_str = config.get('prompt', '{input}')
        system_prompt = config.get('system_prompt', '')

    def record_to_sample(record):
        # Wrap input with prompt template - same as chat_completion training
        formatted_input = prompt_str.format(input=record["input"])
        return Sample(
            input=formatted_input,
            target=record["output"]
        )

    dataset = hf_dataset(
        path="json",
        data_files=data_path,
        field=split,
        split="train",  # HuggingFace quirk - always "train" here
        sample_fields=record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=chain(
            system_message(system_prompt),
            generate(temperature=temperature, max_tokens=max_tokens),
        ),
        scorer=[
            match(location="exact", ignore_case=False),
            includes(ignore_case=False),
        ],
    )

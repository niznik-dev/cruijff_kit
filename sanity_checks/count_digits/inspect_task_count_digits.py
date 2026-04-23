"""
Eval task for count_digits.

Given a comma-delimited sequence of single digits, the model must respond
with the count of digits in the sequence.

Usage:
    inspect eval inspect_task_count_digits.py@count_digits \
        --model hf/local \
        -M model_path=/path/to/checkpoint \
        -T data_path=/path/to/count_digits.json \
        -T config_path=/path/to/eval_config.yaml
"""

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.scorer import match, includes
from inspect_ai.solver import chain, generate, system_message


@task
def count_digits(
    data_path: str,
    config_path: str = "",
    split: str = "validation",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    use_chat_template: bool = True,
) -> Task:
    """
    Eval task for count_digits.

    Args:
        data_path: Path to JSON file with {"train": [...], "validation": [...]}
        config_path: Path to eval config YAML (reads prompt/system_prompt)
        split: Which split to evaluate on (default: validation)
        temperature: Generation temperature
        max_tokens: Max tokens to generate
        use_chat_template: If True, use chat format with system message.
    """
    prompt_str = "{input}"
    system_prompt = ""

    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        prompt_str = config.get("prompt", "{input}")
        system_prompt = config.get("system_prompt", "")

    def record_to_sample(record):
        formatted_input = prompt_str.format(input=record["input"])
        return Sample(input=formatted_input, target=record["output"])

    dataset = hf_dataset(
        path="json",
        data_files=data_path,
        field=split,
        split="train",
        sample_fields=record_to_sample,
    )

    if use_chat_template:
        solver = chain(
            system_message(system_prompt),
            generate(temperature=temperature, max_tokens=max_tokens),
        )
    else:
        solver = chain(
            generate(temperature=temperature, max_tokens=max_tokens),
        )

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=[
            match(location="exact", ignore_case=False),
            includes(ignore_case=False),
        ],
    )

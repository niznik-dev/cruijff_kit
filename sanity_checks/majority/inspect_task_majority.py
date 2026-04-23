"""
Eval task for majority classification.

Given a comma-delimited sequence of 0s and 1s, the model must respond
with whichever digit appears more often.

Usage:
    inspect eval inspect_task_majority.py@majority \
        --model hf/local \
        -M model_path=/path/to/checkpoint \
        -T data_path=/path/to/majority.json \
        -T config_path=/path/to/eval_config.yaml
"""

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import chain, generate, system_message

from cruijff_kit.tools.inspect.scorers import build_scorers


@task
def majority(
    data_path: str,
    config_path: str = "",
    split: str = "validation",
    temperature: float = 1e-7,
    max_tokens: int = 3,
    use_chat_template: bool = True,
    vis_label: str = "",
) -> Task:
    """
    Eval task for majority digit classification.

    Args:
        data_path: Path to JSON file with {"train": [...], "validation": [...]}
        config_path: Path to eval config YAML (reads prompt/system_prompt/scorers)
        split: Which split to evaluate on (default: validation)
        temperature: Generation temperature
        max_tokens: Max tokens to generate
        use_chat_template: If True, use chat format with system message.
        vis_label: Label for visualization (used in task name suffix)
    """
    prompt_str = "{input}"
    system_prompt = ""
    config = {}

    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        prompt_str = config.get("prompt", "{input}")
        system_prompt = config.get("system_prompt", "")

    full_task_name = f"majority_{vis_label}" if vis_label else "majority"

    def record_to_sample(record):
        formatted_input = prompt_str.format(input=record["input"])
        metadata = record.get("metadata", {})
        return Sample(
            input=formatted_input,
            target=record["output"],
            metadata=metadata,
        )

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
        name=full_task_name,
        dataset=dataset,
        solver=solver,
        scorer=build_scorers(config),
        config=GenerateConfig(logprobs=True, top_logprobs=20),
    )

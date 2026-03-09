"""
Unified eval task for all ACS binary prediction tasks.

Supports: ACSIncome, ACSEmployment, ACSMobility, ACSPublicCoverage, ACSTravelTime

Reads prompt and system_prompt from setup_finetune.yaml to ensure train/eval parity.

Usage:
    inspect eval inspect_task_acs.py@acs_binary --model hf/local \
        -M model_path=/path/to/checkpoint \
        -T config_path=/path/to/setup_finetune.yaml \
        -T data_path=/path/to/acs_income_condensed_50000_80P.json

Or use task-specific aliases:
    inspect eval inspect_task_acs.py@acs_income ...
    inspect eval inspect_task_acs.py@acs_employment ...
"""

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.solver import chain, generate, system_message
from inspect_ai.model import GenerateConfig
from cruijff_kit.tools.inspect.scorers import build_scorers


def _create_acs_task(
    task_name: str,
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    vis_label: str = "",
    use_chat_template=True,
) -> Task:
    """
    Create an ACS binary prediction eval task.

    Args:
        task_name: Base name of the task (e.g., "acs_income")
        data_path: Path to JSON file with {"train": [...], "validation": [...], "test": [...]}
        config_path: Path to setup_finetune.yaml (reads prompt/system_prompt from it)
        split: Which split to evaluate on (default: test)
        temperature: Generation temperature
        max_tokens: Max tokens to generate
        vis_label: Optional label for visualization (appended to task name)
        use_chat_template: Whether apply_chat_template should be used for tokenization (i.e., Instruction-tuned models)
    """
    # Construct task name with optional vis_label suffix
    full_task_name = f"{task_name}_{vis_label}" if vis_label else task_name
    # Read prompt and scorer config from YAML
    prompt_str = "{input}"
    system_prompt = ""
    config = {}

    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        prompt_str = config.get("prompt", "{input}")
        system_prompt = config.get("system_prompt", "")

    def record_to_sample(record):
        # Wrap input with prompt template - same as chat_completion training
        formatted_input = prompt_str.format(input=record["input"])
        return Sample(input=formatted_input, target=record["output"])

    dataset = hf_dataset(
        path="json",
        data_files=data_path,
        field=split,
        split="train",  # HuggingFace quirk - always "train" here
        sample_fields=record_to_sample,
    )

    if use_chat_template:
        # Instruct models: use chat format with system message
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
        # generate log probabilities of top 20 tokens from inspect (sets output_logits=True on model generate() call)
        config=GenerateConfig(logprobs=True, top_logprobs=20),
    )


# Generic task - works for any ACS binary prediction
@task
def acs_binary(
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    vis_label: str = "",
    use_chat_template=True,
) -> Task:
    """Generic ACS binary prediction task. Works with any ACS dataset."""
    return _create_acs_task(
        "acs_binary",
        data_path,
        config_path,
        split,
        temperature,
        max_tokens,
        vis_label,
        use_chat_template,
    )


# Task-specific aliases for clarity in eval logs
@task
def acs_income(
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    vis_label: str = "",
    use_chat_template=True,
) -> Task:
    """ACS income prediction (>$50k)."""
    return _create_acs_task(
        "acs_income",
        data_path,
        config_path,
        split,
        temperature,
        max_tokens,
        vis_label,
        use_chat_template,
    )


@task
def acs_employment(
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    vis_label: str = "",
    use_chat_template=True,
) -> Task:
    """ACS employment prediction (employed as civilian)."""
    return _create_acs_task(
        "acs_employment",
        data_path,
        config_path,
        split,
        temperature,
        max_tokens,
        vis_label,
        use_chat_template,
    )


@task
def acs_mobility(
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    vis_label: str = "",
    use_chat_template=True,
) -> Task:
    """ACS mobility prediction (moved in last year)."""
    return _create_acs_task(
        "acs_mobility",
        data_path,
        config_path,
        split,
        temperature,
        max_tokens,
        vis_label,
        use_chat_template,
    )


@task
def acs_publiccoverage(
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    vis_label: str = "",
    use_chat_template=True,
) -> Task:
    """ACS public health coverage prediction."""
    return _create_acs_task(
        "acs_publiccoverage",
        data_path,
        config_path,
        split,
        temperature,
        max_tokens,
        vis_label,
        use_chat_template,
    )


@task
def acs_traveltime(
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    vis_label: str = "",
    use_chat_template=True,
) -> Task:
    """ACS travel time prediction (>20 min commute)."""
    return _create_acs_task(
        "acs_traveltime",
        data_path,
        config_path,
        split,
        temperature,
        max_tokens,
        vis_label,
        use_chat_template,
    )

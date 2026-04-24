"""
Unified eval task for the model-organisms framework.

Drives any dataset produced by ``sanity_checks/model_organisms/generate.py``
(inputs × rules × formats × designs). Supersedes the three bespoke tasks in
``sanity_checks/{bernoulli,count_digits,majority}/inspect_task_*.py`` — their
removal is tracked in #434.

Usage::

    inspect eval inspect_task.py@model_organism \\
        --model hf/local \\
        -M model_path=/path/to/checkpoint \\
        -T data_path=/path/to/dataset.json \\
        -T config_path=/path/to/eval_config.yaml

Calibration opt-in (enables logprobs + appends ``risk_scorer`` if not
already configured via YAML)::

    ... -T calibration=true
"""

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import chain, generate, system_message

from cruijff_kit.tools.inspect.scorers import build_scorers, risk_scorer


@task
def model_organism(
    data_path: str,
    config_path: str = "",
    split: str = "validation",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    use_chat_template: bool = True,
    calibration: bool = False,
    vis_label: str = "",
) -> Task:
    """
    Unified eval task for model-organism datasets.

    Args:
        data_path: Path to JSON file with {"train": [...], "validation": [...]}.
        config_path: Path to eval config YAML (reads prompt/system_prompt/scorers).
        split: Which split to evaluate on (default: validation).
        temperature: Generation temperature.
        max_tokens: Max tokens to generate.
        use_chat_template: If True, prepend a system_message solver.
        calibration: If True, enable logprobs and append risk_scorer to the
            scorer list (unless already listed via YAML).
        vis_label: Optional suffix for the task name (useful for multi-variant
            eval runs across a single experiment).
    """
    prompt_str = "{input}"
    system_prompt = ""
    config: dict = {}

    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        prompt_str = config.get("prompt", "{input}")
        system_prompt = config.get("system_prompt", "")

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

    scorers = build_scorers(config)
    if calibration:
        configured_names = {entry["name"] for entry in config.get("scorer", [])}
        if "risk_scorer" not in configured_names:
            scorers.append(risk_scorer())
        generate_config = GenerateConfig(logprobs=True, top_logprobs=20)
    else:
        generate_config = GenerateConfig()

    task_name = f"model_organism_{vis_label}" if vis_label else "model_organism"

    return Task(
        name=task_name,
        dataset=dataset,
        solver=solver,
        scorer=scorers,
        config=generate_config,
    )

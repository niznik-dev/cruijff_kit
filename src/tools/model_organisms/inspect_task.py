"""
Unified eval task for the model-organisms framework.

Drives any dataset produced by ``tools/model_organisms/generate.py``
(inputs × rules × formats × designs).

Usage::

    inspect eval inspect_task.py@model_organism \\
        --model hf/local \\
        -M model_path=/path/to/checkpoint \\
        -T data_path=/path/to/dataset.json \\
        -T config_path=/path/to/eval_config.yaml

Logprob capture is enabled in two ways:

1. Implicitly — list a scorer in the YAML ``scorer:`` block whose factory sets
   ``requires_logprobs = True`` (e.g., ``risk_scorer``). The task auto-enables
   logprobs because the scorer needs them.
2. Explicitly — pass ``-T logprobs=true`` to capture logprobs for downstream
   analysis without configuring a logprob-consuming scorer.

If a scorer requires logprobs but the user passes ``-T logprobs=false``, the
task fails with a clear configuration error rather than silently overriding.
"""

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import chain, generate, system_message

from cruijff_kit.tools.inspect.scorers import (
    SCORER_FACTORIES,
    build_scorers,
    configured_scorers_require_logprobs,
)


@task
def model_organism(
    data_path: str,
    config_path: str = "",
    split: str = "validation",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    use_chat_template: bool = True,
    logprobs: bool | None = None,
    top_logprobs: int = 20,
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
        logprobs: Capture top logprobs for the generated tokens. ``None`` (the
            default) means "auto" — enabled when a configured scorer declares
            it needs logprobs, otherwise off. Pass ``True``/``False`` to force.
            Passing ``False`` while configuring a scorer that requires
            logprobs raises a ValueError.
        top_logprobs: Number of top logprobs to capture per generated token
            when logprobs are enabled. Default 20; multiclass / tokenization
            variants may need more to keep all option tokens in the top-k.
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
    scorer_needs_logprobs = configured_scorers_require_logprobs(config)

    if logprobs is False and scorer_needs_logprobs:
        required = [
            entry["name"]
            for entry in config.get("scorer", []) or []
            if getattr(
                SCORER_FACTORIES.get(entry.get("name")), "requires_logprobs", False
            )
        ]
        raise ValueError(
            f"Scorer(s) {required} require logprobs, but logprobs=False was "
            "explicitly set. Either remove the logprobs override or drop the "
            "logprob-dependent scorer from the YAML config."
        )

    enable_logprobs = bool(logprobs) or (logprobs is None and scorer_needs_logprobs)

    if enable_logprobs:
        generate_config = GenerateConfig(logprobs=True, top_logprobs=top_logprobs)
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

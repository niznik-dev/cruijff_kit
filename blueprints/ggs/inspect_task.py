"""
Eval task for GGS "books of life" binary prediction.

The synthetic Denmark GGS-II Harmonized Histories data renders each respondent
as a key:value "book of life". Binary targets are derived from a held-out
outcome column (dropped from the input to avoid leakage). The first target,
ever_kid, asks whether the respondent has ever had a child.

Reads prompt and system_prompt from the run config (eval.yaml) to ensure
train/eval parity.

Usage:
    inspect eval inspect_task.py@ever_kid --model hf/local \
        -M model_path=/path/to/checkpoint/epoch_0 \
        -T config_path=/path/to/eval.yaml \
        -T data_path=/path/to/ggs_hh_dk_basic_books_of_life.json

The generic ggs_binary alias works for any future GGS books-of-life binary
target with the same {input, output} + train/validation/test JSON shape.
"""

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.solver import chain, generate, system_message
from inspect_ai.model import GenerateConfig, ChatMessageUser, ChatMessageAssistant
from cruijff_kit.tools.inspect.scorers import build_scorers


def _create_ggs_task(
    task_name: str,
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    vis_label: str = "",
    use_chat_template=True,
    assistant_prefix: str = "",
    top_logprobs: int = 20,
) -> Task:
    """
    Create a GGS books-of-life binary prediction eval task.

    Args:
        task_name: Base name of the task (e.g., "ever_kid")
        data_path: Path to JSON file with {"train": [...], "validation": [...], "test": [...]}
        config_path: Path to eval.yaml (reads prompt/system_prompt from it)
        split: Which split to evaluate on (default: test)
        temperature: Generation temperature
        max_tokens: Max tokens to generate
        vis_label: Optional label for visualization (appended to task name)
        use_chat_template: Whether apply_chat_template should be used for tokenization (i.e., Instruction-tuned models)
        assistant_prefix: If set, prefill an assistant turn with this string. Useful for
            coaxing base (non-instruct) models into the expected output format.
        top_logprobs: Number of top tokens to return logprobs for (passed to GenerateConfig).
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
        if assistant_prefix:
            # Prefill an assistant turn
            return Sample(
                input=[
                    ChatMessageUser(content=formatted_input),
                    ChatMessageAssistant(content=assistant_prefix),
                ],
                target=record["output"],
            )
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
        # generate log probabilities of top_logprobs tokens (sets output_logits=True on model generate() call)
        config=GenerateConfig(logprobs=True, top_logprobs=top_logprobs),
    )


# Generic task - works for any GGS books-of-life binary prediction
@task
def ggs_binary(
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    vis_label: str = "",
    use_chat_template=True,
    assistant_prefix: str = "",
    top_logprobs: int = 20,
) -> Task:
    """Generic GGS books-of-life binary prediction task. Works with any GGS binary target."""
    return _create_ggs_task(
        "ggs_binary",
        data_path,
        config_path,
        split,
        temperature,
        max_tokens,
        vis_label,
        use_chat_template,
        assistant_prefix,
        top_logprobs,
    )


@task
def ever_kid(
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 5,
    vis_label: str = "",
    use_chat_template=True,
    assistant_prefix: str = "",
    top_logprobs: int = 20,
) -> Task:
    """Ever-had-a-child prediction (derived from KID_1; KID_* dropped from input)."""
    return _create_ggs_task(
        "ever_kid",
        data_path,
        config_path,
        split,
        temperature,
        max_tokens,
        vis_label,
        use_chat_template,
        assistant_prefix,
        top_logprobs,
    )

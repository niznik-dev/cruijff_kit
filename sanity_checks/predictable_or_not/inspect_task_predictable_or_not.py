"""
Eval task for predictable_or_not with chat_completion dataset.

COMPATIBILITY NOTE:
    This eval is for models trained with dataset_type=chat_completion (the default).
    For models trained with conditional_completion or other legacy types, use
    inspect_task_predictable_or_not_legacy.py instead.

Clean version that assumes:
- Training used chat_completion dataset with HuggingFace chat templates
- Eval uses inspect-ai which also applies HuggingFace chat templates
- Format parity: both sides use apply_chat_template()

Usage:
    inspect eval inspect_task_predictable_or_not.py --model hf/local \
        -M model_path=/path/to/checkpoint \
        -T data_path=/path/to/pp.json \
        -T prompt="Sequence: {input}"
"""

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.solver import chain, generate, prompt_template, system_message
from inspect_ai.scorer import match, includes


@task
def predictable_or_not(
    data_path: str,
    prompt: str = "{input}",
    system_prompt: str = "",
    split: str = "validation",
    temperature: float = 1e-7,
    max_tokens: int = 10,
) -> Task:
    """
    Eval task for predictable_or_not trained with chat_completion.

    Args:
        data_path: Path to JSON file with {"train": [...], "validation": [...]}
        prompt: Format string to wrap input (must match training prompt)
        system_prompt: Optional system message (must match training)
        split: Which split to evaluate on
        temperature: Generation temperature
        max_tokens: Max tokens to generate
    """
    # Handle prompt being passed as various types from CLI
    if isinstance(prompt, dict):
        prompt = str(prompt.get('prompt', '{input}'))
    prompt_str = str(prompt)

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
            prompt_template("{prompt}"),
            generate(temperature=temperature, max_tokens=max_tokens),
        ),
        scorer=[
            match(location="exact", ignore_case=False),
            includes(ignore_case=False),
        ],
    )

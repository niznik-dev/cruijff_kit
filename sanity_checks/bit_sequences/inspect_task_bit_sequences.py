"""
Eval task for bit_sequences (parity) with chat_completion dataset.

COMPATIBILITY NOTE:
    This eval is for models trained with dataset_type=chat_completion (the default).
    For models trained with conditional_completion or other legacy types, use
    inspect_task_bit_sequences_legacy.py instead.

Usage:
    inspect eval inspect_task_bit_sequences.py --model hf/local \
        -M model_path=/path/to/checkpoint \
        -T data_path=/path/to/bit_sequences.json \
        -T prompt="{input}"
"""

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.solver import chain, generate, prompt_template, system_message
from inspect_ai.scorer import match, includes


@task
def bit_sequences(
    data_path: str,
    prompt: str = "{input}",
    system_prompt: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 5,
) -> Task:
    """Eval task for bit_sequences trained with chat_completion."""
    if isinstance(prompt, dict):
        prompt = str(prompt.get('prompt', '{input}'))
    prompt_str = str(prompt)

    def record_to_sample(record):
        formatted_input = prompt_str.format(input=record["input"])
        return Sample(
            input=formatted_input,
            target=record["output"]
        )

    dataset = hf_dataset(
        path="json",
        data_files=data_path,
        field=split,
        split="train",
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

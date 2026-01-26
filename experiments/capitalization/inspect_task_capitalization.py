"""
Eval task for capitalization.

Supports both instruct models (chat_completion) and base models (text_completion).
Reads prompt and system_prompt from setup_finetune.yaml to ensure train/eval parity.

Usage:
    # For instruct models (chat_completion, default):
    inspect eval inspect_task_capitalization.py --model hf/local \
        -M model_path=/path/to/checkpoint \
        -T config_path=/path/to/setup_finetune.yaml \
        -T data_path=/path/to/words_5L_80P_1000.json

    # For base models (text_completion):
    inspect eval inspect_task_capitalization.py --model hf/local \
        -M model_path=/path/to/checkpoint \
        -T config_path=/path/to/setup_finetune.yaml \
        -T data_path=/path/to/words_5L_80P_1000.json \
        -T use_chat_template=false
"""

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.solver import chain, generate, system_message
from inspect_ai.scorer import match, includes


@task
def capitalization(
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 20,
    use_chat_template: bool = True,
) -> Task:
    """
    Eval task for capitalization.

    Args:
        data_path: Path to JSON file with {"train": [...], "validation": [...], "test": [...]}
        config_path: Path to setup_finetune.yaml (reads prompt/system_prompt from it)
        split: Which split to evaluate on (default: test)
        temperature: Generation temperature
        max_tokens: Max tokens to generate
        use_chat_template: If True, use chat format with system message (instruct models).
                          If False, use simple text (base models). Should match training.
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
        # Wrap input with prompt template - same as training
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

    # Build solver chain based on chat template usage
    if use_chat_template:
        # Instruct models: use chat format with system message
        solver = chain(
            system_message(system_prompt),
            generate(temperature=temperature, max_tokens=max_tokens),
        )
    else:
        # Base models: simple text, no system message
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

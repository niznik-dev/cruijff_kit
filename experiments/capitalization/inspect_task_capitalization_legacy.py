"""
LEGACY eval task for capitalization.

NOTE: This is for models trained with conditional_completion or other legacy
dataset types. For models trained with chat_completion (the new default),
use inspect_task_capitalization.py instead.

Usage:
    inspect eval inspect_task_capitalization_legacy.py --model hf/local \
        -M model_path=/path/to/checkpoint \
        -T config_path=/path/to/setup_finetune.yaml
"""
from __future__ import annotations
from typing import Sequence

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, hf_dataset, FieldSpec, Sample
from inspect_ai.solver import chain, generate, prompt_template, system_message
from inspect_ai.scorer import match, includes
import yaml

@task
def cap_task(config_path: str) -> Task:
    with open(config_path, 'r') as setup_finetune_file:
        setup_finetune = yaml.safe_load(setup_finetune_file)

    try:
        USE_CHAT_TEMPLATE = setup_finetune.get('dataset_type', '') == 'chat_dataset'
        USE_JSON_FORMAT = setup_finetune['dataset_ext'] == '.json'
        DATA_PATH = setup_finetune['input_dir_base'] + setup_finetune['dataset_label'] + ('.json' if (setup_finetune['dataset_ext'] == '.json' and not USE_CHAT_TEMPLATE) else '/test.json')
        SYSTEM_PROMPT = setup_finetune.get('system_prompt', '')
        # For conditional_completion datasets: wrap input with training prompt to match format
        # Falls back to '{input}' (no wrapping) if not specified
        EVAL_PROMPT = setup_finetune.get('prompt', '{input}')
    except KeyError as e:
        raise KeyError(f"Missing required key in setup_finetune.yaml: {e}")

    if isinstance(SYSTEM_PROMPT, Sequence) and not isinstance(SYSTEM_PROMPT, str):
        # CLI parsing may coerce comma-containing strings into iterables; rejoin them.
        SYSTEM_PROMPT = ",".join(SYSTEM_PROMPT)

    # Determine dataset format and load accordingly
    if USE_JSON_FORMAT:
        # JSON format with nested structure (field parameter to access test split)
        if USE_CHAT_TEMPLATE:
            def record_to_sample(record):
                return Sample(
                    input=record["messages"][0]["content"],
                    target=record["messages"][1]["content"]
                )

            dataset = json_dataset(
                DATA_PATH,
                record_to_sample
            )
        else:
            def record_to_sample(record, prompt_template=EVAL_PROMPT):
                # Wrap input with training prompt to match fine-tuning format
                if '{input}' in prompt_template:
                    formatted_input = prompt_template.replace('{input}', record["input"])
                else:
                    # Append input if no placeholder (matches how finetune.yaml adds {input}\n)
                    formatted_input = prompt_template + record["input"]
                return Sample(
                    input=formatted_input,
                    target=record["output"]
                )
            
            dataset = hf_dataset(
                path="json",
                data_files=DATA_PATH,
                field="test",
                split="train", # 'train' here refers to the top-level split in JSON - don't get confused!
                sample_fields=record_to_sample
            )
    else:
        # Parquet format
        dataset = hf_dataset(
            path="parquet",
            data_dir=DATA_PATH,
            split="test",
            sample_fields=FieldSpec(
                input="input",
                target="output",
            ),
        )

    return Task(
        dataset=dataset,
        solver=chain(
            system_message(SYSTEM_PROMPT),
            prompt_template("{prompt}"),
            generate(
                temperature=1e-7,
                max_tokens=10
            ),
        ),
        scorer=[
            match("exact", ignore_case=False),
            includes(ignore_case=False)
        ],
    )

from __future__ import annotations
from typing import Sequence

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, hf_dataset, FieldSpec
from inspect_ai.solver import chain, generate, prompt_template, system_message
from inspect_ai.scorer import match, includes
import yaml

@task
def cap_task() -> Task:
    with open('../../total_config.yaml', 'r') as total_config_file:
        total_config = yaml.safe_load(total_config_file)

    try:
        DATA_PATH = total_config['input_dir_base'] + total_config['dataset_filename']
        SYSTEM_PROMPT = total_config['system_prompt']
        USE_JSON_FORMAT = DATA_PATH.endswith('.json')
    except KeyError as e:
        raise KeyError(f"Missing required key in total_config.yaml: {e}")

    SPLIT_VALUE = "test"
    INPUT_FIELD = "input"
    TARGET_FIELD = "output"

    if isinstance(SYSTEM_PROMPT, Sequence) and not isinstance(SYSTEM_PROMPT, str):
        # CLI parsing may coerce comma-containing strings into iterables; rejoin them.
        SYSTEM_PROMPT = ",".join(SYSTEM_PROMPT)

    # Determine dataset format and load accordingly
    if USE_JSON_FORMAT:
        # JSON format with nested structure (field parameter to access test split)
        dataset = hf_dataset(
            path="json",
            data_files=DATA_PATH,
            field=SPLIT_VALUE,
            split="train", # 'train' here refers to the top-level split in JSON - don't get confused!
            sample_fields=FieldSpec(
                input=INPUT_FIELD,
                target=TARGET_FIELD,
            ),
        )
    else:
        # Parquet format
        dataset = hf_dataset(
            path="parquet",
            data_dir=DATA_PATH,
            split=SPLIT_VALUE,
            sample_fields=FieldSpec(
                input=INPUT_FIELD,
                target=TARGET_FIELD,
            ),
        )

    return Task(
        dataset=dataset,
        solver=chain(
            system_message(SYSTEM_PROMPT),
            prompt_template("{prompt}"),
            generate({
                "temperature": 0.0,
            }),
        ),
        scorer=[
            match("exact", ignore_case=False),
            includes(ignore_case=False)
        ],
    )

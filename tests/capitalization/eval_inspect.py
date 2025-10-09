from __future__ import annotations
import json, os, tempfile
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec
from inspect_ai.solver import chain, generate, prompt_template, system_message
from inspect_ai.scorer import match, includes
from typing import Sequence
import yaml


def filter_to_temp_json(
    data_path: str | os.PathLike,
    *,
    split_key: str = "split",
    split_value: str = "test",
) -> str:
    """Load JSON array, keep rows where row[split_key]==split_value, write to a temp JSON, return its path."""
    src = Path(data_path)
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found: {src}")

    with src.open("r") as f:
        data = json.load(f)

    filtered = [row for row in data if isinstance(row, dict) and row.get(split_key) == split_value]

    temp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix=f"filtered_{split_value}_")
    with os.fdopen(temp_fd, "w") as out:
        json.dump(filtered, out, ensure_ascii=False)

    return tmp_path

def cleanup_temp_file(path: str):
    """Delete the temporary file at the given path."""
    try:
        os.remove(path)
    except OSError as e:
        print(f"Error deleting temp file {path}: {e}")

@task
def cap_task() -> Task:
    with open('../../total_config.yaml', 'r') as total_config_file:
        total_config = yaml.safe_load(total_config_file)

    try:
        DATA_PATH = total_config['input_dir_base'] + total_config['dataset_filename']
        SYSTEM_PROMPT = total_config['system_prompt']
    except KeyError as e:
        raise KeyError(f"Missing required key in total_config.yaml: {e}")

    SPLIT_KEY = "split"
    SPLIT_VALUE = "test"
    INPUT_FIELD = "input"
    TARGET_FIELD = "output"

    # Build a filtered, on-disk JSON array view from your JSON array
    filtered_json = filter_to_temp_json(DATA_PATH, split_key=SPLIT_KEY, split_value=SPLIT_VALUE)

    if isinstance(SYSTEM_PROMPT, Sequence) and not isinstance(SYSTEM_PROMPT, str):
        # CLI parsing may coerce comma-containing strings into iterables; rejoin them.
        SYSTEM_PROMPT = ",".join(SYSTEM_PROMPT)
    return Task(
        dataset=json_dataset(
            filtered_json,
            sample_fields=FieldSpec(
                input=INPUT_FIELD,
                target=TARGET_FIELD,
            ),
        ),
        solver=chain(
            system_message(SYSTEM_PROMPT),
            prompt_template("{prompt}"),
            generate(),
        ),
        scorer=[
            match("exact", ignore_case=False),
            includes(ignore_case=False)
        ],
        cleanup=cleanup_temp_file(filtered_json),
    )

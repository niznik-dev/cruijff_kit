from __future__ import annotations
import json, hashlib
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec
from inspect_ai.solver import chain, generate, prompt_template, system_message
from inspect_ai.scorer import match, includes
from typing import Sequence


def _filter_json_array_to_temp_json(src_path: str, split_value: str) -> str:
    """
    Read a JSON ARRAY file like:
      [ {"input": "...", "output": "...", "split": "train"}, ... ]
    Keep only rows where row['split'] == split_value.
    Write a cached .json next to the source and return that path.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found: {src}")

    cache_dir = src.parent / ".inspect_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    h = hashlib.md5(f"{src.resolve()}::{split_value}".encode()).hexdigest()[:10]
    dst = cache_dir / f"{src.stem}.split-{split_value}.{h}.json"

    if dst.exists():
        return str(dst)

    with src.open("r", encoding="utf-8") as f:
        rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"{src} must be a JSON array (list of objects)")

    filtered = [r for r in rows if isinstance(r, dict) and r.get("split") == split_value]
    if not filtered:
        raise ValueError(f"No rows matched split=='{split_value}' in {src}")

    with dst.open("w", encoding="utf-8") as out:
        json.dump(filtered, out, ensure_ascii=False)

    return str(dst)


@task
def cap_task(
    data_path: str,
    system_prompt: str = "",
    split_value: str = "test",
    input_field: str = "input",
    target_field: str = "output",
    ) -> Task:

    # Build a filtered, on-disk JSON array view from your JSON array
    filtered_json = _filter_json_array_to_temp_json(data_path, split_value)
    
    if isinstance(system_prompt, Sequence) and not isinstance(system_prompt, str):
        # CLI parsing may coerce comma-containing strings into iterables; rejoin them.
        system_prompt = ",".join(system_prompt)
    
    return Task(
        dataset=json_dataset(
            filtered_json,
            sample_fields=FieldSpec(
                input=input_field,
                target=target_field,
            ),
        ),
        solver=chain(
            system_message(system_prompt),
            prompt_template("{prompt}"),
            generate(),
        ),
        scorer=[
            match("exact", ignore_case=False),
            includes(ignore_case=False)
        ]
    )

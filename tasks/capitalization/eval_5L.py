"""Evaluation script for 5-letter capitalization task.

This is a standalone evaluation script that evaluates models on the 5-letter word dataset.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.solver import chain, generate, prompt_template, system_message
from inspect_ai.scorer import match, includes

# Hard-coded configuration for 5-letter words
DATA_PATH = "/home/mjs3/cruijff_kit/tasks/capitalization/input/words_5L_80P_10000.json"
SYSTEM_PROMPT = ""  # Empty for capitalization task

@task
def cap_5L() -> Task:
    """Capitalization task for 5-letter words."""

    def record_to_sample(record):
        return Sample(
            input=record["input"],
            target=record["output"]
        )

    dataset = hf_dataset(
        path="json",
        data_files=DATA_PATH,
        field="test",
        split="train",  # 'train' here refers to the top-level split in JSON
        sample_fields=record_to_sample
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

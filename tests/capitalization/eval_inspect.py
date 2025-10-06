from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec
from inspect_ai.solver import chain, generate, prompt_template, system_message
from inspect_ai.scorer import match, includes
from typing import Sequence

@task
def cap_task(
    data_path: str,
    system_prompt: str = ""
    ) -> Task:
    
    if isinstance(system_prompt, Sequence) and not isinstance(system_prompt, str):
        # CLI parsing may coerce comma-containing strings into iterables; rejoin them.
        system_prompt = ",".join(system_prompt)
    
    return Task(
        dataset=json_dataset(
            data_path,
            sample_fields=FieldSpec(
                input="input",
                target="output",
            )
        ),
        solver=chain(
            system_message(system_prompt),
            prompt_template("{prompt}"),
            generate(),
        ),
        scorer=[match("exact", ignore_case=False), includes(ignore_case=False)]
    )

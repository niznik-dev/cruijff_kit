from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, FieldSpec
from inspect_ai.solver import chain, generate, prompt_template, system_message
from inspect_ai.scorer import match, includes

@task
def cap_task():
    return Task(
        dataset=json_dataset(
            "input/val_words_5L_80P_10000_test.json",
            sample_fields=FieldSpec(
                input="input",
                target="output",
            )
        ),
        solver=chain(
            system_message("Complete the pattern: appleApple funnyFunny "),
            prompt_template("{prompt}"),
            generate(),
        ),
        scorer=[match("exact", ignore_case=False), includes(ignore_case=False)]
    )
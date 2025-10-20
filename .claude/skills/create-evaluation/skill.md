# Create Evaluation

You are helping the user create evaluation scripts for the fine-tuned models or base model using the inspect_ai framework.

## Your Task

Create a standalone evaluation script for each experiment that tests the model on the evaluation dataset.

## Key Principle

**Create standalone evaluation scripts** that directly specify all parameters. Do NOT rely on reading from `setup_finetune.yaml` or other config files.

## CRITICAL: System Prompt Requirement

**ALWAYS find and use the correct system prompt from the training configuration.**

For fine-tuned models, the evaluation MUST use the SAME system prompt that was used during fine-tuning. Otherwise, evaluation results will be invalid.

### How to Find the System Prompt

1. **Check the task's `setup_finetune.yaml`:**
   - Look in `/home/mjs3/cruijff_kit/tasks/<task_name>/setup_finetune.yaml`
   - Search for the `system_prompt:` field

2. **Check experiment-specific configs:**
   - Some experiments may have their own `setup_finetune.yaml` or `finetune.yaml`
   - Look for `system_prompt` in these files

3. **If no system prompt is found:**
   - Use an empty string: `SYSTEM_PROMPT = ""`
   - Do NOT guess or make up a system prompt

4. **Document the source:**
   - Add a comment in the eval script indicating where you found the system prompt

## Script Structure

Each evaluation script should:

1. **Import necessary modules:**
   ```python
   from inspect_ai import Task, task
   from inspect_ai.dataset import json_dataset, hf_dataset, FieldSpec, Sample
   from inspect_ai.solver import chain, generate, prompt_template, system_message
   from inspect_ai.scorer import match, includes
   ```

2. **Define parameters explicitly:**
   - Dataset path (test split)
   - System prompt (MUST match training - see above)
   - Dataset format (JSON, parquet, chat format)
   - Model path (the fine-tuned checkpoint)

3. **Create the task function:**
   - Load the test dataset using inspect_ai's dataset loaders
   - Define the evaluation pipeline (solver chain)
   - Specify scorers (exact match, includes, etc.)

4. **Use appropriate scorers:**
   - `match()`: For exact string matching
   - `includes()`: For partial matching
   - Configure case sensitivity as needed

## Dataset Loading Options

### JSON Format
```python
dataset = json_dataset(
    "path/to/test.json",
    record_to_sample  # Function to convert records to Samples
)
```

### HuggingFace Dataset (JSON)
```python
dataset = hf_dataset(
    path="json",
    data_files="path/to/data.json",
    field="test",  # Access test split
    split="train",
    sample_fields=record_to_sample
)
```

### Parquet Format
```python
dataset = hf_dataset(
    path="parquet",
    data_dir="path/to/data",
    split="test",
    sample_fields=FieldSpec(
        input="input",
        target="output"
    )
)
```

## Example Template

Here's a complete example showing proper system prompt handling:

```python
"""
Standalone evaluation script for experiment_name
Task: capitalization
Model: Llama-3.2-1B-Instruct with LoRA rank 4

System prompt source: /home/mjs3/cruijff_kit/tasks/capitalization/setup_finetune.yaml
"""
from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset
from inspect_ai.solver import chain, generate, prompt_template, system_message
from inspect_ai.scorer import match, includes


@task
def cap_eval() -> Task:
    """Evaluate capitalization task"""

    # Dataset configuration
    DATA_PATH = "/home/mjs3/cruijff_kit/tasks/capitalization/input/words_8L_80P_10000.json"

    # System prompt from tasks/capitalization/setup_finetune.yaml
    SYSTEM_PROMPT = "You are a smart and helpful language model. Convert the first character of the word to uppercase and keep the rest of the word in lowercase. Return only the modified word."

    # Load test dataset from JSON (instruct format)
    def record_to_sample(record):
        from inspect_ai.dataset import Sample
        return Sample(
            input=record["input"],
            target=record["output"]
        )

    dataset = hf_dataset(
        path="json",
        data_files=DATA_PATH,
        field="test",
        split="train",  # 'train' refers to the top-level split in the JSON structure
        sample_fields=record_to_sample
    )

    # Define evaluation pipeline
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
```

See `tasks/capitalization/inspect.py` for the original reference implementation, but remember to:
- Hard-code dataset paths and system prompts (don't read from config files)
- **ALWAYS verify the system prompt matches training**
- Make the script fully self-contained

## Workflow: Creating Evaluation Scripts

Follow these steps IN ORDER:

1. **Identify the task** (e.g., capitalization, synthetic_twins)

2. **Find the system prompt:**
   ```bash
   # Look in the task's setup file
   grep -i "system_prompt" /home/mjs3/cruijff_kit/tasks/<task_name>/setup_finetune.yaml
   ```

3. **Find dataset configuration:**
   - Dataset path and format (JSON, Parquet, chat)
   - Field names (input/output)

4. **Create eval.py for each experiment:**
   - Document system prompt source in docstring
   - Hard-code all parameters
   - Use appropriate dataset loader for the format
   - Set temperature=0.0 for deterministic evaluation

5. **Verify:**
   - System prompt matches training exactly
   - Dataset path points to test split
   - No dependencies on external config files

## Running Evaluations

After creating evaluation scripts, they can be run with:

```bash
inspect eval path/to/eval_script.py --model <model-path>
```

Or submitted as SLURM jobs using `setup_inspect.py`:

```bash
python tools/inspect/setup_inspect.py --finetune_epoch_dir /path/to/experiment/epoch_N
sbatch inspect.slurm
```

## Output

Create one evaluation script (e.g., `eval.py`) in each experiment directory that:
- **Uses the SAME system prompt as training** (CRITICAL)
- Loads the correct test dataset
- Uses the fine-tuned model checkpoint or the base model (as appropriate)
- Applies appropriate scorers
- Is completely standalone and doesn't require external config files

## Next Steps

After creating evaluation scripts, you can either:
- Run evaluations locally for quick results
- Create SLURM scripts to run evaluations on the cluster
- Compare results across all experiments

# Create Evaluation

You are helping the user create evaluation scripts for their fine-tuned models using the inspect_ai framework.

## Your Task

Create a standalone evaluation script for each experiment that tests the fine-tuned model on the evaluation dataset.

## Key Principle

**Create standalone evaluation scripts** that directly specify all parameters. Do NOT rely on reading from `setup_finetune.yaml` or other config files.

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
   - System prompt
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

See `tasks/capitalization/inspect.py` for a reference implementation, but remember to:
- Remove dependency on `setup_finetune.yaml`
- Hard-code dataset paths and system prompts
- Make the script fully self-contained

## Running Evaluations

After creating evaluation scripts, they can be run with:

```bash
inspect eval path/to/eval_script.py --model <model-path>
```

Or submitted as SLURM jobs if needed.

## Output

Create one evaluation script (e.g., `eval.py`) in each experiment directory that:
- Loads the correct test dataset
- Uses the fine-tuned model checkpoint
- Applies appropriate scorers
- Is completely standalone and doesn't require external config files

## Next Steps

After creating evaluation scripts, you can either:
- Run evaluations locally for quick results
- Create SLURM scripts to run evaluations on the cluster
- Compare results across all experiments

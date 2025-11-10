# General Evaluation Task

**Created:** 2025-10-24
**Task File:** `experiments/inspect_task_general.py`
**Inspect-AI Version:** Latest (designed for current API)

## Evaluation Objective

This is a general-purpose evaluation task designed to assess model performance on input/output pair datasets. It evaluates whether the model can produce the exact expected output given an input prompt.

**Primary use cases:**
- Capitalization task (input: "word" → output: "Word")
- Any transformation task with clear input/output pairs
- Classification tasks where output is a specific label
- Other experiments in the `experiments/` directory with similar structure

**Success criteria:**
- Exact match with target output (case-sensitive)
- Substring match (target appears in output)

## Dataset Configuration

**Format:** JSON with nested train/test splits

**Structure:**
```json
{
  "train": [
    {"input": "...", "output": "..."},
    ...
  ],
  "test": [
    {"input": "...", "output": "..."},
    ...
  ]
}
```

**Field Mapping:**
- Input field: `input`
- Target field: `output`
- Metadata fields: none (can be extended)

**Loading Method:**
Uses `hf_dataset` with JSON format and custom `record_to_sample` function to handle the nested structure.

**Data Structure:**
The JSON file contains top-level keys for data splits ("train", "test", etc.). Each split contains an array of records with "input" and "output" fields.

## Solver Chain

**Components:**
1. **system_message()**: Provides optional system-level instructions to the model
2. **prompt_template()**: Passes the input directly using `{prompt}` template
3. **generate()**: Generates model response with specified parameters

**System Message:**
```
{configurable - empty by default}
```

**Prompt Template:**
```
{prompt}
```
(Direct input with no additional formatting)

**Generation Parameters:**
- Temperature: 0.0 (default) - Deterministic output for consistency
- Max tokens: Model default (configurable via parameter)

**Rationale:**
- Temperature 0.0 ensures consistent, reproducible results across evaluations
- Direct prompt template keeps the task simple and focuses on core capability
- System message is optional to allow task-specific instructions when needed

## Scorer Configuration

**Primary Scorers:** Multiple scorers for comprehensive evaluation

1. **match(location="exact", ignore_case=False)**
   - Exact matching after normalization
   - Case-sensitive (important for capitalization task)
   - Ignores whitespace and punctuation by default

2. **includes(ignore_case=False)**
   - Checks if target appears anywhere in output
   - Case-sensitive
   - Useful for partial credit or verbose responses

**Scorer Options:**
- `location="exact"` - Target must match entire output
- `ignore_case=False` - Preserves case sensitivity (critical for capitalization)

**Rationale:**
Using both scorers provides:
- Strict evaluation (exact match)
- Lenient evaluation (substring match)
- Insight into how often models produce correct answer with extra text

## Task Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| config_dir | str (optional) | None | Path to epoch directory with ../setup_finetune.yaml |
| dataset_path | str (optional) | None | Direct path to dataset JSON file |
| system_prompt | str | "" | System message for the model |
| temperature | float | 0.0 | Generation temperature |
| split | str | "test" | Which data split to use |
| max_tokens | int (optional) | None | Maximum tokens to generate |

**Parameter Precedence:**
- If `config_dir` is provided, reads dataset path and system prompt from config
- If `dataset_path` is provided, uses it directly
- `system_prompt` parameter always overrides config value
- One of `config_dir` or `dataset_path` must be provided

**Parameter Usage:**
```bash
# With config_dir
inspect eval experiments/inspect_task_general.py -T config_dir=/path/to/epoch_0

# With dataset_path
inspect eval experiments/inspect_task_general.py -T dataset_path=/path/to/data.json

# With custom parameters
inspect eval experiments/inspect_task_general.py -T dataset_path=/path/to/data.json -T temperature=0.5
```

## Model Specification

**Two primary usage modes:**

### Mode 1: Evaluating Fine-Tuned Models

For models from fine-tuning experiments:

```bash
inspect eval experiments/inspect_task_general.py \
  --model hf/local \
  -M model_path=/scratch/gpfs/MSALGANIK/niznik/ck-outputs/ck-out-rank8/epoch_0 \
  -T config_dir=/scratch/gpfs/MSALGANIK/niznik/ck-outputs/ck-out-rank8/epoch_0
```

This mode:
- Reads dataset configuration from `setup_finetune.yaml`
- Uses same system prompt as training
- Maintains consistency between training and evaluation

### Mode 2: Standalone Evaluation

For evaluating any model on a specific dataset:

```bash
inspect eval experiments/inspect_task_general.py \
  --model hf/local \
  -M model_path=/path/to/model \
  -T dataset_path=/path/to/experiments/capitalization/input/words_4L_80P_5000.json
```

This mode:
- Directly specifies dataset
- More flexible for ad-hoc evaluations
- Useful for testing base models or external models

## Example Usage

### Evaluating Capitalization Task

**Fine-tuned model from experiment:**
```bash
cd /scratch/gpfs/MSALGANIK/niznik/experiment_name/run_name/epoch_0

inspect eval /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/experiments/inspect_task_general.py \
  --model hf/local \
  -M model_path=$PWD \
  -T config_dir=$PWD
```

**Base model on capitalization data:**
```bash
inspect eval experiments/inspect_task_general.py \
  --model hf/local \
  -M model_path=/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-1B-Instruct \
  -T dataset_path=experiments/capitalization/input/words_4L_80P_5000.json
```

### Testing with Limited Samples

```bash
inspect eval experiments/inspect_task_general.py \
  --model hf/local \
  -M model_path=/path/to/model \
  -T dataset_path=experiments/capitalization/input/words_4L_80P_5000.json \
  --limit 10
```

### With Custom System Prompt

```bash
inspect eval experiments/inspect_task_general.py \
  --model hf/local \
  -M model_path=/path/to/model \
  -T dataset_path=experiments/capitalization/input/words_4L_80P_5000.json \
  -T system_prompt="You are a helpful assistant that capitalizes words."
```

## Integration with Fine-Tuning Workflow

This task integrates seamlessly with the cruijff_kit fine-tuning workflow:

**1. Train a model:**
```bash
cd /path/to/experiment/run_dir
sbatch finetune.slurm
```

**2. Evaluate the fine-tuned model:**
```bash
cd /path/to/experiment/run_dir/epoch_0

inspect eval /path/to/cruijff_kit/experiments/inspect_task_general.py \
  --model hf/local \
  -M model_path=$PWD \
  -T config_dir=$PWD
```

The task automatically:
- Reads dataset configuration from `../setup_finetune.yaml`
- Uses the same system prompt as training
- Evaluates on the correct data split

## Output Files

Inspect-ai creates:
- `logs/general_eval_{timestamp}.eval` - Detailed evaluation log with all samples and scores
- Console output with accuracy metrics

**Metrics reported:**
- Accuracy (from match scorer) - Percentage of exact matches
- Accuracy (from includes scorer) - Percentage with target substring
- Standard error of the mean

**Viewing results:**
```bash
# Interactive web UI
inspect view

# Or view specific log
inspect view logs/general_eval_{timestamp}.eval
```

## Expected Performance

**Capitalization task:**
- Base models (untrained): Near 0% accuracy
- Fine-tuned models: Should approach 100% on trained word lengths
- Generalization: Performance may degrade on unseen word lengths

**Other tasks:**
Performance varies by task complexity and model capability.

## Extending for Other Tasks

This task can be adapted for other experiments in `experiments/` by:

1. **No changes needed if:**
   - Dataset has same JSON structure (train/test splits)
   - Fields are named "input" and "output"
   - Evaluation is exact string matching

2. **Modify `record_to_sample` if:**
   - Different field names (e.g., "question"/"answer")
   - Need to transform inputs/outputs
   - Want to preserve metadata

3. **Change scorers if:**
   - Need case-insensitive matching
   - Need custom scoring logic
   - Want model-graded evaluation

## Notes

**Design decisions:**
- Temperature 0.0 prioritizes consistency over creativity
- Case-sensitive matching is critical for capitalization task
- Two scorers provide both strict and lenient evaluation
- config_dir mode maintains training/eval consistency

**Limitations:**
- Assumes JSON format with nested structure
- Requires "input" and "output" field names (or code modification)
- Scorers are text-based (not suitable for numerical tasks without modification)

**Future improvements:**
- Add support for Parquet format
- Add configurable field names via parameters
- Add support for multiple-choice format
- Add custom scorer options

## References

- Inspect-AI documentation: https://inspect.aisi.org.uk/
- Inspect-AI tasks guide: https://inspect.aisi.org.uk/tasks.html
- Inspect-AI scorers guide: https://inspect.aisi.org.uk/scorers.html
- cruijff_kit capitalization task: `experiments/capitalization/cap_task.py`

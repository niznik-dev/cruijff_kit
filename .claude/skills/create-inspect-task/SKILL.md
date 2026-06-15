---
name: create-inspect-task
description: Create custom inspect-ai evaluation tasks through interacted, guided workflow. 
---

# Create Inspect Task

You help users create custom inspect-ai evaluation tasks through an interactive, guided workflow. Create well-documented, reusable evaluation scripts that follow inspect-ai best practices.

## Your Task

Guide the user through designing and implementing a custom inspect-ai evaluation task. Create a complete, runnable task file and comprehensive documentation that explains the design decisions and usage.

## Operating Modes

This skill supports two modes:

### Mode 1: Experiment-Guided (Recommended)
When an `experiment_summary.yaml` file exists (created by `design-experiment` skill), extract configuration to pre-populate:
- Dataset path and format
- Model information
- Evaluation objectives
- System prompts
- Common parameters

**Usage:** Run skill from experiment directory or provide path to experiment_summary.yaml

### Mode 2: Standalone
Create evaluation tasks from scratch without experiment context. User provides all configuration manually.

**Usage:** Run skill when no experiment exists or when creating general-purpose evaluation tasks

## Workflow

### Initial Setup (Both Modes)

1. **Check for experiment context**
   - Look for `experiment_summary.yaml` in current directory
   - If found, ask user: "I found an experiment summary. Would you like me to use it to configure the evaluation task?"
   - If user says yes, proceed with Mode 1
   - If no or not found, proceed with Mode 2

### Mode 1: Experiment-Guided Workflow

1. **Read experiment_summary.yaml** - Extract configuration
2. **Confirm extracted info** - Show user what was found (dataset, models, etc.)
3. **Understand evaluation objective** - What specific aspect to evaluate?
4. **Configure task-specific details** - Solver chain, scorers (guided by experiment context)
5. **Add task parameters** - Make the task flexible and reusable
6. **Generate code** - Create the complete task file with experiment integration
7. **Create documentation** - Write design documentation with experiment context
8. **Create log** - Document all decisions in `logs/create-inspect-task.log`
9. **Provide usage guidance** - Show user how to run the task with their models

### Mode 2: Standalone Workflow

1. **Understand the objective** - What does the user want to evaluate?
2. **Configure dataset** - Guide dataset format selection and loading
3. **Design solver chain** - Build the solver pipeline (prompts, generation, etc.)
4. **Select scorers** - Choose appropriate scoring mechanisms
5. **Add task parameters** - Make the task flexible and reusable
6. **Generate code** - Create the complete task file
7. **Create documentation** - Write design documentation with rationale
8. **Create log** - Document all decisions in `logs/create-inspect-task.log`
9. **Provide usage guidance** - Show user how to run the task

## Extracting Information from experiment_summary.yaml (Mode 1)

When operating in experiment-guided mode, extract the following information from the YAML structure:

### YAML Structure Overview

```yaml
experiment:
  name: string
  project: string
  question: string

data:
  training:
    path: string
    dataset_label: string
    format: string
    splits:
      train: int
      validation: int
      test: int

models:
  base:
    - name: string
      path: string

evaluation:
  system_prompt: string
  temperature: float

runs:
  - name: string
    type: string  # "fine-tuned" or "control"
    model: string
```

### Extraction Algorithm

```python
import yaml
from pathlib import Path

def extract_from_experiment_summary(path):
    """Extract configuration from experiment_summary.yaml"""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract dataset configuration
    dataset_path = config['data']['training']['path']
    dataset_format = config['data']['training']['format']
    dataset_splits = config['data']['training']['splits']

    # Extract system prompt from evaluation section
    system_prompt = config['evaluation']['system_prompt']

    # Extract research question
    research_question = config['experiment']['question']
    project = config['experiment']['project']

    # Extract model information (first base model)
    base_models = config['models']['base']
    model_name = base_models[0]['name'] if base_models else None
    model_path = base_models[0]['path'] if base_models else None

    # Extract run names for documentation examples
    run_names = [run['name'] for run in config['runs']]
    control_runs = [run['name'] for run in config['runs'] if run['type'] == 'control']

    return {
        'dataset_path': dataset_path,
        'dataset_format': dataset_format,
        'dataset_splits': dataset_splits,
        'system_prompt': system_prompt,
        'research_question': research_question,
        'project': project,
        'model_name': model_name,
        'model_path': model_path,
        'run_names': run_names,
        'control_runs': control_runs
    }
```

### Key Fields to Extract

**From `experiment` section:**
- `question` → Research question/objective (informs evaluation goal)
- `project` → Blueprint directory name under `blueprints/` (pins the task family)

**From `data.training` section:**
- `path` → Dataset path for evaluation
- `format` → Dataset format (json)
- `splits` → Sample counts (use test split for evaluation)

**From `models.base[]` section:**
- `name` → Model identifier
- `path` → Full path to base model (for usage examples)

**From `evaluation` section:**
- `system_prompt` → Use same prompt for consistency
- `temperature` → Default temperature setting

**From `runs[]` section:**
- `name` → Run identifiers (for documentation)
- `type` → Filter for "control" runs that need evaluation

### Presenting Extracted Information

After extraction, show the user what was found:

```markdown
## Configuration Extracted from Experiment

I found the following configuration in your experiment:

**Dataset:**
- Path: `{ck_data_dir}/capitalization/words_4L_80P_300.json`
- Format: JSON
- Splits: train (240), test (60)

**Models:**
- Llama-3.2-1B-Instruct
- Path: `/scratch/gpfs/.../pretrained-llms/Llama-3.2-1B-Instruct`

**System Prompt:**
```
{extracted_prompt or "(none)"}
```

**Research Question:**
{extracted_question}

I'll use this information to help configure your evaluation task. You can override any of these settings if needed.
```

### Validation

Check extracted information:
- ✓ Dataset path exists (verify with `ls`)
- ✓ Dataset format is supported (.json, .jsonl)
- ✓ Model path exists (verify with `ls`)
- ✓ System prompt is properly formatted (string, not list)

If validation fails:
- Warn user but continue
- Ask user to provide correct information
- Log validation failures

## Logging

**IMPORTANT:** Create a detailed log file at `{task_dir}/logs/create-inspect-task.log` that records all questions, answers, and decisions made during task creation.

### Log Format

```
[YYYY-MM-DD HH:MM:SS] ACTION: Description
Details: {specifics}
Result: {outcome}

```

### What to Log

- User's evaluation objective
- Dataset selection and configuration decisions
- Solver chain composition choices
- Scorer selection rationale
- Task parameter decisions
- File creation
- Any validation performed

### Example Log Entries

#### Mode 1: Experiment-Guided

```
[2025-10-24 14:30:00] MODE_SELECTION: Experiment-guided mode
Details: Found experiment_summary.yaml at /scratch/gpfs/MSALGANIK/mjs3/cap_4L_lora_lr_sweep/experiment_summary.yaml
Result: User confirmed to use experiment configuration

[2025-10-24 14:30:05] EXTRACT_CONFIG: Reading experiment_summary.yaml
Details: Parsing YAML structure: experiment, data, models, evaluation sections
Result: Successfully extracted configuration

[2025-10-24 14:30:10] EXTRACTED_DATASET: Dataset configuration
Details: Path: {ck_data_dir}/capitalization/words_4L_80P_300.json
Format: JSON, Splits: train (240), test (60)
Result: Verified dataset exists (43KB)

[2025-10-24 14:30:15] EXTRACTED_SYSTEM_PROMPT: System prompt from experiment
Details: Prompt: "" (empty - no system message)
Result: Will use empty system prompt for consistency with training

[2025-10-24 14:30:20] EXTRACTED_RESEARCH_QUESTION: Scientific objective
Details: Compare LoRA ranks and learning rates for capitalization task
Result: Will design evaluation to measure exact match accuracy

[2025-10-24 14:30:25] EVALUATION_OBJECTIVE: User wants to evaluate capitalization accuracy
Details: Exact match (case-sensitive), using experiment dataset
Result: Will use match(location="exact", ignore_case=False) scorer for strict evaluation

[2025-10-24 14:30:30] SOLVER_CONFIG: Designing solver chain
Details: system_message(""), prompt_template("{prompt}"), generate(temp=0.0)
Result: Matches training configuration for consistency
```

#### Mode 2: Standalone

```
[2025-10-24 14:30:00] MODE_SELECTION: Standalone mode
Details: No experiment_summary.yaml found
Result: User will provide all configuration manually

[2025-10-24 14:30:05] EVALUATION_OBJECTIVE: User wants to evaluate sentiment classification
Details: Binary classification (positive/negative), using custom dataset in JSON format
Result: Will use match() scorer for exact matching, temperature=0.0 for consistency

[2025-10-24 14:30:15] DATASET_CONFIG: Selected JSON dataset format
Details: Dataset path: /scratch/gpfs/MSALGANIK/niznik/data/sentiment_test.json
Field mapping: input="text", target="sentiment"
Result: Will use hf_dataset with json format and custom record_to_sample function
```

## Questions to Ask

### 1. Evaluation Objective

**What do you want to evaluate?**
- Classification task? (sentiment, topic, entity type, etc.)
- Generation quality? (summarization, translation, etc.)
- Factual accuracy? (question answering, fact checking)
- Reasoning ability? (math, logic, chain-of-thought)
- Task-specific capability? (code generation, instruction following)

**What defines a correct answer?**
- Exact match with target?
- Contains specific information?
- Model-graded quality assessment?
- Multiple acceptable answers?

### 2. Dataset Configuration

**What dataset format do you have?**
- JSON file (`.json` or `.jsonl`)
- HuggingFace dataset (specify dataset name)
- CSV file
- Custom format (will need conversion)

**Where is the dataset located?**
- Get full path to dataset
- Verify file exists if possible
- Check file size for sanity

**What are the field names?**
- Input field name (e.g., "question", "text", "prompt")
- Target/answer field name (e.g., "answer", "label", "output")
- Any metadata fields to preserve? (e.g., "category", "difficulty")

**Dataset structure specifics:**
- For JSON: Is it a single JSON file with nested structure or JSONL?
- For JSON with splits: Which field contains the test split?
- For HuggingFace: Dataset name and split to use?

**Example questions:**
- "Does your JSON file have a structure like `{'train': [...], 'test': [...]}`?"
- "Is each line a separate JSON object (JSONL format)?"
- "Do you need to load from a specific split like 'test' or 'validation'?"

### 3. Solver Configuration

**System message:**
- Do you want to provide instructions to the model via system message?
- What role should the model play? (e.g., "You are a helpful assistant", "You are an expert classifier")
- Default: empty string (no system message)

**Prompt template:**
- Should we use the input directly or wrap it in a template?
- Do you need chain-of-thought prompting?
- Default: `"{prompt}"` (direct input)

**Generation parameters:**
- **Temperature**:
  - 0.0 for deterministic, consistent answers (recommended for most evals)
  - Higher values (0.7-1.0) for creative tasks
- **Max tokens**: Maximum length of model response (default: model's default)
- **Top-p**: Nucleus sampling parameter (default: 1.0)

**Common solver patterns:**
- Simple generation: `[system_message(""), prompt_template("{prompt}"), generate()]`
- Chain-of-thought: `[chain_of_thought(), generate()]`
- Multiple-choice: `[multiple_choice()]` (don't add separate generate())
- Custom template: `[prompt_template("Answer: {prompt}\n"), generate()]`

### 4. Scorer Selection

**Based on evaluation objective, suggest scorers:**

**cruijff_kit custom scorers (preferred for experiment tasks — driven by the `scorers:` block in `eval.yaml`):**

Rather than hard-coding a scorer in the task, read the `scorers:` list from `config_path` and build it via the shared registry. This lets the experiment design (not the task code) pick the scorer:

```python
from cruijff_kit.tools.inspect.scorers import (
    build_scorers,
    configured_scorers_require_logprobs,
)

scorers = build_scorers(config)  # falls back to DEFAULT_SCORERS if no scorers: block
```

The `scorers:` block is a list of `{name, params}` entries:

```yaml
scorers:
  - name: match
  - name: risk_scorer
    params:
      option_tokens: ["0", "1"]
```

Registry names: `match`, `includes`, `risk_scorer`, `numeric_risk_scorer`, `continuous_scorer`.

**logprobs contract:** logprob-based scorers (e.g. `risk_scorer`) set `requires_logprobs = True` on their factory. Call `configured_scorers_require_logprobs(config)` and auto-enable logprob capture when it returns `True`:

```python
scorer_needs_logprobs = configured_scorers_require_logprobs(config)
enable_logprobs = bool(logprobs) or (logprobs is None and scorer_needs_logprobs)
if enable_logprobs:
    generate_config = GenerateConfig(logprobs=True, top_logprobs=top_logprobs)
else:
    generate_config = GenerateConfig()
# ... and fail loudly if the user passed logprobs=False while configuring a
# scorer that needs them, rather than silently producing unscored logs.
```

Note: `risk_scorer` accuracy is exact-string-match (`completion == target`), not argmax — it conflates output format with judgment. Keep that in mind when interpreting its `accuracy`.

**For exact matching (inspect-ai built-ins — fine for standalone tasks without the registry):**
- `match()` - Target appears at beginning/end; ignores case, whitespace, punctuation
  - Options: `location="begin"/"end"/"any"`, `ignore_case=True/False`
- `exact()` - Precise matching after normalization
- `includes()` - Target appears anywhere in output
  - Options: `ignore_case=True/False`

**For multiple choice:**
- `choice()` - Works with `multiple_choice()` solver
- Returns letter of selected answer (A, B, C, D, etc.)

**For pattern extraction:**
- `pattern()` - Extract answer using regex
  - Requires regex pattern parameter

**For model-graded evaluation:**
- `model_graded_qa()` - Another model assesses answer quality
  - Options: `partial_credit=True/False`, custom `template`
- `model_graded_fact()` - Checks if specific facts appear
- Note: Requires additional model, adds latency and cost

**For numeric/F1 scoring:**
- `f1()` - F1 score for text overlap

**Multiple scorers:**
- Can use a list: `[match(), includes()]` to get multiple scores
- Helpful for comparing scoring methods

### 5. Task Parameters

**Should the task accept parameters for flexibility?**

**Standard parameters — match the current blueprint tasks so the task drops into the `eval.yaml` pipeline** (cleanest reference: `blueprints/capitalization/inspect_task.py`):

- `data_path` — Path to the dataset JSON (required; no default).
- `config_path` — Path to `eval.yaml`. The task reads `prompt`, `system_prompt`, and (optionally) the `scorers:` block from it. `setup_inspect.py` auto-derives this from the config file's own location — you do not pass it by hand.
- `vis_label` — Optional label appended to the task name (`f"{name}_{vis_label}" if vis_label else name`). **Required for the viz pipeline**: `viz_helpers` reads `vis_label` from the eval log's `task_args` to dedup and label runs. Omit it and multi-variant runs collide in the visualizations.
- `split` — Which split to evaluate (e.g. `"test"`, `"validation"`).
- `use_chat_template` — `True` for instruct models (adds a `system_message` solver), `False` for base models (plain text completion).
- `temperature` / `max_tokens` — Generation params. Greedy default is `temperature=1e-7`.
- `logprobs` / `top_logprobs` — Capture top logprobs. Leave `logprobs` defaulting to `None` (auto) so it enables only when a configured scorer needs them — see Scorer Selection.
- `assistant_prefix` — Optional text to seed the assistant turn.

These names are not arbitrary. `setup_inspect.py` maps a fixed `TASK_ARG_KEYS` set onto `-T` flags — commonly the likes of:

```
data_path, config_path, vis_label, split, temperature, max_tokens, …
```

`TASK_ARG_KEYS` in `setup_inspect.py` is the source of truth for the full set — read it there rather than trusting this list to stay complete. A parameter outside that set will **not** receive a value from `eval.yaml` (you'd get a startup warning from `load_eval_config` about an unconsumed key); add genuinely new task args to `TASK_ARG_KEYS` if the task needs them.

**Benefits of parameters:**
- Run variations without code changes
- Easier experimentation
- Better reusability

**How to pass parameters:**
```bash
inspect eval task.py -T param_name=value
```

### 6. Model Specification

**How will the model be specified?**

**Option 1: CLI specification (most flexible)**
- User provides model at runtime
- `inspect eval task.py --model hf/local -M model_path=/path/to/model`
- Recommended for most cases

**Option 2: Experiment pipeline (eval.yaml)**
- The model path is baked into the SLURM cell by `setup_inspect.py` from `eval.yaml`
- The task itself only reads `prompt`/`system_prompt`/`scorers` from `config_path` — it never resolves the model
- This is how `scaffold-inspect` runs tasks; see Generated Task Pattern below

**Option 3: Hard-coded in task**
- Less flexible but simpler
- Can specify model inside task definition
- Better for benchmarking specific models

## Output Files

Create two files:

### 1. Task Script: `{task_name}_task.py`

The complete, runnable inspect-ai task following best practices.

**File naming convention:**
- Descriptive name: `sentiment_classification_task.py`
- Include domain: `math_reasoning_task.py`
- Follow pattern: `{domain}_{type}_task.py`

**Required components:**
```python
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset, hf_dataset, FieldSpec
from inspect_ai.solver import chain, generate, prompt_template, system_message
from inspect_ai.scorer import match, includes

@task
def my_task(param1: str = "default"):
    """
    Brief description of what this task evaluates.

    Args:
        param1: Description of parameter

    Returns:
        Task: Configured inspect-ai task
    """

    # Dataset loading
    dataset = ...

    # Solver chain
    solver = chain(
        system_message("..."),
        prompt_template("{prompt}"),
        generate(temperature=0.0)
    )

    # Return task
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=...
    )
```

**Best practices to follow:**
- Use type hints for parameters
- Include docstring explaining purpose
- Add comments explaining non-obvious choices
- Handle errors gracefully (try/except for file operations)
- Validate required parameters
- Use descriptive variable names

### 2. Design Documentation: `{task_name}_design.md`

Comprehensive documentation of design decisions.

**Required sections:**

```markdown
# {Task Name} Evaluation Task

**Created:** {timestamp}
**Inspect-AI Version:** {version if known}

## Evaluation Objective

{What this task evaluates and why}

## Dataset Configuration

**Format:** {JSON/HuggingFace/etc.}
**Location:** `{full_path_to_dataset}`
**Size:** {number of samples if known}

**Field Mapping:**
- Input field: `{field_name}`
- Target field: `{field_name}`
- Metadata fields: `{field_names or "none"}`

**Loading Method:**
{Description of how dataset is loaded}

**Data Structure:**
{Explanation of JSON structure, splits, etc.}

## Solver Chain

**Components:**
1. {Solver 1}: {Purpose}
2. {Solver 2}: {Purpose}
3. ...

**System Message:**
```
{system message text or "none"}
```

**Prompt Template:**
```
{template or "direct input"}
```

**Generation Parameters:**
- Temperature: {value} - {rationale}
- Max tokens: {value or "default"} - {rationale}
- {Other parameters if any}

**Rationale:**
{Why this solver chain was chosen}

## Scorer Configuration

**Primary Scorer:** `{scorer_name}()`

**Options:**
- {option1}: {value} - {reason}
- {option2}: {value} - {reason}

**Additional Scorers:**
{List if multiple scorers used, or "none"}

**Rationale:**
{Why this scorer is appropriate for the task}

## Task Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| {param1} | {type} | {default} | {description} |

**Parameter Usage:**
```bash
inspect eval {task_file}.py -T {param}={value}
```

## Model Specification

**Recommended usage:**
```bash
inspect eval {task_file}.py --model hf/local -M model_path=/path/to/model
```

{Any specific notes about model compatibility}

## Example Usage

**Basic evaluation:**
```bash
inspect eval {task_name}_task.py --model hf/local -M model_path=/path/to/model
```

**With parameters:**
```bash
inspect eval {task_name}_task.py --model hf/local -M model_path=/path/to/model -T temperature=0.5
```

**Evaluating fine-tuned model:** {if applicable}
```bash
inspect eval {task_name}_task.py@{task_name} --model hf/local \
  -M model_path=/path/to/run/artifacts/epoch_0 \
  -T data_path=/path/to/dataset.json \
  -T config_path=/path/to/eval.yaml
```

## Output Files

Inspect-ai will create:
- `logs/{task_name}_{timestamp}.eval` - Evaluation results log
- Console output with accuracy and metrics

## Expected Performance

{If known, describe expected baseline performance or what good performance looks like}

## Notes

{Any additional considerations, limitations, or future improvements}

## References

- Inspect-AI documentation: https://inspect.aisi.org.uk/
- {Any other relevant references}
```

## Code Generation Guidelines

### Dataset Loading Patterns

**JSON with nested splits:**
```python
from inspect_ai.dataset import hf_dataset

def record_to_sample(record):
    return Sample(
        input=record["input"],
        target=record["output"]
    )

dataset = hf_dataset(
    path="json",
    data_files="/path/to/data.json",
    field="test",  # Access the "test" split
    split="train",  # Don't get confused - this refers to top-level split
    sample_fields=record_to_sample
)
```

**JSONL (one JSON object per line):**
```python
from inspect_ai.dataset import json_dataset

def record_to_sample(record):
    return Sample(
        input=record["question"],
        target=record["answer"]
    )

dataset = json_dataset(
    "/path/to/data.jsonl",
    record_to_sample
)
```

**HuggingFace dataset:**
```python
from inspect_ai.dataset import hf_dataset, FieldSpec

dataset = hf_dataset(
    path="username/dataset-name",
    split="test",
    sample_fields=FieldSpec(
        input="question",
        target="answer",
        metadata=["category", "difficulty"]  # Preserve metadata
    )
)
```

### Solver Chain Patterns

**Simple generation:**
```python
from inspect_ai.solver import chain, generate, prompt_template, system_message

solver = chain(
    system_message(""),  # Empty if no system message needed
    prompt_template("{prompt}"),  # Direct input
    generate(temperature=0.0)
)
```

**With system message and custom template:**
```python
solver = chain(
    system_message("You are an expert classifier. Respond with only the category label."),
    prompt_template("Text: {prompt}\n\nCategory:"),
    generate(temperature=0.0, max_tokens=50)
)
```

**Chain-of-thought:**
```python
from inspect_ai.solver import chain_of_thought, generate

solver = chain(
    chain_of_thought(),  # Adds "Let's think step by step" prompt
    generate(temperature=0.0)
)
```

**Multiple choice:**
```python
from inspect_ai.solver import multiple_choice

solver = multiple_choice()  # Don't add generate() separately
# Or with chain-of-thought:
solver = multiple_choice(cot=True)
```

### Scorer Patterns

**Exact matching (case-insensitive):**
```python
from inspect_ai.scorer import match

scorer = match()  # Default: ignore case, whitespace, punctuation
# Or customize:
scorer = match(location="exact", ignore_case=False)
```

**Substring matching:**
```python
from inspect_ai.scorer import includes

scorer = includes()  # Default: case-sensitive
# Or:
scorer = includes(ignore_case=True)
```

**Multiple scorers:**
```python
scorer = [
    match("exact", ignore_case=False),
    includes(ignore_case=False)
]
# Results will show scores from both
```

**Model-graded:**
```python
from inspect_ai.scorer import model_graded_qa

scorer = model_graded_qa(
    partial_credit=True,  # Allow 0.5 scores
    model="openai/gpt-4o"  # Specify grading model
)
```

## Integration with Fine-Tuning Workflow

### Experiment-Guided Task Creation (Recommended)

When creating tasks for an experiment, the task does **not** read fine-tuning configs directly. Instead it reads its prompt/scorers config from the `eval.yaml` that `scaffold-inspect` writes, via the auto-derived `config_path`:

1. `design-experiment` produces `experiment_summary.yaml` (research question, data, models, `evaluation.system_prompt`, the `scorers:` block).
2. `scaffold-inspect` writes one `eval.yaml` per (run, task, epoch) cell, carrying `prompt`, `system_prompt`, `scorers`, `data_path`, `vis_label`, etc.
3. `setup_inspect.py` renders the SLURM script: it auto-derives `config_path` (the path to that `eval.yaml`), maps `TASK_ARG_KEYS` onto `-T` flags, and the task reads `prompt`/`system_prompt`/`scorers` back out of `config_path` at runtime.

So a generated task's job is: accept the standard params, read prompt/system_prompt (and optionally the `scorers:` block) from `config_path`, build the dataset/solver/scorer, and name itself with `vis_label`.

### Generated Task Pattern

**For tasks integrated with experiments** (mirrors `blueprints/capitalization/inspect_task.py`, the cleanest reference):

```python
import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import chain, generate, system_message

from cruijff_kit.tools.inspect.scorers import (
    build_scorers,
    configured_scorers_require_logprobs,
)


@task
def my_task(
    data_path: str,
    config_path: str = "",
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 20,
    use_chat_template: bool = True,
    logprobs: bool | None = None,
    top_logprobs: int = 20,
    vis_label: str = "",
) -> Task:
    """
    Evaluate a model on <task description>.

    Args:
        data_path: Path to the dataset JSON.
        config_path: Path to eval.yaml (reads prompt/system_prompt/scorers).
        split: Which split to evaluate (default: "test").
        temperature: Generation temperature (greedy default 1e-7).
        max_tokens: Max tokens to generate.
        use_chat_template: True for instruct models (adds system_message solver),
            False for base models (plain text completion).
        logprobs: None = auto (enable iff a configured scorer needs them).
        top_logprobs: Top-k logprobs to capture when enabled.
        vis_label: Optional suffix for the task name; read by viz_helpers for
            dedup/labeling across a multi-variant experiment.

    Returns:
        Task: Configured inspect-ai task
    """
    # Task name carries vis_label so the viz pipeline can label/dedup runs.
    task_name = f"my_task_{vis_label}" if vis_label else "my_task"

    prompt_str = "{input}"
    system_prompt = ""
    config: dict = {}
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        prompt_str = config.get("prompt", "{input}")
        system_prompt = config.get("system_prompt", "")

    def record_to_sample(record):
        return Sample(
            input=prompt_str.format(input=record["input"]),
            target=record["output"],
            metadata=record.get("metadata", {}),
        )

    dataset = hf_dataset(
        path="json",
        data_files=data_path,
        field=split,
        split="train",  # HuggingFace quirk — always "train" here
        sample_fields=record_to_sample,
    )

    if use_chat_template:
        solver = chain(
            system_message(system_prompt),
            generate(temperature=temperature, max_tokens=max_tokens),
        )
    else:
        solver = chain(generate(temperature=temperature, max_tokens=max_tokens))

    # Scorer + logprobs driven by the eval config's scorers: block.
    scorers = build_scorers(config)
    scorer_needs_logprobs = configured_scorers_require_logprobs(config)
    if logprobs is False and scorer_needs_logprobs:
        raise ValueError(
            "A configured scorer requires logprobs, but logprobs=False was set. "
            "Drop the override or remove the logprob-dependent scorer."
        )
    enable_logprobs = bool(logprobs) or (logprobs is None and scorer_needs_logprobs)
    generate_config = (
        GenerateConfig(logprobs=True, top_logprobs=top_logprobs)
        if enable_logprobs
        else GenerateConfig()
    )

    return Task(
        name=task_name,
        dataset=dataset,
        solver=solver,
        scorer=scorers,
        config=generate_config,
    )
```

For a **standalone** task with no experiment context, drop the `config_path`/`build_scorers`/logprobs machinery and hard-code a built-in scorer (e.g. `scorer=match(...)`) — but keep `vis_label` (and pass it yourself via `-T vis_label=…`, since there's no `eval.yaml` to auto-map it) if the eval logs will feed the viz pipeline:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate


@task
def my_task(
    data_path: str,
    split: str = "test",
    temperature: float = 1e-7,
    max_tokens: int = 20,
    vis_label: str = "",
) -> Task:
    """Standalone eval on <task description> (no eval.yaml)."""
    task_name = f"my_task_{vis_label}" if vis_label else "my_task"

    def record_to_sample(record):
        return Sample(
            input=record["input"],
            target=record["output"],
            metadata=record.get("metadata", {}),
        )

    dataset = hf_dataset(
        path="json",
        data_files=data_path,
        field=split,
        split="train",  # HuggingFace quirk — always "train" here
        sample_fields=record_to_sample,
    )

    return Task(
        name=task_name,
        dataset=dataset,
        solver=generate(temperature=temperature, max_tokens=max_tokens),
        scorer=match(),  # hard-coded — no scorers: block to read
    )
```

### Usage Examples

In the experiment pipeline you don't invoke `inspect eval` by hand — `scaffold-inspect` + `setup_inspect.py` generate the SLURM cell. For a quick manual smoke test of a generated task:

```bash
inspect eval my_task.py@my_task \
  --model hf/local \
  -M model_path=/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-1B-Instruct \
  -T data_path=/path/to/dataset.json \
  -T config_path=/path/to/eval.yaml \
  -T vis_label=smoke \
  --limit 5
```

### Integration with setup_inspect.py

This task pattern integrates with `setup_inspect.py`, which renders eval SLURM scripts from a template. The scaffold-inspect agent writes `eval.yaml` (referencing the task script created here) and calls:
```bash
python src/tools/inspect/setup_inspect.py \
  --config eval.yaml \
  --model_name Llama-3.2-1B-Instruct
```

## Validation Before Completion

### Common Validation (Both Modes)

Before finishing, verify:
- ✓ Task file is syntactically correct Python
- ✓ All imports are present
- ✓ Task decorated with `@task`
- ✓ Dataset loading code matches format
- ✓ Solver chain follows inspect-ai patterns
- ✓ Scorer is appropriate for task
- ✓ Design documentation includes all sections
- ✓ Example usage commands are correct
- ✓ Log file documents all decisions

### Mode 1 Specific Validation

Additional checks for experiment-guided mode:
- ✓ experiment_summary.yaml was successfully parsed
- ✓ Extracted dataset path exists and format matches
- ✓ System prompt matches training configuration
- ✓ Task accepts `data_path` + `config_path` and reads prompt/system_prompt/scorers from `config_path`
- ✓ Task accepts `vis_label` and folds it into the task name
- ✓ Parameter names are within `setup_inspect.py`'s `TASK_ARG_KEYS`
- ✓ Documentation includes experiment context (research question, runs)
- ✓ Usage examples show both fine-tuned and base model evaluation
- ✓ Log includes extraction details and validation results

## Next Steps After Creation

After creating the task, guide user:

1. **Test the task:**
   ```bash
   # Validate syntax
   python -m py_compile {task_file}.py

   # Test with small sample
   inspect eval {task_file}.py --model {model} --limit 5
   ```

2. **Run full evaluation:**
   ```bash
   inspect eval {task_file}.py --model {model}
   ```

3. **View results:**
   ```bash
   inspect view
   # Opens web UI to browse evaluation logs
   ```

4. **Iterate if needed:**
   - Adjust scorer settings
   - Modify prompts
   - Change generation parameters
   - Use `inspect score` to re-score without re-running

## Important Notes

### General Best Practices
- Follow inspect-ai best practices from https://inspect.aisi.org.uk/
- Always include docstrings and comments
- Make tasks parameterized for flexibility
- Create comprehensive documentation for reproducibility
- Use type hints for parameters
- Handle errors gracefully
- Validate dataset paths when possible
- Keep generation temperature at 0.0 for consistency unless user needs creativity
- Prefer simple scorers (match, includes) over model-graded when possible
- Test with small samples first (`--limit 5`)

### Experiment Integration
- **Prefer Mode 1 (experiment-guided)** when working with designed experiments
- Always check for experiment_summary.yaml before starting
- Extract and validate all configuration before proceeding
- **System prompt consistency is critical** - eval must match training
- Generated tasks should work for both fine-tuned and base models (`use_chat_template` toggles instruct vs. base)
- Include experiment context in documentation (research question, runs)
- Read prompt/system_prompt/scorers from `config_path` (the eval.yaml), and fold `vis_label` into the task name
- Log all extraction and validation steps for reproducibility

## Error Handling

**If dataset file not found:**
- Warn user but proceed with code generation
- Note in documentation that path should be verified
- Include validation suggestion in next steps

**If unsure about dataset format:**
- Ask for example record
- Offer to help convert to supported format
- Suggest user examine file structure

**If scorer choice unclear:**
- Recommend starting with simple scorers
- Suggest using multiple scorers for comparison
- Note that scorers can be changed later without re-running generation

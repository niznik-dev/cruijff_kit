# Logging - design-experiment

**See [shared/logging_spec.md](../../shared/logging_spec.md) for complete format specification and general logging guidelines.**

This document covers design-experiment-specific logging practices.

---

## Log File Location

```
{experiment_dir}/logs/design-experiment.log
```

Created during the planning workflow to record all verification steps, calculations, and decisions.

---

## Action Types

| Action Type | Purpose |
|-------------|---------|
| `START_DESIGN` | Mark beginning of experiment design |
| `CONSULT_PRIOR_EXPERIMENT` | Examine previous experiment results |
| `CONSULT_DOCUMENTATION` | Read research papers or documentation |
| `VERIFY_MODEL` | Check model directory exists and get size |
| `VERIFY_DATASET` | Check dataset file exists and get size |
| `COUNT_DATASET_SAMPLES` | Count samples in each dataset split |
| `VERIFY_EVAL_TASK` | Check evaluation task script exists |
| `SEARCH_PRIOR_RUNS` | Search for prior runs to extract training speed |
| `EXTRACT_TRAINING_SPEED` | Extract training speed from prior run logs |
| `CALCULATE_TRAINING_TIME` | Compute estimated training time |
| `CHECK_DISK_SPACE` | Check available disk space |
| `CALCULATE_DISK_USAGE` | Estimate disk space for checkpoints |
| `USER_QUESTION` | Ask user for input during design |
| `USER_RESPONSE` | Capture user input and decisions |
| `DESIGN_DECISION` | Record key design choices with rationale |
| `GENERATE_RUNS` | Generate list of runs from variables |
| `GENERATE_EVAL_MATRIX` | Generate evaluation matrix |
| `CREATE_YAML` | Write experiment_summary.yaml |
| `CREATE_LOG` | Write the log file itself |
| `COMPLETE_DESIGN` | Mark end of experiment design |

---

## When to Log

### During Parameter Selection
- Experiment start (name, type, directory)
- Prior experiment consultations and documentation review
- All resource verifications (models, datasets, eval tasks)
- Prior run searches and training speed extraction
- Time and disk space calculations
- User questions, responses, and design decisions

### During Generation
- Run list generation
- Evaluation matrix generation
- File creation (YAML and log)

### On Completion
- Total duration
- Files created
- Summary of experiment

---

## Example Log Entries

### START_DESIGN

```
[2025-10-22 14:30:00] START_DESIGN
Details: Beginning experiment design
Experiment: cap_4L_lora_rank_comparison
Type: sanity_check
Directory: {scratch_dir}/cap_4L_lora_rank_comparison
Result: success
```

### CONSULT_PRIOR_EXPERIMENT

```
[2025-10-22 14:30:00] CONSULT_PRIOR_EXPERIMENT
Details: Examining prior experiment for design insights
Path: {scratch_dir}/prior_cap_experiment
Files examined: experiment_summary.yaml, run_001/results.json, run_001/slurm-12345.out
Insights: LoRA rank 4 achieved 95% accuracy; ~2 min/epoch on same hardware; no issues
Influence: Using same dataset and model size; testing higher ranks to explore capacity limits
Result: success
```

### CONSULT_DOCUMENTATION

```
[2025-10-22 14:30:00] CONSULT_DOCUMENTATION
Details: Reading research paper for design guidance
Source: QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
Section: Optimal LoRA rank selection
Key findings: Ranks 4-16 sufficient for most tasks; higher ranks add minimal gains; rank 8 good default for 1B models
Influence: Constraining rank search space to 4-16 range based on model size
Result: success
```

### VERIFY_MODEL

```
[2025-10-22 14:30:01] VERIFY_MODEL
Details: Verifying model directory exists
Model: Llama-3.2-1B-Instruct
Path: {scratch_dir}/llms/Meta-Llama-3.2-1B-Instruct
Size: 4.0 GB
Result: success
```

Failure example:

```
[2025-10-22 14:30:01] VERIFY_MODEL
Details: Verifying model directory exists
Model: Llama-3.2-1B-Instruct
Path: {scratch_dir}/llms/Meta-Llama-3.2-1B-Instruct
Error: FileNotFoundError - Model directory does not exist
Result: failure
```

### VERIFY_DATASET

```
[2025-10-22 14:30:01] VERIFY_DATASET
Details: Verifying dataset file exists
Path: /path/to/cruijff_kit/data/green/words_5L_80P_1000.json
Size: 84 KB
Result: success
```

### COUNT_DATASET_SAMPLES

```
[2025-10-22 14:30:01] COUNT_DATASET_SAMPLES
Details: Counting samples in dataset splits
Dataset: words_5L_80P_1000.json
Train: 1000 samples
Validation: 200 samples
Test: 200 samples
Result: success
```

### VERIFY_EVAL_TASK

```
[2025-10-22 14:30:02] VERIFY_EVAL_TASK
Details: Verifying evaluation task script exists
Task: capitalization
Path: /path/to/cruijff_kit/experiments/capitalization/cap_task.py
Size: 12 KB
Result: success
```

### SEARCH_PRIOR_RUNS

```
[2025-10-22 14:30:02] SEARCH_PRIOR_RUNS
Details: Searching for prior runs to estimate training speed
Pattern: find {scratch_dir} -name 'slurm-*.out' -path '*/ck-out-*'
Found: 3 prior run logs
Result: success
```

### EXTRACT_TRAINING_SPEED

```
[2025-10-22 14:30:03] EXTRACT_TRAINING_SPEED
Details: Extracting training speed from prior run
Source: {scratch_dir}/prior_exp/run1/slurm-123.out
Speed: 2.5 it/s
Estimated: 120 seconds/epoch
Result: success
```

### CALCULATE_TRAINING_TIME

```
[2025-10-22 14:30:04] CALCULATE_TRAINING_TIME
Details: Computing estimated training time for all runs
Basis: prior_run_average
Per epoch: 120 seconds
Epochs: 1
Runs: 2
Total: 240 seconds (4 minutes)
Result: success
```

### CHECK_DISK_SPACE

```
[2025-10-22 14:30:05] CHECK_DISK_SPACE
Details: Checking available disk space
Filesystem: {scratch_dir}
Available: 5120 GB
Used: 2048 GB
Total: 7168 GB
Result: success
```

### CALCULATE_DISK_USAGE

```
[2025-10-22 14:30:05] CALCULATE_DISK_USAGE
Details: Estimating disk space for checkpoints
Per checkpoint: 2.5 GB
Epochs: 1
Runs: 2
Total checkpoints: 2
Total: 5.0 GB
Result: success
```

### USER_QUESTION

```
[2025-10-22 14:30:06] USER_QUESTION
Details: Asking user for hyperparameter_selection input
Question: What LoRA ranks should we compare in this experiment?
Context: Prior experiments used rank 4 successfully. Literature suggests ranks 4-16 are effective for models this size.
Options: 4, 8 (quick comparison) | 4, 8, 16 (comprehensive) | 8, 16, 32 (higher capacity focus)
Result: success
```

Common question types: `hyperparameter_selection`, `dataset_choice`, `evaluation_task_selection`, `resource_allocation`, `experimental_design`

### USER_RESPONSE

```
[2025-10-22 14:30:06] USER_RESPONSE
Details: Capturing user input for hyperparameter_selection
Input: Let's use ranks 4 and 8 to keep it quick
Decision: lora_rank = [4, 8]
Rationale: User prioritized faster iteration over comprehensive comparison
Result: success
```

### DESIGN_DECISION

```
[2025-10-22 14:30:06] DESIGN_DECISION
Details: Recording variable_selection decision
Decision: independent variable lora_rank = [4, 8]
Alternatives considered:
  - lora_rank [4, 8, 16]: More comprehensive comparison, but 50% more training time
  - lora_rank [8, 16]: Test higher capacity, but no baseline comparison to prior work
Rationale: Balancing scientific rigor with computational efficiency; maintaining comparability with prior experiments
Evidence: prior_experiment_results, user_preference, resource_constraints, literature_review
Result: success
```

Common decision types: `variable_selection`, `control_run_design`, `dataset_selection`, `evaluation_strategy`, `resource_allocation`

### GENERATE_RUNS

```
[2025-10-22 14:30:06] GENERATE_RUNS
Details: Generating run list from variables
Method: cartesian_product
Variables: lora_rank = [4, 8]
Generated: 2 runs + 1 control = 3 total
Result: success
```

### GENERATE_EVAL_MATRIX

```
[2025-10-22 14:30:07] GENERATE_EVAL_MATRIX
Details: Generating evaluation matrix (runs x tasks x epochs)
Runs: 3
Tasks: 1
Total evaluations: 3
Result: success
```

### CREATE_YAML

```
[2025-10-22 14:30:08] CREATE_YAML
Details: Writing experiment configuration
Path: {scratch_dir}/cap_4L_lora_rank_comparison/experiment_summary.yaml
Size: 1456 bytes
Result: success
```

### CREATE_LOG

```
[2025-10-22 14:30:08] CREATE_LOG
Details: Writing design log
Path: {scratch_dir}/cap_4L_lora_rank_comparison/logs/design-experiment.log
Entries: 14
Result: success
```

### COMPLETE_DESIGN

```
[2025-10-22 14:30:08] COMPLETE_DESIGN
Details: Experiment design finished
Experiment: cap_4L_lora_rank_comparison
Files created: experiment_summary.yaml, logs/design-experiment.log
Duration: 8.89 seconds
Result: success
```

---

## Error Handling

When `Result: failure` or `Result: warning`, include an `Error:` line with the error type and message:

```
[2025-10-22 14:30:01] VERIFY_MODEL
Details: Verifying model directory exists
Model: Llama-3.2-1B-Instruct
Path: {scratch_dir}/llms/Meta-Llama-3.2-1B-Instruct
Error: FileNotFoundError - Model directory does not exist
Result: failure
```

---

## Writing Logs

### Write Entries Incrementally

Write log entries as soon as actions complete. Do not wait until the end to write all logs at once.

### Timestamp Format

Use `YYYY-MM-DD HH:MM:SS` format:

```
[2025-10-22 14:30:01] ACTION_TYPE
```

### Entry Structure

Each entry follows this pattern:

```
[TIMESTAMP] ACTION_TYPE
Details: {what happened}
{action-specific key: value lines}
Result: success|failure|warning
```

Separate entries with a blank line for readability.

---

## Example Complete Log

```
[2025-10-22 14:30:00] START_DESIGN
Details: Beginning experiment design
Experiment: cap_4L_lora_rank_comparison
Type: sanity_check
Directory: {scratch_dir}/cap_4L_lora_rank_comparison
Result: success

[2025-10-22 14:30:01] VERIFY_MODEL
Details: Verifying model directory exists
Model: Llama-3.2-1B-Instruct
Path: {scratch_dir}/llms/Meta-Llama-3.2-1B-Instruct
Size: 4.0 GB
Result: success

[2025-10-22 14:30:01] VERIFY_DATASET
Details: Verifying dataset file exists
Path: /path/to/cruijff_kit/data/green/words_5L_80P_1000.json
Size: 84 KB
Result: success

[2025-10-22 14:30:01] COUNT_DATASET_SAMPLES
Details: Counting samples in dataset splits
Dataset: words_5L_80P_1000.json
Train: 1000 samples
Validation: 200 samples
Test: 200 samples
Result: success

[2025-10-22 14:30:02] VERIFY_EVAL_TASK
Details: Verifying evaluation task script exists
Task: capitalization
Path: /path/to/cruijff_kit/experiments/capitalization/cap_task.py
Size: 12 KB
Result: success

[2025-10-22 14:30:02] SEARCH_PRIOR_RUNS
Details: Searching for prior runs to estimate training speed
Pattern: find {scratch_dir} -name 'slurm-*.out' -path '*/ck-out-*'
Found: 3 prior run logs
Result: success

[2025-10-22 14:30:03] EXTRACT_TRAINING_SPEED
Details: Extracting training speed from prior run
Source: {scratch_dir}/prior_exp/run1/slurm-123.out
Speed: 2.5 it/s
Estimated: 120 seconds/epoch
Result: success

[2025-10-22 14:30:04] CALCULATE_TRAINING_TIME
Details: Computing estimated training time for all runs
Basis: prior_run_average
Per epoch: 120 seconds
Epochs: 1
Runs: 2
Total: 240 seconds (4 minutes)
Result: success

[2025-10-22 14:30:05] CHECK_DISK_SPACE
Details: Checking available disk space
Filesystem: {scratch_dir}
Available: 5120 GB
Used: 2048 GB
Total: 7168 GB
Result: success

[2025-10-22 14:30:05] CALCULATE_DISK_USAGE
Details: Estimating disk space for checkpoints
Per checkpoint: 2.5 GB
Epochs: 1
Runs: 2
Total checkpoints: 2
Total: 5.0 GB
Result: success

[2025-10-22 14:30:06] GENERATE_RUNS
Details: Generating run list from variables
Method: cartesian_product
Variables: lora_rank = [4, 8]
Generated: 2 runs + 1 control = 3 total
Result: success

[2025-10-22 14:30:07] GENERATE_EVAL_MATRIX
Details: Generating evaluation matrix (runs x tasks x epochs)
Runs: 3
Tasks: 1
Total evaluations: 3
Result: success

[2025-10-22 14:30:08] CREATE_YAML
Details: Writing experiment configuration
Path: {scratch_dir}/cap_4L_lora_rank_comparison/experiment_summary.yaml
Size: 1456 bytes
Result: success

[2025-10-22 14:30:08] CREATE_LOG
Details: Writing design log
Path: {scratch_dir}/cap_4L_lora_rank_comparison/logs/design-experiment.log
Entries: 14
Result: success

[2025-10-22 14:30:08] COMPLETE_DESIGN
Details: Experiment design finished
Experiment: cap_4L_lora_rank_comparison
Files created: experiment_summary.yaml, logs/design-experiment.log
Duration: 8.89 seconds
Result: success
```

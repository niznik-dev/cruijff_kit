# Verify Inspect-AI Tasks

This module describes how to verify that inspect-ai task scripts exist and are compatible with the experiment.

## Purpose

Before scaffolding evaluation scripts, verify that the inspect-ai task files exist and contain the expected task functions.

## Verification Steps

For each evaluation task in the experiment:

### 1. Check if task script exists

```bash
ls {task_script_path}
```

If the file doesn't exist, note it for the user.

### 2. List available tasks in the script

```bash
module load anaconda3/2025.6
conda activate {conda_env}
inspect list {task_script_path}
```

This shows all `@task` decorated functions in the file, confirming:
- The task name exists
- The exact spelling/capitalization
- What other tasks are available

**Example output:**
```
inspect_task_predictable_or_not.py
  predictable_or_not
```

Use this to verify the task name in experiment_summary.md matches what's actually in the file.

### 3. Verify task compatibility

Check if the task is compatible with the experiment:
- Can it accept `config_path` parameter? (for both fine-tuned and base models)
- Can it accept `dataset_path` parameter? (fallback approach)
- Check docstring/parameters if accessible

**Preferred approach for experiments:**
Tasks that support `config_path` can read dataset path and system prompt from `setup_finetune.yaml`, making configuration more consistent for both fine-tuned and base models.

**Fallback approach:**
Tasks that accept `dataset_path` and `system_prompt` as direct parameters work when explicit control is needed.

## Handling Missing Tasks

**If task doesn't exist:**
- Note in log that task needs to be created
- Suggest running `create-inspect-task` skill first
- Continue with other tasks (don't fail completely)

**If task name doesn't match:**
- Report the mismatch (e.g., experiment_summary.md says "predictable_or_not" but file has "predictable")
- Suggest either updating experiment_summary.md or renaming the task function
- Show available tasks from `inspect list` output

## Example Verification Pattern

```bash
echo "Verifying inspect-ai tasks..."
for task in predictable_or_not bit_sequences; do
  if [ -f "path/to/inspect_task_${task}.py" ]; then
    echo "✓ inspect_task_${task}.py exists"
    inspect list "path/to/inspect_task_${task}.py" | grep "${task}"
    if [ $? -eq 0 ]; then
      echo "✓ ${task} function found"
    else
      echo "✗ ${task} function not found in file"
    fi
  else
    echo "✗ inspect_task_${task}.py not found"
  fi
done
```

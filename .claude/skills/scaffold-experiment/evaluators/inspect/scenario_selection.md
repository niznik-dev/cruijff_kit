# Evaluation Scenarios

This module describes different approaches for configuring inspect-ai evaluations based on model type and task design.

## Scenario 1: Fine-tuned Model with Config Integration

**When to use:** Task supports `config_path` parameter (preferred for experiments)

**How it works:** The task reads dataset path and system prompt from `setup_finetune.yaml`

**inspect eval command:**
```bash
inspect eval cap_task.py@capitalization \
  --model hf/local \
  -M model_path="/path/to/epoch_0" \
  -T config_path="/path/to/run_dir/setup_finetune.yaml"
```

**Advantages:**
- Configuration stays consistent with training
- No need to duplicate dataset path and system prompt
- Simpler command (fewer parameters)
- Works for both fine-tuned and base models

**Requirements:**
- Task must accept `config_path` parameter
- Task must read and parse `setup_finetune.yaml`
- `setup_finetune.yaml` must exist with correct dataset configuration

## Scenario 2: Base Model Evaluation

**When to use:** Evaluating models that weren't fine-tuned (control runs)

**How it works:** Explicitly pass dataset path and system prompt as task parameters

**inspect eval command:**
```bash
inspect eval cap_task.py@capitalization \
  --model hf/local \
  -M model_path="/path/to/base/model" \
  -T dataset_path="/path/to/test_data.json" \
  -T system_prompt=""
```

**Advantages:**
- Works with any task that accepts standard parameters
- No dependency on setup_finetune.yaml
- Explicit configuration visible in command

**Requirements:**
- Task must accept `dataset_path` parameter
- Task must accept `system_prompt` parameter
- System prompt must match what fine-tuned models used (for fair comparison)

## Scenario 3: Custom Evaluation Dataset

**When to use:** Fine-tuned model but evaluating on different dataset (e.g., generalization test)

**How it works:** Use fine-tuned model but override dataset path

**inspect eval command:**
```bash
inspect eval cap_task.py@capitalization \
  --model hf/local \
  -M model_path="/path/to/epoch_0" \
  -T dataset_path="/path/to/generalization_test.json" \
  -T system_prompt="{from_training_config}"
```

**Advantages:**
- Test generalization beyond training data
- Keep model configuration but vary evaluation data
- Compare performance across multiple test sets

**Requirements:**
- Task must accept `dataset_path` parameter
- System prompt should still match training configuration
- Alternative dataset must be compatible with task format

## Choosing the Right Scenario

**Decision tree:**

1. **Does the task support `config_path` parameter?**
   - Yes → Use **Scenario 1** (config integration) — **PREFERRED** for all model types
   - No → Continue

2. **Are you testing generalization on a different dataset?**
   - Yes → Use **Scenario 3** (custom dataset)
   - No → Use **Scenario 2** (explicit parameters)

**Note:** With the updated `config_path` approach, Scenario 1 now works for both fine-tuned and base models, making it the preferred approach in all cases where the task supports it.

## Error Handling

**If unclear which approach to use:**
- Check if task file has `config_path` parameter (preferred for experiments)
- Fall back to `dataset_path` + `system_prompt` approach
- Log the decision

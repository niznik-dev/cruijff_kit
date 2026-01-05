# Evaluation Scenarios

This module describes how to configure inspect-ai evaluations for chat_completion trained models.

## Standard Approach: Direct Parameters

All evaluations use the same approach - pass `data_path`, `prompt`, and `system_prompt` directly to inspect eval. scaffold-inspect extracts these values at scaffolding time and bakes them into the SLURM script.

**inspect eval command:**
```bash
inspect eval task.py@task_name \
  --model hf/local \
  -M model_path="$MODEL_PATH" \
  -T data_path="$DATA_PATH" \
  -T prompt="$PROMPT" \
  -T system_prompt="$SYSTEM_PROMPT"
```

## Scenario 1: Fine-tuned Model Evaluation

**Source of values:** `setup_finetune.yaml` in the run directory

**How scaffold-inspect extracts values:**
```python
# Read setup_finetune.yaml
data_path = config['input_dir_base'] + config['dataset_label'] + config['dataset_ext']
prompt = config['prompt']
system_prompt = config.get('system_prompt', '')
```

**Generated SLURM script variables:**
```bash
MODEL_PATH="/path/to/ck-out-run_name/epoch_0"
DATA_PATH="/path/to/data/words_5L_80P_1000.json"
PROMPT="{input}"
SYSTEM_PROMPT=""
```

## Scenario 2: Base Model Evaluation

**Source of values:** `experiment_summary.md` Configuration section

**How scaffold-inspect extracts values:**
- `data_path`: From Resources → Dataset → Path
- `prompt`: From Configuration → prompt
- `system_prompt`: From Configuration → System prompt
- `model_path`: From Resources → Models (base model path)

**Generated SLURM script variables:**
```bash
MODEL_PATH="/path/to/pretrained-llms/Llama-3.2-1B-Instruct"
DATA_PATH="/path/to/data/words_5L_80P_1000.json"
PROMPT="{input}"
SYSTEM_PROMPT=""
```

## Scenario 3: Custom Evaluation Dataset

**When to use:** Fine-tuned model but evaluating on different dataset (e.g., generalization test)

**How it works:** Override `data_path` in the SLURM script while keeping other parameters from training config.

**Requirements:**
- Alternative dataset must be compatible with task format
- System prompt should still match training configuration for fair comparison

## Key Principles

1. **Values are baked at scaffolding time** - No config file parsing at runtime
2. **Same parameters for all model types** - Fine-tuned and base models use identical inspect eval syntax
3. **Training/eval parity** - `prompt` and `system_prompt` must match what was used during training
4. **Explicit is better** - All values visible in the SLURM script for debugging

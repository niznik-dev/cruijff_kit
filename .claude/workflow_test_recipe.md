# Workflow Integration Test — base_recipe Inheritance

When the user says "test the workflow" and selects this option, read this brief and run `design-experiment` → `scaffold-experiment` → `run-experiment` end-to-end.

## Purpose

Integration test that `base_recipe` correctly pulls hyperparameter defaults from a torchtune recipe and that override precedence works as designed. The workflow must extract recipe defaults via the `tune` CLI and merge them with user-specified controls and per-run variables.

## Scientific question (test framing)

Does `lora_rank` matter when other hyperparameters come from a stock recipe? Again, the workflow is the subject under test, not the science.

## What to design

- **Project:** `capitalization`
- **Experiment name:** `workflow_test_recipe_{date}` (skill fills in the date)
- **Tools:** torchtune for fine-tuning, inspect-ai for evaluation
- **Model:** `Llama-3.2-1B-Instruct`
- **Dataset:** `{ck_data_dir}/capitalization/words_5L_80P_1000.json` — 1000 5-letter words, 800 train / 100 val / 100 test
- **Runs:** two fine-tuned runs varying only LoRA rank; everything else flows from the recipe or controls
  - `Llama-3.2-1B-Instruct_rank4` — `lora_rank: 4` (overrides recipe default of 64)
  - `Llama-3.2-1B-Instruct_rank16` — `lora_rank: 16` (overrides recipe default of 64)
- **Evaluation:** the `capitalization` task at `blueprints/capitalization/inspect_task.py`, same dataset, last epoch, both runs

## Common training settings

- **`base_recipe: llama3_2/1B_lora_single_device`** — provides the defaults under test
- Explicit overrides:
  - `epochs: 1`
  - `batch_size: 4` (overrides whatever the recipe specifies)
- **Not** specified — should fall through to recipe defaults:
  - `lr` (recipe default: `3e-4`)
  - `gradient_accumulation_steps` (recipe default: `8`)
  - `weight_decay` (recipe default: `0.01`)
  - `num_warmup_steps` (recipe default: `100`)
- 1 GPU
- System prompt: `You are a helpful assistant.`
- Prompt template: `Capitalize the given word: {input}\n`
- `validation_during_training: true`

## Parameter precedence under test

The skill resolves parameters in this order (highest precedence first):

1. CLI args (not exercised here)
2. `setup_finetune.yaml` values (controls + per-run variables)
3. Recipe defaults (from `base_recipe`)
4. Argparse defaults (final fallback)

Expected source for each key in this test:

| Parameter | Source | Value |
|---|---|---|
| `lora_rank` | run variables | 4 or 16 (overrides recipe's 64) |
| `lr` | recipe | 3e-4 |
| `batch_size` | controls | 4 (overrides recipe) |
| `epochs` | controls | 1 |
| `gradient_accumulation_steps` | recipe | 8 |
| `weight_decay` | recipe | 0.01 |

## Expected outputs (per run directory)

Each of `Llama-3.2-1B-Instruct_rank4/` and `Llama-3.2-1B-Instruct_rank16/` should contain:

- `setup_finetune.yaml`
- `finetune.yaml`
- `finetune.slurm`
- `eval/capitalization_epoch0.slurm`
- `slurm-*.out`
- `eval/logs/*.eval`

## Validation checks

Standard:

- All run directories created
- Training jobs complete successfully and produce `epoch_0/` checkpoint dirs
- Evaluation jobs complete successfully and produce result logs

Recipe-specific:

- `setup_finetune.yaml` carries a `base_recipe` field
- `finetune.yaml` `lora_rank` matches the run name (4 or 16, **not** the recipe default 64)
- `finetune.yaml` `lr` is the recipe default (`3e-4`) since it wasn't overridden
- `finetune.yaml` `gradient_accumulation_steps` is the recipe default (`8`)
- `finetune.yaml` `batch_size` is the override value (`4`), not the recipe default
- Console output shows `Loaded defaults from recipe: llama3_2/1B_lora_single_device`

## Estimated wall time

- Training: ~10 min
- Evaluation: ~2 min
- Total: ~12 min

## Cleanup

Don't auto-delete; keep logs for debugging.

# Workflow Integration Test — LoRA Comparison

When the user says "test the workflow" and selects this option, read this brief and run `design-experiment` → `scaffold-experiment` → `run-experiment` end-to-end.

## Purpose

Integration test that the full workflow (design, scaffold, run) executes correctly for a minimal fine-tuning experiment with two LoRA ranks. Catches regressions in skills and skill documentation.

## Scientific question (test framing)

Does varying LoRA rank affect downstream capitalization accuracy on a small dataset? The science isn't the point — the point is exercising parameter variation through the workflow.

## What to design

- **Project:** `capitalization`
- **Experiment name:** `workflow_test_{date}` (skill fills in the date)
- **Tools:** torchtune for fine-tuning, inspect-ai for evaluation
- **Model:** `Llama-3.2-1B-Instruct` (small enough for fast iteration)
- **Dataset:** `{ck_data_dir}/capitalization/words_5L_80P_1000.json` — 1000 5-letter words, 800 train / 100 val / 100 test
- **Runs:** two fine-tuned runs that differ only in LoRA rank
  - `Llama-3.2-1B-Instruct_rank4` — `lora_rank: 4`
  - `Llama-3.2-1B-Instruct_rank8` — `lora_rank: 8`
- **Evaluation:** the `capitalization` task at `blueprints/capitalization/inspect_task.py`, run on the same dataset, last epoch only, both runs

## Common training settings

- 1 epoch
- `batch_size: 4` (works on 40GB GPUs without further constraints)
- `lr: 1e-4`
- 1 GPU
- System prompt: `You are a helpful assistant.`
- Prompt template (used for **both** training and eval — must include `{input}`): `Capitalize the given word: {input}\n`
- `validation_during_training: true`

## Expected outputs (per run directory)

Each of `Llama-3.2-1B-Instruct_rank4/` and `Llama-3.2-1B-Instruct_rank8/` should contain:

- `setup_finetune.yaml`
- `finetune.yaml`
- `finetune.slurm`
- `eval/capitalization_epoch0.slurm`
- `slurm-*.out` (training log)
- `eval/logs/*.eval` (evaluation log)

## Validation checks

- All run directories created
- `setup_finetune.yaml` carries the correct `lora_rank` per run
- `finetune.yaml` parameters match the run name
- Training jobs complete successfully and produce `epoch_0/` checkpoint dirs
- Evaluation jobs complete successfully and produce result logs

## Estimated wall time

- Training: ~10 min (1B model, 1 epoch, 1000 samples)
- Evaluation: ~2 min (2 runs × 1 task × 1 epoch)
- Total: ~12 min

## Cleanup

Don't auto-delete; keep logs for debugging.

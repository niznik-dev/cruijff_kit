# Workflow Integration Test — Base vs Fine-tuned

When the user says "test the workflow" and selects this option, read this brief and run `design-experiment` → `scaffold-experiment` → `run-experiment` end-to-end.

## Purpose

Integration test that the workflow handles **base model evaluation** correctly alongside fine-tuning. Verifies the skill chain knows how to skip training for control runs and still produce evaluations.

## Scientific question (test framing)

Does fine-tuning improve capitalization accuracy over the base model? Again, the science isn't the point — the point is exercising the base/fine-tuned mixed-run path.

## What to design

- **Project:** `capitalization`
- **Experiment name:** `workflow_test_base_{date}` (skill fills in the date)
- **Tools:** torchtune for fine-tuning, inspect-ai for evaluation
- **Model:** `Llama-3.2-1B-Instruct`
- **Dataset:** `{ck_data_dir}/capitalization/words_5L_80P_1000.json` — 1000 5-letter words, 800 train / 100 val / 100 test
- **Runs:** one control + one fine-tuned
  - `Llama-3.2-1B-Instruct_base` — control, no training (the base model evaluated as-is)
  - `Llama-3.2-1B-Instruct_rank4` — fine-tuned, `lora_rank: 4`
- **Evaluation:** the `capitalization` task at `blueprints/capitalization/inspect_task.py`, same dataset, last epoch, both runs

## Common training settings

(Apply to the fine-tuned run; the base run skips training.)

- 1 epoch
- `batch_size: 4`
- `lr: 1e-4`
- 1 GPU
- System prompt: `You are a helpful assistant.`
- Prompt template: `Capitalize the given word: {input}\n`
- `validation_during_training: true`

## Expected outputs

**Both runs** (`Llama-3.2-1B-Instruct_base/`, `Llama-3.2-1B-Instruct_rank4/`):

- `setup_finetune.yaml` (the base run carries it but doesn't act on it)
- `eval/capitalization_epoch0.slurm`
- `eval/logs/*.eval`

**Fine-tuned run only:**

- `finetune.yaml`
- `finetune.slurm`
- `slurm-*.out` (training log)

## Validation checks

- Both run directories created
- Base run has eval configs but **no training configs**
- Fine-tuned run has both training and eval configs
- `setup_finetune.yaml` carries the correct `lora_rank` for the fine-tuned run
- Fine-tuning job completes successfully and produces `epoch_0/` checkpoint
- Evaluation jobs complete successfully for **both** runs and produce result logs
- The base run's evaluation `checkpoint_path` points to the HuggingFace model (not a local checkpoint)

## Estimated wall time

- Training: ~10 min (only one run trains)
- Evaluation: ~2 min (2 runs × 1 task × 1 epoch)
- Total: ~12 min

## Cleanup

Don't auto-delete; keep logs for debugging.

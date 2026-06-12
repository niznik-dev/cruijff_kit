# Supported Models

This page lists all models currently supported by cruijff_kit. The canonical source of truth is [`src/tools/torchtune/model_configs.py`](../src/tools/torchtune/model_configs.py), which contains the `MODEL_CONFIGS` dictionary with tokenizer settings, checkpoint layouts, and SLURM resource requirements for each model.

## Model Table

| Model | Family | Type | Parameters | GPUs | VRAM | Download command |
|-------|--------|------|------------|------|------|------------------|
| Llama-3.2-1B | Llama | Base | 1B | 1 | 40G | `tune download meta-llama/Llama-3.2-1B --output-dir <model_directory> --hf-token <token>` |
| Llama-3.2-1B-Instruct | Llama | Instruct | 1B | 1 | 40G | `tune download meta-llama/Llama-3.2-1B-Instruct --output-dir <model_directory> --hf-token <token>` |
| Llama-3.2-3B-Instruct | Llama | Instruct | 3B | 1 | 80G | `tune download meta-llama/Llama-3.2-3B-Instruct --output-dir <model_directory> --hf-token <token>` |
| Llama-3.1-8B-Instruct | Llama | Instruct | 8B | 1 | 80G | `tune download meta-llama/Llama-3.1-8B-Instruct --output-dir <model_directory> --hf-token <token>` |
| Llama-3.3-70B-Instruct | Llama | Instruct | 70B | 4 | 80G | `tune download meta-llama/Llama-3.3-70B-Instruct --output-dir <model_directory> --hf-token <token>` |
| Qwen2.5-3B | Qwen | Base | 3B | 1 | 80G | `tune download Qwen/Qwen2.5-3B --output-dir <model_directory>` |
| Qwen2.5-3B-Instruct | Qwen | Instruct | 3B | 1 | 80G | `tune download Qwen/Qwen2.5-3B-Instruct --output-dir <model_directory>` |

**VRAM** is the per-GPU partition size you should request. For multi-GPU models like Llama-3.3-70B-Instruct, the same per-GPU partition is requested for each of the `GPUs` cards.

## Notes

- **Base vs Instruct**: Base models use the `text_completion` dataset format; instruct models use `chat_completion`. This is **not** inferred from the model — you set it explicitly per experiment via `controls.dataset_type` in `experiment_summary.yaml` (a required field). Choose `text_completion` for base models and `chat_completion` for instruct/chat models; a wrong choice silently breaks train/eval parity.
- **HuggingFace access**: Meta Llama models require requesting access on HuggingFace before downloading. Navigate to the model page, agree to the license, and wait for confirmation before running the download command.
- **HuggingFace token**: Llama downloads require `--hf-token`. Qwen models are openly available and do not require a token. **Never commit your HuggingFace token to a repository.**

## Adding a New Model

To add support for a new model, add an entry to the `MODEL_CONFIGS` dictionary in [`src/tools/torchtune/model_configs.py`](../src/tools/torchtune/model_configs.py). Each entry requires:

1. A torchtune model component (e.g., `torchtune.models.llama3_2.lora_llama3_2_1b`)
2. Checkpoint file layout (single file or multi-file pattern)
3. Tokenizer configuration with a supported `model_family` (`llama`, `mistral`, or `qwen`)
4. SLURM resource requirements (memory, GPUs, CPUs, partition/constraint)

Dataset type is **not** a model-config field — it is chosen per experiment via `controls.dataset_type` in `experiment_summary.yaml`, so the same model can be fine-tuned with either format.

If the new model uses a tokenizer format not yet supported, also add a handler in `configure_tokenizer()` and update `SUPPORTED_MODEL_FAMILIES`.

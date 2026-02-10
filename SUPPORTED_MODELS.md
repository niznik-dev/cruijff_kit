# Supported Models

This page lists all models currently supported by cruijff_kit. The canonical source of truth is [`tools/torchtune/model_configs.py`](tools/torchtune/model_configs.py), which contains the `MODEL_CONFIGS` dictionary with tokenizer settings, checkpoint layouts, and SLURM resource requirements for each model.

## Model Table

| Model | Family | Type | Parameters | GPUs | VRAM | Download command |
|-------|--------|------|------------|------|------|------------------|
| Llama-3.2-1B | Llama | Base | 1B | 1 | 40G | `tune download meta-llama/Llama-3.2-1B --output-dir <model_dir> --hf-token <token>` |
| Llama-3.2-1B-Instruct | Llama | Instruct | 1B | 1 | 40G | `tune download meta-llama/Llama-3.2-1B-Instruct --output-dir <model_dir> --hf-token <token>` |
| Llama-3.2-3B-Instruct | Llama | Instruct | 3B | 1 | 80G | `tune download meta-llama/Llama-3.2-3B-Instruct --output-dir <model_dir> --hf-token <token>` |
| Llama-3.1-8B-Instruct | Llama | Instruct | 8B | 1 | 80G | `tune download meta-llama/Llama-3.1-8B-Instruct --output-dir <model_dir> --hf-token <token>` |
| Llama-3.3-70B-Instruct | Llama | Instruct | 70B | 4 | 320G | `tune download meta-llama/Llama-3.3-70B-Instruct --output-dir <model_dir> --hf-token <token>` |
| Qwen2.5-3B | Qwen | Base | 3B | 1 | 80G | `tune download Qwen/Qwen2.5-3B --output-dir <model_dir>` |
| Qwen2.5-3B-Instruct | Qwen | Instruct | 3B | 1 | 80G | `tune download Qwen/Qwen2.5-3B-Instruct --output-dir <model_dir>` |

**VRAM** is the minimum GPU memory required per GPU. The SLURM memory allocation matches VRAM to ensure checkpoint saving doesn't cause out-of-memory errors.

## Notes

- **Base vs Instruct**: Base models use `text_completion` dataset format; instruct models use `chat_completion`. The scaffolding tools handle this automatically based on the model config.
- **HuggingFace access**: Meta Llama models require requesting access on HuggingFace before downloading. Navigate to the model page, agree to the license, and wait for confirmation before running the download command.
- **HuggingFace token**: Llama downloads require `--hf-token`. Qwen models are openly available and do not require a token. **Never commit your HuggingFace token to a repository.**

## Adding a New Model

To add support for a new model, add an entry to the `MODEL_CONFIGS` dictionary in [`tools/torchtune/model_configs.py`](tools/torchtune/model_configs.py). Each entry requires:

1. A torchtune model component (e.g., `torchtune.models.llama3_2.lora_llama3_2_1b`)
2. Checkpoint file layout (single file or multi-file pattern)
3. Dataset type (`text_completion` for base, `chat_completion` for instruct)
4. Tokenizer configuration with a supported `model_family` (`llama` or `qwen`)
5. SLURM resource requirements (memory, GPUs, CPUs, partition/constraint)

If the new model uses a tokenizer format not yet supported, also add a handler in `configure_tokenizer()` and update `SUPPORTED_MODEL_FAMILIES`.

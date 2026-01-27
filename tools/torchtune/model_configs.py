"""Model configurations for torchtune fine-tuning.

This module contains model-specific configurations for all supported models,
including tokenizer settings, checkpoint file patterns, and SLURM resource
requirements.

To add a new model:
1. Add an entry to MODEL_CONFIGS with the model directory name as the key
2. If the model uses a new tokenizer format, add a handler in configure_tokenizer()
   and update VALID_TOKENIZER_PATH_TYPES
"""

# Valid tokenizer path types - update when adding new model families
VALID_TOKENIZER_PATH_TYPES = {"llama", "qwen"}

# Model-specific configurations
# Keys are model directory names (e.g., "Llama-3.2-3B-Instruct")
# SLURM resources follow RAM=VRAM rule to ensure checkpoint saving doesn't OOM
MODEL_CONFIGS = {
    # -------------------------------------------------------------------------
    # Llama models - use SentencePiece tokenizer (original/tokenizer.model)
    # -------------------------------------------------------------------------
    # Base/foundation models - use text_completion
    "Llama-3.2-1B": {
        "component": "torchtune.models.llama3_2.lora_llama3_2_1b",
        "checkpoint_files": ["model.safetensors"],
        "model_type": "LLAMA3_2",
        "dataset_type": "text_completion",
        "tokenizer": {
            "component": "torchtune.models.llama3.llama3_tokenizer",
            "path_type": "llama",
        },
        "slurm": {
            "mem": "40G",
            "partition": "nomig",
            "constraint": None,
            "cpus": 4,
            "gpus": 1,
        },
    },
    # Instruct models - use chat_completion
    "Llama-3.2-1B-Instruct": {
        "component": "torchtune.models.llama3_2.lora_llama3_2_1b",
        "checkpoint_files": ["model.safetensors"],
        "model_type": "LLAMA3_2",
        "dataset_type": "chat_completion",
        "tokenizer": {
            "component": "torchtune.models.llama3.llama3_tokenizer",
            "path_type": "llama",
        },
        "slurm": {
            "mem": "40G",
            "partition": "nomig",  # Avoid MIG partitions by default
            "constraint": None,
            "cpus": 4,
            "gpus": 1,
        },
    },
    "Llama-3.2-3B-Instruct": {
        "component": "torchtune.models.llama3_2.lora_llama3_2_3b",
        "checkpoint_files": {
            "filename_format": "model-{}-of-{}.safetensors",
            "max_filename": "00002",
        },
        "model_type": "LLAMA3_2",
        "dataset_type": "chat_completion",
        "tokenizer": {
            "component": "torchtune.models.llama3.llama3_tokenizer",
            "path_type": "llama",
        },
        "slurm": {
            "mem": "80G",
            "partition": None,
            "constraint": "gpu80",
            "cpus": 4,
            "gpus": 1,
        },
    },
    "Llama-3.1-8B-Instruct": {
        "component": "torchtune.models.llama3_1.lora_llama3_1_8b",
        "checkpoint_files": {
            "filename_format": "model-{}-of-{}.safetensors",
            "max_filename": "00004",
        },
        "model_type": "LLAMA3",
        "dataset_type": "chat_completion",
        "tokenizer": {
            "component": "torchtune.models.llama3.llama3_tokenizer",
            "path_type": "llama",
        },
        "slurm": {
            "mem": "80G",
            "partition": None,
            "constraint": "gpu80",
            "cpus": 4,
            "gpus": 1,
        },
    },
    "Llama-3.3-70B-Instruct": {
        "component": "torchtune.models.llama3_3.lora_llama3_3_70b",
        "checkpoint_files": {
            "filename_format": "model-{}-of-{}.safetensors",
            "max_filename": "00030",
        },
        "model_type": "LLAMA3",
        "dataset_type": "chat_completion",
        "tokenizer": {
            "component": "torchtune.models.llama3.llama3_tokenizer",
            "path_type": "llama",
        },
        "slurm": {
            "mem": "320G",
            "partition": None,
            "constraint": "gpu80",
            "cpus": 16,
            "gpus": 4,
        },
    },
    # -------------------------------------------------------------------------
    # Qwen2.5 models - use BPE tokenizer (vocab.json + merges.txt)
    # -------------------------------------------------------------------------
    "Qwen2.5-3B": {
        "component": "torchtune.models.qwen2_5.lora_qwen2_5_3b",
        "checkpoint_files": {
            "filename_format": "model-{}-of-{}.safetensors",
            "max_filename": "00002",
        },
        "model_type": "QWEN2",
        "dataset_type": "text_completion",
        "tokenizer": {
            "component": "torchtune.models.qwen2_5.qwen2_5_tokenizer",
            "path_type": "qwen",
        },
        "slurm": {
            "mem": "80G",
            "partition": None,
            "constraint": "gpu80",
            "cpus": 4,
            "gpus": 1,
        },
    },
    "Qwen2.5-3B-Instruct": {
        "component": "torchtune.models.qwen2_5.lora_qwen2_5_3b",
        "checkpoint_files": {
            "filename_format": "model-{}-of-{}.safetensors",
            "max_filename": "00002",
        },
        "model_type": "QWEN2",
        "dataset_type": "chat_completion",
        "tokenizer": {
            "component": "torchtune.models.qwen2_5.qwen2_5_tokenizer",
            "path_type": "qwen",
        },
        "slurm": {
            "mem": "80G",
            "partition": None,
            "constraint": "gpu80",
            "cpus": 4,
            "gpus": 1,
        },
    },
}

def configure_tokenizer(config, model_config, model_dir, model_name):
    """Configure tokenizer settings based on model family.

    Args:
        config: The configuration dictionary to modify (must have 'tokenizer' key)
        model_config: Model-specific configuration from MODEL_CONFIGS
        model_dir: Directory name of the model within models_dir
        model_name: Name of the model (for error messages)

    Returns:
        Modified configuration dictionary

    Raises:
        ValueError: If tokenizer path_type is missing or unknown
    """
    tokenizer_config = model_config.get("tokenizer", {})
    path_type = tokenizer_config.get("path_type")

    if path_type == "llama":
        # Llama models use SentencePiece tokenizer
        config["tokenizer"]["_component_"] = tokenizer_config["component"]
        config["tokenizer"]["path"] = f"${{models_dir}}/{model_dir}/original/tokenizer.model"
    elif path_type == "qwen":
        # Qwen models use BPE tokenizer with vocab.json + merges.txt
        config["tokenizer"]["_component_"] = tokenizer_config["component"]
        config["tokenizer"]["path"] = f"${{models_dir}}/{model_dir}/vocab.json"
        config["tokenizer"]["merges_file"] = f"${{models_dir}}/{model_dir}/merges.txt"
    else:
        raise ValueError(
            f"Unknown tokenizer path_type '{path_type}' for model '{model_name}'. "
            f"Supported path_types: {sorted(VALID_TOKENIZER_PATH_TYPES)}"
        )

    return config

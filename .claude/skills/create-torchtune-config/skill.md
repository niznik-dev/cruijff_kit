# Create Torchtune Config

You are helping the user create torchtune configuration files for their fine-tuning experiments.

## Your Task

Generate and validate a `.yaml` config file for each experiment based on the experiment parameters.

## Steps

### 1. Select the Base Config Template

Based on the model being used, pick the appropriate default config file:
- Config templates: https://github.com/meta-pytorch/torchtune/tree/main/recipes/configs
- Match the config to the model and recipe being used

### 2. Customize the Config

Modify the base config to match the experiment parameters. Common changes include:
- Model path and checkpoint location
- Dataset paths (training, validation, evaluation)
- LoRA rank (if varying)
- Batch size, learning rate, epochs
- Output directory for checkpoints
- Any other hyperparameters specific to the experiment

### 3. Critical Checks

**Model name consistency:** Double-check all lines that reference the model name. The mapping between model name and config values is not always consistent.

**Example issue:** Some Llama models require `checkpointer.model_type: LLAMA3` even when the model is version 3.X.

**How to verify:** Check the official configs at https://github.com/meta-pytorch/torchtune/tree/main/recipes/configs for the correct values.

### 4. Follow Best Practices

Ensure your config follows the recommended best practices:
- Reference: https://meta-pytorch.org/torchtune/main/deep_dives/configs.html#best-practices-for-writing-configs
- Use clear, descriptive names
- Keep configs organized and well-commented
- Use environment variables for paths when appropriate

### 5. Validate the Config

Before finalizing, validate the config file using:

```bash
tune validate <path-to-config.yaml>
```

Documentation: https://meta-pytorch.org/torchtune/main/tune_cli.html#validate-a-config

Fix any errors reported by the validation tool.

## Important Paths

- Base LLMs: `/scratch/gpfs/MSALGANIK/pretrained-llms`
- Output location: `/scratch/gpfs/MSALGANIK/mjs3`
- All base LLMs follow HuggingFace naming conventions

## Output

Create one validated `.yaml` config file in each experiment subdirectory.

## Next Steps

After creating and validating all configs, suggest using the `create-slurm-script` skill to generate SLURM scripts for running the experiments.

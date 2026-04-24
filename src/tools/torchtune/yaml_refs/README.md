# YAML Reference Configs

This folder contains default torchtune configuration files for reference purposes. These are the baseline configs that come with torchtune for different model sizes.

## Why keep these?

Instead of running `tune cp` every time you need to check the default parameters for a model configuration, you can refer to these files directly. They serve as:

- **Quick reference** for default hyperparameters (LoRA rank, alpha, learning rates, etc.)
- **Documentation** of what the out-of-the-box configs look like
- **Comparison baseline** when customizing your own configs

## Files

- `1B_defaults.yaml` - Default config for Llama 3.1 1B Instruct with LoRA
- `8B_defaults.yaml` - Default config for Llama 3.1 8B Instruct with LoRA

## Note

These are snapshots for reference only. For actual fine-tuning, use the setup scripts in `tools/torchtune/` which generate customized configs based on your specific needs.

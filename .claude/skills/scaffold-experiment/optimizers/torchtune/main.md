# Torchtune Optimizer Module

Generates torchtune fine-tuning configurations for all runs in an experiment.

## Prerequisites

- experiment_summary.md exists with torchtune specified
- claude.local.md exists with environment settings
- Fine-tuning templates exist (setup_finetune_json.yaml or setup_finetune_parquet.yaml)

## Submodules

- **[parsing.md](parsing.md)** - Extract config from experiment_summary.md and claude.local.md
- **[directory_name_selection.md](directory_name_selection.md)** - Algorithm for run directory names based on varying parameters
- **[directory_generation.md](directory_generation.md)** - Create run directories
- **[yaml_generation.md](yaml_generation.md)** - Generate setup_finetune.yaml files
- **[script_execution.md](script_execution.md)** - Execute setup_finetune.py to create finetune.yaml and finetune.slurm
- **[validation.md](validation.md)** - Verify parameter correctness in generated files

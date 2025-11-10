# Torchtune Optimizer Module

Generates torchtune fine-tuning configurations for all runs in an experiment.

## Prerequisites

- experiment_summary.md exists with torchtune specified
- claude.local.md exists with environment settings
- Fine-tuning templates exist (setup_finetune_json.yaml or setup_finetune_parquet.yaml)

## Submodules

- **[parsing.md](parsing.md)** - Extract config from experiment_summary.md and claude.local.md
- **[directory_naming.md](directory_naming.md)** - Algorithm for run directory names based on varying parameters
- **[yaml_generation.md](yaml_generation.md)** - Generate setup_finetune.yaml files
- **[execution.md](execution.md)** - Execute setup_finetune.py to create finetune.yaml and finetune.slurm
- **[validation.md](validation.md)** - Verify parameter correctness in generated files

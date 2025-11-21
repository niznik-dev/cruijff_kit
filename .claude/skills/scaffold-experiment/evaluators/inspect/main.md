# Inspect-ai Evaluator Module

Generates inspect-ai evaluation configurations for all runs in an experiment.

## Prerequisites

- experiment_summary.md exists with inspect-ai specified
- claude.local.md exists with environment settings
- Inspect-ai task scripts exist (or note as prerequisite for user to create)
- Run directories exist (created by optimizer scaffolding)

## Key Responsibilities

1. **Identify base/control runs** - Parse All Runs table to detect runs without fine-tuning (Type = "Control" or "Base")
2. **Create eval_config.yaml for base models** - Generate evaluation config files for base/control runs (they don't have setup_finetune.yaml)
3. **Generate SLURM evaluation scripts** - Create inspect.slurm files for all evaluations with correct model and config paths
4. **Verify task scripts** - Confirm inspect-ai task files exist and are accessible

## Submodules

- **[parsing.md](parsing.md)** - Extract evaluation config from experiment_summary.md and claude.local.md, including detection of base/control runs
- **[scenario_selection.md](scenario_selection.md)** - Choose evaluation approach (fine-tuned, base, custom dataset)
- **[slurm_generation.md](slurm_generation.md)** - Generate SLURM evaluation scripts with correct model paths and config files (setup_finetune.yaml for fine-tuned, eval_config.yaml for base)
- **[validation.md](validation.md)** - Verify inspect-ai task scripts exist using `inspect list`

## Important Notes

- **Base/control models** require special handling:
  - They don't undergo fine-tuning, so no `setup_finetune.yaml` exists
  - scaffold-inspect creates `eval_config.yaml` with dataset path, system prompt, and format info
  - Evaluation SLURM scripts reference `eval_config.yaml` instead of `setup_finetune.yaml`
  - This ensures consistent configuration approach across all model types

# Inspect-ai Evaluator Module

Generates inspect-ai evaluation configurations for all runs in an experiment.

## Prerequisites

- experiment_summary.md exists with inspect-ai specified
- claude.local.md exists with environment settings
- Inspect-ai task scripts exist (or note as prerequisite for user to create)
- Run directories exist (created by optimizer scaffolding)

## Submodules

- **[parsing.md](parsing.md)** - Extract evaluation config from experiment_summary.md and claude.local.md
- **[scenario_selection.md](scenario_selection.md)** - Choose evaluation approach (fine-tuned, base, custom dataset)
- **[slurm_generation.md](slurm_generation.md)** - Generate SLURM evaluation scripts with correct model paths
- **[validation.md](validation.md)** - Verify inspect-ai task scripts exist using `inspect list`

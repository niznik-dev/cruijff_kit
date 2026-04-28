# Workflow Guide

This guide covers manual workflows for users who prefer direct control or don't have access to [Claude Code](https://docs.anthropic.com/en/docs/claude-code). If you do have Claude Code, the skills-based workflow described in the [README](../README.md#running-experiments) is the recommended approach.

## Single-Run Experiments

For full instructions on a specific experiment, see the blueprint README, e.g.:
- [Capitalization](../blueprints/capitalization/README.md) — generalization test on word capitalization
- [Folktexts (ACS)](../blueprints/folktexts/README.md) — demographic prediction tasks
- [Model Organism](../blueprints/model_organism/README.md) — synthetic sequence-labeling sanity checks

For an end-to-end walkthrough using the Claude Code skills (recommended), see [ACS_EXAMPLE.md](ACS_EXAMPLE.md).

## Running Multi-Run Experiments

For experiments with multiple runs (e.g., parameter sweeps):

1. Create an experiment directory with a subdirectory for each run
2. Copy and customize `setup_finetune.yaml` for each run
3. Generate configs for all runs:
   ```bash
   for dir in run_*/; do
     (cd "$dir" && python ../../src/tools/torchtune/setup_finetune.py)
   done
   ```
4. Submit all jobs with a stagger delay to prevent HuggingFace cache race conditions:
   ```bash
   for dir in run_*/; do
     (cd "$dir" && sbatch finetune.slurm)
     sleep 5
   done
   ```

## Troubleshooting

**Issue**: Import errors for cruijff_kit
**Solution**: Ensure you ran `make install` (or `make install-dev` for contributors) from the repository root directory.

**Issue**: `torch.cuda.is_available()` returns `False`
**Solution**: This is expected on login nodes, which typically don't have GPUs. CUDA will only be available inside a SLURM job or interactive GPU session (e.g., `salloc --gres=gpu:1`).

For additional help, see [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) or open an issue on GitHub.

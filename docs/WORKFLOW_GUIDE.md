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

1. Create an experiment directory with a subdirectory for each run. Experiments typically live outside the repo (e.g. under `{ck_data_dir}/ck-projects/{project}/{experiment_name}/`); the canonical layout is described in [ARTIFACT_LOCATIONS.md](ARTIFACT_LOCATIONS.md).
2. Copy and customize `setup_finetune.yaml` for each run.
3. Generate fine-tuning configs for all runs. After `make install` the package is importable from anywhere, so `python -m` works regardless of where the experiment dir lives:
   ```bash
   for dir in run_*/; do
     (cd "$dir" && python -m cruijff_kit.tools.torchtune.setup_finetune)
   done
   ```
4. Submit all fine-tuning jobs with a stagger delay to prevent HuggingFace cache race conditions:
   ```bash
   for dir in run_*/; do
     (cd "$dir" && sbatch finetune.slurm)
     sleep 5
   done
   ```
5. Once fine-tuning completes, set up evaluation. Each run gets an `eval/` subdirectory with an `eval_config.yaml` (see `.claude/skills/scaffold-experiment/evaluators/inspect_agent.md` for the full schema). Render the eval SLURM scripts:
   ```bash
   for dir in run_*/eval/; do
     (cd "$dir" && python -m cruijff_kit.tools.inspect.setup_inspect \
       --config eval_config.yaml \
       --model_name Llama-3.2-1B-Instruct)
   done
   ```
   This produces one `{task}_epoch{N}.slurm` per checkpoint. Submit them:
   ```bash
   for slurm in run_*/eval/*_epoch*.slurm; do
     (cd "$(dirname "$slurm")" && sbatch "$(basename "$slurm")")
   done
   ```
   View results with `inspect view` (on della, append `--port=$(get_free_port)`).

## Troubleshooting

**Issue**: Import errors for cruijff_kit
**Solution**: Ensure you ran `make install` (or `make install-dev` for contributors) from the repository root directory.

**Issue**: `torch.cuda.is_available()` returns `False`
**Solution**: This is expected on login nodes, which typically don't have GPUs. CUDA will only be available inside a SLURM job or interactive GPU session (e.g., `salloc --gres=gpu:1`).

For additional help, see [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) or open an issue on GitHub.

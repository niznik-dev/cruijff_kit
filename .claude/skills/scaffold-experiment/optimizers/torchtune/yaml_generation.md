# YAML Generation

This module handles generating setup_finetune.yaml files for each run.

## Template Selection

Select appropriate template based on dataset format:
- Check dataset path extension in experiment_summary.md
- If `.json` → use `experiments/capitalization/templates/finetuning/setup_finetune_json.yaml`
- If `.parquet` → use `experiments/capitalization/templates/finetuning/setup_finetune_parquet.yaml`

## setup_finetune.yaml Structure

See [templates/setup_finetune_template.yaml](../templates/setup_finetune_template.yaml) for complete structure.

For each run, populate the template with run-specific values from experiment_summary.md and claude.local.md.

## File Location

Write file to: `{experiment_dir}/{run_directory_name}/setup_finetune.yaml`

## Important Notes

- **Use absolute paths** for robustness (e.g., `/scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/...`) rather than relative paths
- **WandB project**: Prefer using `my_wandb_project` from `claude.local.md` for consistency
- **Learning rate format**: Keep scientific notation format from experiment summary (1e-5, 5e-5, etc.)
- **Parameter name**: Use `lr` not `learning_rate` in the YAML
- **System prompt**: Often an empty string `""` if not specified
- **Account field**: Only include if present in claude.local.md SLURM Defaults

## Error Handling

**If template not found:**
- Report which template was expected
- Ask user to verify task and dataset format
- Suggest checking template path

**If model or dataset paths don't exist:**
- Warn user but proceed (paths might be correct on compute nodes)
- Note in log which paths couldn't be verified

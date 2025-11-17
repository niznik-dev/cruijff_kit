# Directory Generation

This module handles creating run directories based on the naming algorithm from directory_name_selection.md.

## Prerequisites

- Directory names determined (from directory_name_selection.md)
- Experiment directory exists and is accessible

## Single Directory Creation

For each run:

```bash
mkdir -p {experiment_dir}/{run_directory_name}
```

**Example:**
```bash
mkdir -p /scratch/gpfs/MSALGANIK/niznik/cap_experiment_2025-11-11/r8_lr1e-5
```

## Batch Directory Creation

Create all run directories at once:

```bash
cd {experiment_dir}
mkdir -p r8_lr1e-5 r16_lr1e-5 r32_lr1e-5
```

Or using a loop:

```bash
for run_name in "${run_names[@]}"; do
  mkdir -p "{experiment_dir}/${run_name}"
done
```

## Verification

After creation, verify directories exist:

```bash
ls -d {experiment_dir}/*/
```

## Error Handling

**If directory creation fails:**
- Check write permissions on experiment directory
- Verify experiment directory path is correct
- Check for invalid characters in directory names
- Report error and continue with other directories if possible

## Important Notes

- Use `mkdir -p` to avoid errors if directory already exists (idempotent)
- Create directories before attempting to write config files into them
- Directory names should match exactly what was determined in directory_name_selection.md

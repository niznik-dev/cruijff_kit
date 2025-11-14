# Executing setup_finetune.py

This module describes how to run setup_finetune.py to generate finetune.yaml and finetune.slurm files.

## Prerequisites

- `setup_finetune.yaml` exists in each run directory
- Conda environment is specified in claude.local.md
- cruijff_kit is accessible (typically in scratch directory)

## Single Run Execution

For each run directory:

### 1. Activate conda environment (CRITICAL)

```bash
module load anaconda3/2025.6
conda activate cruijff
```

**Important:** The script will fail with `ModuleNotFoundError: No module named 'cruijff_kit'` if the conda environment is not activated first.

### 2. Navigate to run directory

```bash
cd {experiment_dir}/{run_directory_name}
```

### 3. Execute setup script

```bash
python /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/tools/torchtune/setup_finetune.py
```

**Note:** Use absolute path to setup_finetune.py for robustness. Adjust based on actual cruijff_kit location.

### 4. Verify outputs exist

- `finetune.yaml` should be created
- `finetune.slurm` should be created

### 5. Capture any errors

Report errors to user if script fails.

## Batch Execution

Process all runs in one command:

```bash
module load anaconda3/2025.6
conda activate cruijff
cd {experiment_dir}
for dir in rank*/; do
  echo "Processing $dir..."
  (cd "$dir" && python /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/tools/torchtune/setup_finetune.py)
  echo "✓ $dir complete"
done
```

**Pattern for different directory structures:**
- If directories are `r8_lr1e-5/`, use `r*/`
- If directories are `b4_lr1e-5/`, use `b*/`
- If all runs share a common prefix, use that prefix with wildcard

## Path Resolution

**Finding cruijff_kit from experiment directory:**
- Experiment is typically in `{scratch_dir}/{experiment_name}/`
- cruijff_kit is typically in `{scratch_dir}/GitHub/cruijff_kit/`
- **Recommended:** Use absolute path `/scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/` for robustness
- Relative path alternative: `../GitHub/cruijff_kit/` (less reliable)

**If path doesn't work:**
- Ask user where cruijff_kit is located
- Always prefer absolute paths over relative paths

## Error Handling

**If setup_finetune.py fails for a run:**
- Log the error
- Continue with remaining runs
- Report all failures at the end

## Expected Outputs

After successful execution, each run directory should contain:
- `setup_finetune.yaml` (input, already existed)
- `finetune.yaml` (output, generated)
- `finetune.slurm` (output, generated)

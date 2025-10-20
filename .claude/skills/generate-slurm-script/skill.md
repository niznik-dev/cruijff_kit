# Generate SLURM Scripts

You are helping the user create SLURM job scripts for running fine-tuning experiments on the della cluster at Princeton.

## Automated Script Available

For batch generation of SLURM scripts, use the automated Python script:

```bash
python tools/slurm/generate_slurm_scripts.py <experiment_dir> [options]
```

This script will:
- Automatically find all subdirectories with `finetune.yaml` files
- Read each config to determine model size and resource requirements
- Parse `claude.local.md` for user-specific settings (email, account, partition, etc.)
- Generate appropriate `finetune.slurm` scripts with correct resources
- Detect and use custom recipes if specified

**Options:**
- `--dry-run`: Preview what would be generated without writing files
- `--overwrite`: Replace existing finetune.slurm files
- `--template <path>`: Use a custom SLURM template

**Example usage:**
```bash
# Preview what will be generated
python tools/slurm/generate_slurm_scripts.py /scratch/gpfs/MSALGANIK/mjs3/my_experiment/ --dry-run

# Generate scripts
python tools/slurm/generate_slurm_scripts.py /scratch/gpfs/MSALGANIK/mjs3/my_experiment/

# Regenerate all scripts
python tools/slurm/generate_slurm_scripts.py /scratch/gpfs/MSALGANIK/mjs3/my_experiment/ --overwrite
```

## Manual Generation (if needed)

If you need to manually create scripts, start with the template from `tools/torchtune/templates/finetune_template.slurm` and customize it for each experiment.

## Key Parameters to Configure

### Resource Requirements
- `--job-name`: Descriptive name for the experiment
- `--nodes`: Number of nodes (typically 1 for single-device training)
- `--cpus-per-task`: CPU cores needed
- `--mem`: Memory allocation (adjust based on model size)
- `--time`: Time limit (HH:MM:SS) - estimate based on model size and dataset
- `--gres=gpu:X`: Number of GPUs needed

### Optional Parameters
- `--array`: For job arrays (if running multiple similar jobs)
- `--account`: Specific account to charge
- `--partition`: Specific partition to use
- `--constraint`: Hardware constraints

### Environment Setup
- `module load anaconda3/2025.6`
- Activate the appropriate conda environment
- Ensure torchtune is available

### Execution
- Create necessary output directories
- Copy config files to output directory for reference
- Run the torchtune command with the appropriate config file
- Move SLURM logs to output directory on success (check if file already exists to avoid errors)

## Important Considerations

### Avoiding Common Errors

**CRITICAL: SLURM Output File Movement**

When moving SLURM output files at the end of a job, ALWAYS check if the file already exists at the destination. Otherwise, you'll get this error:
```bash
mv: 'slurm-123.out' and '/path/to/slurm-123.out' are the same file
```

This causes jobs to be marked as FAILED even when training completed successfully!

**Incorrect approach (causes error):**
```bash
[ $? == 0 ] && mv slurm-${SLURM_JOB_ID}.out <OUTPUT_DIR>/
```

**Correct approach:**
```bash
# Only move if the file doesn't already exist at destination
[ $? == 0 ] && [ ! -f <OUTPUT_DIR>/slurm-${SLURM_JOB_ID}.out ] && mv slurm-${SLURM_JOB_ID}.out <OUTPUT_DIR>/
```

The template in `tools/torchtune/templates/finetune_template.slurm` has been updated with the correct approach.

### Other Considerations

1. **GPU Requirements**: Adjust based on model size
   - Small models (1B-3B): 1 GPU usually sufficient
   - Larger models (8B+): May need multiple GPUs or more memory

2. **Time Estimates**: Be generous with time limits
   - Consider: model size, dataset size, number of epochs
   - Better to overestimate than have jobs killed

3. **Memory**: Scale with model size
   - 1B models: ~16-32GB
   - 3B models: ~32-64GB
   - 8B+ models: ~64GB+

4. **Email Notifications**: Include user's NetID for job status updates

## Cluster Information

- Cluster: della at Princeton University
- Reference: https://researchcomputing.princeton.edu/systems/della

## Output

Create one `.slurm` script in each experiment subdirectory that:
- References the correct config file
- Has appropriate resource allocations
- Saves outputs to the experiment directory

## Next Steps

After creating all SLURM scripts, suggest using the `run-experiments` skill to submit the jobs to the cluster.

# Create SLURM Script

You are helping the user create SLURM job scripts for running fine-tuning experiments on the della cluster at Princeton.

## Your Task

Generate a SLURM script for each experiment based on the torchtune config and computational requirements.

## Base Template

Start with the template from `tools/torchtune/templates/finetune_template.slurm` and customize it for each experiment.

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
- Move SLURM logs to output directory on success

## Important Considerations

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

# Workflow Guide

This guide covers manual workflows for users who prefer direct control or don't have access to [Claude Code](https://docs.anthropic.com/en/docs/claude-code). If you do have Claude Code, the skills-based workflow described in the [README](README.md#running-experiments) is the recommended approach.

## Formatting Input Files to JSON

### Capitalization Experiment

For full instructions, see the [capitalization experiment README](experiments/capitalization/README.md).

### Twin Dataset

1. Obtain the CSV files (named `twindat_sim_?k_NN.csv`, where `?` = thousands of rows, `NN` = 24 or 99 variables) and place them in a directory (e.g., `/scratch/gpfs/$USER/zyg_raw`).

2. Run the preprocessing script (can run on a login node):

```bash
python preproc.py /path/to/csv/files
```

Multiple JSON files will be created in the same input directory; move them to a working location (e.g., `/scratch/gpfs/$USER/zyg_in`) for use in `finetune.yaml`.

## Running a Single Experiment

### Capitalization Experiment

See `experiments/capitalization/README.md` for a complete walkthrough.

### Twin Dataset (or other experiments)

1. **Create configuration file**

   Copy a config template from the experiment's `templates/finetuning/` folder:

   ```bash
   cd experiments/your_experiment/
   cp templates/finetuning/setup_finetune_json.yaml setup_finetune.yaml
   ```

2. **Edit the configuration**

   Key settings to verify:
   - `input_dir_base` - Path to your input data directory
   - `input_formatting` - Subfolder name (usually empty string `''`)
   - `dataset_label` - Dataset filename without extension
   - `conda_env` - Your conda environment name (e.g., `cruijff`)
   - `torchtune_model_name` - Model name as listed by `tune ls`
   - `output_dir_base` - Where to save model checkpoints
   - `lora_rank` - LoRA adapter rank (e.g., 8, 16, 32, 64)
   - `lr` - Learning rate (e.g., 1e-5, 5e-5)

3. **Generate SLURM scripts**

   ```bash
   python ../../tools/torchtune/setup_finetune.py
   ```

   This creates `finetune.yaml` (torchtune configuration) and `finetune.slurm` (SLURM batch script).

4. **Submit the job**

   ```bash
   sbatch finetune.slurm
   ```

5. **Monitor the job**

   ```bash
   squeue -u $USER
   tail -f slurm-*.out
   ```

## Running Multi-Run Experiments

For experiments with multiple runs (e.g., parameter sweeps):

1. Create an experiment directory with a subdirectory for each run
2. Copy and customize `setup_finetune.yaml` for each run
3. Generate configs for all runs:
   ```bash
   for dir in run_*/; do
     (cd "$dir" && python ../../tools/torchtune/setup_finetune.py)
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

For additional help, see [KNOWN_ISSUES.md](KNOWN_ISSUES.md) or open an issue on GitHub.

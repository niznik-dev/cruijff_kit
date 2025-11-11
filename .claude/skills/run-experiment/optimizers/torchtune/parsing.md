# Parsing - Input Processing

Extract experiment information and identify fine-tuning runs to execute.

## Parse experiment_summary.md

### Required Information

1. **Experiment name** - From title (line 1)
2. **All runs** - From "All Runs" table
3. **Run directories** - Infer from scaffolding pattern (e.g., `r8_lr1e-5/`)

## Scan for Fine-Tuning Jobs

Find all directories containing `finetune.slurm`:

```bash
for dir in */; do
  if [ -f "$dir/finetune.slurm" ]; then
    echo "Found run: $dir"
  fi
done
```

## Build Run List

For each directory with `finetune.slurm`:

**Collect:**
- Run directory name (e.g., `r8_lr1e-5`)
- Path to SLURM script (`{run_dir}/finetune.slurm`)
- Current status from experiment_summary.md (if exists)

**Output:** List of all potential runs to execute

## Next Stage

Pass run list to run_selection.md to determine which need submission.

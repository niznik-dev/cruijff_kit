# Parsing - Input Processing

Extract experiment information and identify evaluations to execute.

## Parse experiment_summary.md

### Required Information

1. **Experiment name** - From title (line 1)
2. **All runs** - From "All Runs" table
3. **Evaluation plan** - From "Evaluations" status table:
   - Which run/task/epoch combinations to evaluate
   - Which evaluations are already completed
4. **Output directories** - Where fine-tuned models were saved

## Scan for Evaluation Jobs

Find all evaluation SLURM scripts:

```bash
for run_dir in */; do
  if [ -d "$run_dir/eval" ]; then
    for eval_script in "$run_dir/eval"/*.slurm; do
      if [ -f "$eval_script" ]; then
        echo "Found evaluation: $eval_script"
      fi
    done
  fi
done
```

## Build Evaluation List

For each `eval/*.slurm` file found:

**Collect:**
- Run directory name (e.g., `r8_lr1e-5`)
- Task name (extracted from filename, e.g., `capitalization`)
- Epoch number (extracted from filename, e.g., `0`)
- Path to SLURM script (`{run_dir}/eval/{task}_epoch{N}.slurm`)
- Expected model checkpoint path (`{output_dir_base}/ck-out-{run_name}/epoch_{N}/`)
- Current status from experiment_summary.md (if exists)

**Output:** List of all potential evaluations to execute

## Next Stage

Pass evaluation list to dependency_checking.md to verify prerequisites met.

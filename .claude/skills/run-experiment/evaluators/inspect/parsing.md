# Parsing - Input Processing

Extract experiment information and identify evaluations to execute.

## Parse experiment_summary.yaml

### Required Information

Load the YAML file and extract:

1. **Experiment name** - `experiment.name`
2. **Evaluation matrix** - `evaluation.matrix[]` (run/task/epoch combinations)
3. **Output directory** - `output.base_directory` (where checkpoints are saved)

### Example YAML Parsing

```python
import yaml

with open('experiment_summary.yaml', 'r') as f:
    config = yaml.safe_load(f)

experiment_name = config['experiment']['name']
eval_matrix = config['evaluation']['matrix']
output_base = config['output']['base_directory']

for entry in eval_matrix:
    run_name = entry['run']
    tasks = entry['tasks']  # List of task names
    epochs = entry['epochs']  # List of epoch numbers (or None for control runs)

    # For control runs, epochs will be None
    if epochs is None:
        # Base model evaluation - no checkpoint path
        for task in tasks:
            print(f"Eval: {run_name} / {task} / base")
    else:
        # Fine-tuned model evaluation - checkpoint paths
        for task in tasks:
            for epoch in epochs:
                checkpoint_path = f"{output_base}/ck-out-{run_name}/epoch_{epoch}/"
                print(f"Eval: {run_name} / {task} / epoch {epoch}")
```

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

For each entry in evaluation matrix:

**Collect:**
- Run name from YAML (`evaluation.matrix[].run`)
- Task list from YAML (`evaluation.matrix[].tasks`)
- Epoch list from YAML (`evaluation.matrix[].epochs`)
- Path to SLURM script (`{run_name}/eval/{task}_epoch{N}.slurm` or `{run_name}/eval/{task}_base.slurm`)
- Expected model checkpoint path (if fine-tuned run)
- Verify directories exist

**Output:** List of all potential evaluations to execute

## Next Stage

Pass evaluation list to dependency_checking.md to verify prerequisites met.

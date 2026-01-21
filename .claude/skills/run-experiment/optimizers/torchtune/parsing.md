# Parsing - Input Processing

Extract experiment information and identify fine-tuning runs to execute.

## Parse experiment_summary.yaml

### Required Information

Load the YAML file and extract:

1. **Experiment name** - `experiment.name`
2. **Fine-tuning runs** - Filter `runs[]` where `type == "fine-tuned"`
3. **Run directories** - Use `runs[].name` as directory name

### Example YAML Parsing

```python
import yaml

with open('experiment_summary.yaml', 'r') as f:
    config = yaml.safe_load(f)

experiment_name = config['experiment']['name']
finetune_runs = [run for run in config['runs'] if run['type'] == 'fine-tuned']

for run in finetune_runs:
    run_name = run['name']  # e.g., "Llama-3.2-1B-Instruct_rank4"
    # Expect directory: {run_name}/
```

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

For each fine-tuning run from YAML:

**Collect:**
- Run name from YAML (`runs[].name`)
- Path to SLURM script (`{run_name}/finetune.slurm`)
- Verify directory exists

**Output:** List of all potential runs to execute

## Next Stage

Pass run list to run_selection.md to determine which need submission.

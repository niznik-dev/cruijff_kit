# Parsing Evaluation Configuration

This module handles parsing experiment_summary.yaml and claude.local.md to extract evaluation configuration.

## Parsing experiment_summary.yaml

Load the YAML file and extract required fields:

```python
import yaml

with open(f"{experiment_dir}/experiment_summary.yaml", 'r') as f:
    config = yaml.safe_load(f)
```

### YAML Structure → Extracted Fields

| What | YAML Path | Example |
|------|-----------|---------|
| Experiment name | `config['experiment']['name']` | `"workflow_test_2025-11-27"` |
| Experiment dir | `config['experiment']['directory']` | `"/scratch/.../ck-sanity-checks/..."` |
| Model paths | `config['models']['base'][i]['path']` | `"/scratch/.../pretrained-llms/..."` |
| System prompt | `config['evaluation']['system_prompt']` | `""` or `"You are..."` |
| Temperature | `config['evaluation']['temperature']` | `0.0` |
| Scorer | `config['evaluation']['scorer']` | `"match"` |
| Output base dir | `config['output']['base_directory']` | `"/scratch/.../sarahep"` |
| Checkpoint pattern | `config['output']['checkpoint_pattern']` | `"ck-outputs/.../ck-out-{run_name}/epoch_{N}"` |

### Evaluation Tasks

Extract from `config['evaluation']['tasks']`:

```python
tasks = config['evaluation']['tasks']
for task in tasks:
    task_name = task['name']  # e.g., "capitalization"
    script_path = task['script']  # e.g., "/home/.../cap_task.py"
    dataset = task.get('dataset')  # Optional, may be None
    description = task['description']  # Human-readable description
```

### Evaluation Matrix

Extract from `config['evaluation']['matrix']`:

```python
eval_matrix = config['evaluation']['matrix']
for entry in eval_matrix:
    run_name = entry['run']  # e.g., "Llama-3.2-1B-Instruct_rank4"
    tasks_list = entry['tasks']  # e.g., ["capitalization", "reasoning"]
    epochs = entry['epochs']  # e.g., [0, 1] or null for control runs
```

**Key points:**
- Fine-tuned runs: `epochs: [0, 1, 2]` - list of which epochs to evaluate
- Control runs: `epochs: null` - no epoch suffix (evaluate base model once)

### Identifying Control/Base Runs

Control runs need `eval_config.yaml` (no fine-tuning configs exist):

```python
control_runs = [r for r in config['runs'] if r['type'] == 'control']

for run in control_runs:
    run_name = run['name']  # e.g., "Llama-3.2-1B-Instruct_base"
    model_name = run['model']  # e.g., "Llama-3.2-1B-Instruct"

    # Find model path
    model_info = next(m for m in config['models']['base'] if m['name'] == model_name)
    model_path = model_info['path']
```

## Reading claude.local.md

Extract environment-specific settings:
- `conda_env` - Conda environment name
- `account` - SLURM account (OPTIONAL, skip if not found)

## Error Handling

Check for required evaluation fields:
```python
if not config.get('evaluation', {}).get('tasks'):
    raise ValueError("No evaluation tasks found in experiment_summary.yaml")

if not config.get('evaluation', {}).get('matrix'):
    raise ValueError("No evaluation matrix found in experiment_summary.yaml")
```

**System prompt consistency:** Verify `config['evaluation']['system_prompt']` matches `config['controls']['system_prompt']`.

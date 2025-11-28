# Parsing Experiment Configuration

This module handles parsing experiment_summary.yaml and claude.local.md to extract configuration needed for torchtune scaffolding.

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
| Model name | `config['models']['base'][0]['name']` | `"Llama-3.2-1B-Instruct"` |
| Model path | `config['models']['base'][0]['path']` | `"/scratch/.../pretrained-llms/..."` |
| Dataset path | `config['data']['training']['path']` | `"data/green/words_5L_80P_1000.json"` |
| Dataset label | `config['data']['training']['label']` | `"words_5L_80P_1000"` |
| Dataset format | `config['data']['training']['format']` | `"json"` |
| Base recipe | `config['controls']['base_recipe']` | `"lora_finetune_single_device"` |
| Epochs | `config['controls']['epochs']` | `1` |
| Batch size (default) | `config['controls']['batch_size']` | `4` |
| System prompt | `config['controls']['system_prompt']` | `""` or `"You are..."` |
| Validation | `config['controls']['validation_during_training']` | `true` |
| GPUs | `config['controls']['gpus']` | `1` |
| LoRA alpha (default) | `config['controls'].get('lora_alpha')` | `16` |
| Learning rate (default) | `config['controls'].get('learning_rate')` | `1e-4` |
| Output base dir | `config['output']['base_directory']` | `"/scratch/.../sarahep"` |
| WandB project | `config['output']['wandb_project']` | `"cruijff_kit"` |

### Processing Runs

Filter for fine-tuned runs only (skip control runs):

```python
fine_tuned_runs = [r for r in config['runs'] if r['type'] == 'fine-tuned']

for run in fine_tuned_runs:
    run_name = run['name']
    model = run['model']

    # Get parameters: try run-specific first, fall back to controls
    lora_rank = run['parameters'].get('lora_rank') or config['controls'].get('lora_rank')
    learning_rate = run['parameters'].get('learning_rate') or config['controls'].get('learning_rate')
    batch_size = run['parameters'].get('batch_size') or config['controls']['batch_size']
```

**Key point:** Parameters can be in `run['parameters']` (varied) or `config['controls']` (constant).

## Reading claude.local.md

Extract environment-specific settings:
- `conda_env` - Conda environment name
- `account` - SLURM account (OPTIONAL, skip if not found)

**Note:** `output_dir_base` and `wandb_project` come from experiment_summary.yaml.

## Error Handling

Check for required fields:
```python
if not config.get('runs'):
    raise ValueError("No runs found in experiment_summary.yaml")

fine_tuned = [r for r in config['runs'] if r['type'] == 'fine-tuned']
if not fine_tuned:
    raise ValueError("No fine-tuned runs found (only control runs)")
```
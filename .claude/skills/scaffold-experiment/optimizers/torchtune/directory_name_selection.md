# Directory Naming Algorithm

This module defines how to generate run directory names.

## Goal

Use the full run name from experiment_summary.yaml for consistency across the entire workflow.

## Algorithm

Use `runs[].name` directly from the YAML as the directory name:

```python
for run in config['runs']:
    if run['type'] == 'fine-tuned':
        run_dir_name = run['name']  # e.g., "Llama-3.2-1B-Instruct_rank4"
```

## Examples

**From experiment_summary.yaml:**
```yaml
runs:
  - name: "Llama-3.2-1B-Instruct_rank4"
    type: "fine-tuned"
    model: "Llama-3.2-1B-Instruct"
    parameters:
      lora_rank: 4
```

**Resulting directory:**
- `Llama-3.2-1B-Instruct_rank4/`

**Checkpoint path:**
- `ck-out-Llama-3.2-1B-Instruct_rank4/epoch_0/`

## Implementation Notes

- No transformation needed - use YAML run name as-is
- Ensure run names are filesystem-safe (no slashes, etc.)
- Run names appear in experiment directory, checkpoint directories, and SLURM job names

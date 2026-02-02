# Parsing: Locate Experiment and Parse Configuration

This module describes how to find the experiment directory and extract relevant information from experiment_summary.yaml.

## Finding the Experiment Directory

**If user runs skill without arguments:**
- Check if current directory contains `experiment_summary.yaml`
- If found, use current directory as experiment directory
- If not found, ask user for the experiment directory path

**If user provides a path:**
- Use that path as the experiment directory
- Verify `experiment_summary.yaml` exists at that location

```bash
# Check for experiment_summary.yaml
ls {experiment_dir}/experiment_summary.yaml
```

## Parsing experiment_summary.yaml

Load the YAML file and extract information needed for visualization:

```python
import yaml

with open(f"{experiment_dir}/experiment_summary.yaml") as f:
    config = yaml.safe_load(f)
```

### Key Sections to Extract

**1. Runs section** - List of run directories:

```yaml
runs:
  - name: 1B_10K
    type: fine-tuned
    model: Llama-3.2-1B-Instruct
    dataset: acs_employment_verbose_10000_80P

  - name: 3B_10K
    type: fine-tuned
    model: Llama-3.2-3B-Instruct
    dataset: acs_employment_verbose_10000_80P
```

Extract:
- `run['name']` → subdirectory names for loading evaluation logs
- `run['model']` → model names for grouping
- `run['dataset']` → training dataset used

**2. Variables section** - Experimental design:

The variables section supports two formats:

**Format A: Flat dictionary (common in existing experiments)**
```yaml
variables:
  model:
    - Llama-3.2-1B-Instruct
    - Llama-3.2-3B-Instruct
  sample_size:
    - 1000
    - 10000
    - 50000
```

**Format B: Nested with independent/dependent (structured)**
```yaml
variables:
  independent:
    - name: "model"
      values: ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"]
    - name: "sample_size"
      values: [1000, 10000, 50000]
  dependent:
    - name: "accuracy"
      metric: "match_accuracy"
```

Extract:
- Independent variables → used for inference logic
- Variable types (categorical, continuous, binary) → determines view selection

**3. Evaluation section** - Task information:

```yaml
evaluation:
  matrix:
    - run: 1B_balanced
      vis_label: "1B Balanced"
      tasks: [acs_income]
      epochs: [0, 1, 2]

    - run: 3B_balanced
      vis_label: "3B Balanced"
      tasks: [acs_income]
      epochs: [0, 1, 2]
```

Extract:
- `vis_label` → human-readable label for visualizations (used as task name suffix)
- Task names → for labeling plots
- Epochs → for filtering data if needed

## Extracting Variable Types for Inference

Analyze the variables section to determine appropriate visualizations. Handle both flat and nested formats:

```python
def extract_variable_info(config):
    """Extract variable information for view inference.

    Handles two formats:
    - Flat: variables: {model_size: ["1B", "3B"]}
    - Nested: variables: {independent: [{name: "model_size", values: ["1B", "3B"]}]}
    """
    variables = config.get('variables', {})
    var_info = []

    # Check for nested format (Format B)
    if 'independent' in variables:
        independent = variables.get('independent', [])
        for var in independent:
            name = var['name']
            values = var['values']
            var_info.append(_make_var_info(name, values))
    else:
        # Flat format (Format A) - each key is a variable name
        for name, values in variables.items():
            # Skip non-list values (could be metadata)
            if isinstance(values, list):
                var_info.append(_make_var_info(name, values))

    return var_info


def _make_var_info(name, values):
    """Create variable info dict with type detection."""
    # Determine type
    if len(values) == 2 and all(
        isinstance(v, bool) or str(v).lower() in ['0', '1', 'yes', 'no', 'true', 'false',
                                                    'with_prompt', 'no_prompt', 'enabled', 'disabled']
        for v in values
    ):
        var_type = 'binary'
    elif all(isinstance(v, (int, float)) for v in values):
        var_type = 'continuous'
    else:
        var_type = 'categorical'

    return {
        'name': name,
        'values': values,
        'type': var_type,
        'count': len(values)
    }
```

## Building Subdirs List

Construct the subdirs list for loading evaluation logs:

```python
def get_subdirs(config):
    """Extract run directory names from config."""
    runs = config.get('runs', [])
    return [run['name'] for run in runs]
```

## Error Handling

**If experiment_summary.yaml is missing:**
```
Error: experiment_summary.yaml not found in {experiment_dir}
Please run design-experiment skill first to create the experiment configuration.
```

**If YAML is malformed:**
```
Error: Failed to parse experiment_summary.yaml
{yaml_error_message}
Please check the file for syntax errors.
```

**If required sections are missing:**
- Log warning
- Use defaults where possible
- Report which sections were missing

## Logging

Log parsing actions to analyze-experiment.jsonl:

```json
{"action": "PARSE", "timestamp": "...", "file": "experiment_summary.yaml", "status": "success", "runs_found": 4, "variables": ["model", "sample_size"]}
```

See `logging.md` for complete format specification.

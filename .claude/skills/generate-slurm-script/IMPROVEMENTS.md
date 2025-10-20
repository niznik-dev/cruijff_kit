# Recommended Improvements for generate-slurm-script Skill

Based on usage for the `cap_cross_eval_5_9_13L_2025-10-19` experiment on 2025-10-19.

## Critical Issues to Address

### 1. Skill Registration
**Problem**: The skill is not registered as a slash command, so `/generate-slurm-script` doesn't work.

**Solution**: Add proper skill registration or create a Python script entry point similar to other skills like `setup-experiment-dirs`.

**Priority**: HIGH

---

### 2. Automation Level
**Problem**: Currently the skill is just documentation - Claude has to manually write each SLURM script one by one. This doesn't scale well for experiments with many runs (e.g., 12 runs in this case).

**Solution**: Create a Python script that:
- Accepts an experiment directory as input
- Discovers all subdirectories with `finetune.yaml` files
- Automatically generates `finetune.slurm` scripts for each
- Reads the yaml to determine model size and adjust resources accordingly
- Uses `claude.local.md` for user-specific settings (email, account, partition, etc.)

**Priority**: HIGH

**Implementation sketch**:
```python
#!/usr/bin/env python3
# .claude/skills/generate-slurm-script/generate_slurm_scripts.py

import argparse
import yaml
from pathlib import Path
from jinja2 import Template

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', help='Path to experiment directory')
    parser.add_argument('--template', default='tools/torchtune/templates/finetune_template.slurm')
    args = parser.parse_args()

    # Find all finetune.yaml files
    exp_path = Path(args.experiment_dir)
    for yaml_file in exp_path.glob('*/finetune.yaml'):
        run_dir = yaml_file.parent

        # Parse yaml to extract model info
        with open(yaml_file) as f:
            config = yaml.safe_load(f)

        # Determine resources based on model size
        resources = determine_resources(config)

        # Generate SLURM script from template
        generate_slurm(run_dir, resources)
```

---

## Enhancement Opportunities

### 3. Resource Auto-Detection
**Problem**: Resources (memory, time) are hardcoded in the skill or manually determined each time.

**Solution**:
- Read model size from `finetune.yaml` (e.g., `lora_llama3_2_1b` vs `lora_llama3_2_3b`)
- Read batch size, epochs, dataset size from yaml
- Use heuristics or lookup table to estimate resources
- Allow overrides via command line arguments

**Priority**: MEDIUM

**Example logic**:
```python
def determine_resources(config):
    model_component = config['model']['_component_']
    batch_size = config['batch_size']

    # Extract model size from component name
    if '1b' in model_component.lower():
        mem = '32G'
        time = '00:20:00'
    elif '3b' in model_component.lower():
        mem = '64G'
        time = '00:30:00'
    elif '8b' in model_component.lower():
        mem = '128G'
        time = '01:00:00'

    return {'mem': mem, 'time': time}
```

---

### 4. Template-Based Generation
**Problem**: The skill references `finetune_template.slurm` but doesn't actually use it - scripts are written from scratch.

**Solution**: Use Jinja2 templating to populate the template with experiment-specific values.

**Priority**: MEDIUM

**Benefits**:
- Single source of truth for SLURM script structure
- Easier to maintain and update
- Consistent formatting across all scripts

---

### 5. User Configuration Integration
**Problem**: User-specific settings (email, account, partition) are scattered in the skill description.

**Solution**:
- Read from `claude.local.md` to extract:
  - Email (NetID)
  - SLURM account
  - Partition
  - Constraint
  - Conda environment
- Parse this automatically rather than hardcoding

**Priority**: MEDIUM

**Example**:
```python
def parse_local_config():
    with open('claude.local.md') as f:
        content = f.read()
        # Extract settings using regex or simple parsing
        config = {
            'email': extract_field(content, 'Username'),
            'account': extract_field(content, 'Account'),
            'partition': extract_field(content, 'Partition'),
            'constraint': extract_field(content, 'Constraint'),
            'conda_env': extract_field(content, 'Default conda environment'),
        }
    return config
```

---

### 6. Recipe Detection
**Problem**: The custom recipe path is hardcoded in generated scripts.

**Solution**:
- Check if `finetune.yaml` already specifies a recipe
- Default to standard `lora_finetune_single_device` if not
- Allow override for custom recipes

**Priority**: LOW (current hardcoded approach works fine)

---

### 7. Validation & Dry Run
**Problem**: No validation before writing scripts.

**Solution**: Add validation checks:
- Verify `finetune.yaml` exists and is valid YAML
- Check that referenced datasets exist
- Verify model checkpoint directories exist
- Add `--dry-run` flag to preview without writing

**Priority**: LOW

---

### 8. Job Name Conventions
**Problem**: Job names are manually shortened (e.g., `1B_5L_r4`).

**Solution**: Auto-generate concise job names from run directory name using consistent conventions:
- Extract key info: model size, dataset, rank
- Keep under SLURM's job name length limit (typically 15-20 chars)

**Priority**: LOW (current approach works well)

---

## Implementation Priority

**Phase 1 (Essential)**:
1. Create Python script for automated generation
2. Resource auto-detection from yaml
3. Template-based generation

**Phase 2 (Nice to have)**:
1. User configuration integration from `claude.local.md`
2. Validation & dry-run mode
3. Recipe detection

**Phase 3 (Optional)**:
1. Job name convention improvements
2. Additional resource estimation heuristics

---

## Usage After Implementation

Ideal workflow:
```bash
# From cruijff_kit root
python .claude/skills/generate-slurm-script/generate_slurm_scripts.py \
    /scratch/gpfs/MSALGANIK/mjs3/cap_cross_eval_5_9_13L_2025-10-19/

# Or as a skill
/generate-slurm-script /scratch/gpfs/MSALGANIK/mjs3/cap_cross_eval_5_9_13L_2025-10-19/

# With options
python .claude/skills/generate-slurm-script/generate_slurm_scripts.py \
    /scratch/gpfs/MSALGANIK/mjs3/cap_cross_eval_5_9_13L_2025-10-19/ \
    --dry-run \
    --template custom_template.slurm \
    --overwrite
```

---

## Current Workaround

Until automated script is implemented, Claude can:
1. Read `runs_plan.md` to understand experiment structure
2. Identify all directories with `finetune.yaml`
3. Manually write each `finetune.slurm` using Write tool
4. Use consistent resource allocations based on model size

This manual approach works but is time-consuming for large experiments.

---

*Document created: 2025-10-19*
*Last updated: 2025-10-19*

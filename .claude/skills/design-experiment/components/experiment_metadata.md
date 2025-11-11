# Experiment Metadata

Determine the experiment type, location, and naming.

## Auto-Detect Experiment Type

Based on current working directory:

```python
import os

# Get current working directory
cwd = os.getcwd()

# Determine base directory based on context
if "/sanity_checks/" in cwd or cwd.endswith("/sanity_checks"):
    # Working from sanity_checks directory -> this is a sanity check
    base_dir = "/scratch/gpfs/MSALGANIK/niznik/ck-sanity-checks/"
    experiment_type = "sanity_check"
else:
    # Default to experiments
    base_dir = "/scratch/gpfs/MSALGANIK/niznik/ck-experiments/"
    experiment_type = "experiment"

# Full experiment directory
experiment_dir = f"{base_dir}{experiment_name}/"
```

## Directory Structure

- **Experiments** (research tasks): `/scratch/gpfs/MSALGANIK/niznik/ck-experiments/{experiment_name}/`
- **Sanity checks** (workflow validation): `/scratch/gpfs/MSALGANIK/niznik/ck-sanity-checks/{sanity_check_name}/`

**Outputs are automatically grouped:**
- Output directory: `/scratch/gpfs/MSALGANIK/niznik/ck-outputs/{experiment_or_sanity_check_name}/ck-out-{run_name}/`

## Confirm with User

This step is typically automated, but you should confirm:

**Are you working on a sanity check or a research experiment?**
- Auto-detect from current working directory
- If in `/sanity_checks/` directory → sanity check
- Otherwise → research experiment

**Where should this experiment be created?**
- Log the detected type and path for user confirmation

## Establish Naming

Help the user choose a descriptive experiment name that includes:
- Task/dataset indicator (e.g., `cap_8L` for capitalization 8-letter)
- Key experimental factor (e.g., `lora_comparison`, `model_sizes`)
- Date (YYYY-MM-DD format)

**Example patterns:**
- `cap_8L_lora_comparison_2025-10-18` (capitalization, varying LoRA rank)
- `twins_model_sizes_2025-10-22` (synthetic twins, varying model size)
- `reasoning_ablation_2025-11-01` (reasoning task, ablation study)

**Run naming within experiment:**
Use full model names with experimental factors:
- `Llama-3.2-1B-Instruct_rank4`
- `Llama-3.2-3B-Instruct_rank64`
- `Llama-3.2-1B-Instruct_base` (control)

## Logging

Log the detected experiment type and full path where the experiment will be created. Note in experiment_summary.md that outputs will be grouped under the same name in ck-outputs/.

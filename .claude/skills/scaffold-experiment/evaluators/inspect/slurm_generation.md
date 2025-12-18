# SLURM Script Generation

This module describes how to generate inspect.slurm evaluation scripts.

## Evaluation Naming Convention

**IMPORTANT: Epochs are 0-indexed**
- First epoch after training is `epoch_0`, not `epoch_1`
- Training for 1 epoch produces checkpoint at `epoch_0/`
- Training for 2 epochs produces `epoch_0/` and `epoch_1/`
- Evaluation script names must match: `{task_name}_epoch0.slurm`, not `epoch1`
- When experiment_summary.md says "evaluate last epoch after 1 epoch of training", use `epoch_0`

## Directory Organization

**For fine-tuned models:**
```
{experiment_dir}/{run_dir}/
├── finetune.slurm
├── finetune.yaml
├── setup_finetune.yaml
└── eval/
    ├── {task_name}_epoch0.slurm
    ├── {task_name}_epoch1.slurm
    └── logs/
```

**For base models (controls):**
```
{experiment_dir}/{run_dir}_base/
└── eval/
    ├── {task_name}_base.slurm
    └── logs/
```

## Model-Aware Resource Allocation

Different model sizes require different SLURM resources for evaluation. Parse the model name from experiment_summary.md and set resources accordingly:

| Model Size | Memory | GPUs | Constraint | CPUs | Time |
|------------|--------|------|------------|------|------|
| 1B (Llama-3.2-1B-Instruct) | 32G | 1 | - | 4 | 0:30:00 |
| 3B (Llama-3.2-3B-Instruct) | 64G | 1 | gpu80 | 4 | 0:30:00 |
| 8B (Llama-3.1-8B-Instruct, etc.) | 96G | 1 | gpu80 | 8 | 0:30:00 |
| 70B (Llama-3.3-70B-Instruct, etc.) | 256G | 4 | gpu80 | 8 | 0:30:00 |

**Detection logic:**
1. Parse model name from experiment_summary.md Resources → Models section
2. Look for size indicator in model name: "1B", "3B", "8B", "70B"
3. Apply corresponding resource configuration
4. Default to 1B settings if model size cannot be determined

## SLURM Script Template

Generate a SLURM script for each evaluation with model-appropriate resources.

**Key principle:** Extract dataset path, prompt, and system prompt from `setup_finetune.yaml` (for fine-tuned models) or experiment_summary.md (for base models) at scaffolding time, then pass them directly to inspect eval.

```bash
#!/bin/bash
#SBATCH --job-name=eval-{task_name}-{run_id}
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_from_model_size}
#SBATCH --mem={mem_from_model_size}
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:{gpus_from_model_size}
{optional: #SBATCH --account={account}}
{if 3B or larger: #SBATCH --constraint=gpu80}

# Load environment
module load anaconda3/2025.6
conda activate {conda_env}

# Model path
{if fine-tuned:}
MODEL_PATH="{output_dir_base}/ck-out-{run_name}/epoch_{N}"
{if base model:}
MODEL_PATH="{base_model_path}"

# Dataset and prompt configuration
# (extracted from setup_finetune.yaml or experiment_summary.md at scaffolding time)
DATA_PATH="{data_path}"
PROMPT="{prompt}"
SYSTEM_PROMPT="{system_prompt}"

# Run inspect-ai evaluation
cd {experiment_dir}/{run_dir}/eval

inspect eval {task_script_path}@{task_name} \\
  --model hf/local \\
  -M model_path="$MODEL_PATH" \\
  -T data_path="$DATA_PATH" \\
  -T prompt="$PROMPT" \\
  -T system_prompt="$SYSTEM_PROMPT" \\
  --log-dir ./logs \\
  --log-level info

echo "Evaluation complete"
```

## Script Configuration

### SLURM Parameters

- **Time**: Default to 30 minutes (adjust based on experiment estimates if available)
- **GPUs/Memory/CPUs**: Set based on model size (see Model-Aware Resource Allocation table above)
- **Constraint**: gpu80 required for 3B+ models
- **Account**: Use from claude.local.md if specified

### Model Paths

- **Fine-tuned**: `{output_dir_base}/ck-out-{run_name}/epoch_{N}`
- **Base model**: Original model path from experiment_summary.md

### Task Parameters (passed directly to inspect eval)

- `data_path`: Full path to dataset file (e.g., `/path/to/words_5L_80P_1000.json`)
- `prompt`: The prompt template used during training (e.g., `"{input}"`)
- `system_prompt`: System message used during training (often empty string)

**Extracting values for fine-tuned models:**
Read `setup_finetune.yaml` and compute: `data_path` = `input_dir_base` + `dataset_label` + `dataset_ext`

**Extracting values for base models:**
Use values from experiment_summary.md Configuration section.

### Output Location

- **Log directory**: `{run_dir}/eval/logs/`
- **SLURM output**: `{run_dir}/eval/slurm-{job_id}.out`

## Directory Structure Creation

Create eval directories as needed:

```bash
# For each run directory
mkdir -p {experiment_dir}/{run_dir}/eval
mkdir -p {experiment_dir}/{run_dir}/eval/logs

# Write SLURM script
cat > {experiment_dir}/{run_dir}/eval/{task_name}_epoch{N}.slurm << 'EOF'
{script content}
EOF

chmod +x {experiment_dir}/{run_dir}/eval/{task_name}_epoch{N}.slurm
```

## Important Notes

- Evaluation scripts should not be submitted until fine-tuning completes
- Model paths reference fine-tuning output directories that don't exist yet (created during training)
- System prompt consistency between training and evaluation is critical
- Evaluation logs will be written to `{run_dir}/eval/logs/` subdirectories
- **Base/control models**: Values are extracted from experiment_summary.md and baked directly into SLURM scripts
  - No separate config file needed - all parameters are hardcoded at scaffolding time
  - Uses same dataset/prompt/system_prompt as fine-tuned runs for fair comparison

# Scaffold Inspect

You help users automatically set up inspect-ai evaluation configurations for all runs in a designed experiment.

## Your Task

Read an `experiment_summary.md` file and generate all the necessary inspect-ai files (inspect.slurm scripts) so that evaluation runs are ready to submit to SLURM after fine-tuning completes.

## Workflow

1. **Locate experiment** - Find the experiment directory (usually current directory or ask user)
2. **Read experiment_summary.md** - Parse the experiment plan to extract evaluation configuration
3. **Read claude.local.md** - Get environment-specific settings (conda env, etc.)
4. **Verify inspect-ai tasks exist** - Check if evaluation task scripts are available
5. **For each run and evaluation combination:**
   - Generate `inspect.slurm` script for that run/epoch
   - Configure model paths, task parameters, output locations
   - Verify configuration
6. **Create scaffold log** - Document all actions taken in `scaffold-inspect.log`
7. **Report summary** - Show user what was created and any issues

## Finding the Experiment

**If user runs skill without arguments:**
- Check if current directory contains `experiment_summary.md`
- If not, ask user for the experiment directory path

**If user provides a path:**
- Use that path as the experiment directory

## Parsing experiment_summary.md

Extract the following information:

### Required Information

1. **Experiment name** - From the title (line 1)
2. **Experiment directory** - From Quick Reference → Paths → Experiment
3. **All runs table** - Extract run names and their configurations
4. **Model paths** - From Resources → Models
5. **Evaluation tasks** - From Resources → Evaluation Tasks table
6. **Evaluation plan** - From Evaluation Plan section:
   - Which epochs to evaluate
   - Which runs get which evaluations
   - Evaluation datasets (if different from training)
7. **System prompt** - From Configuration section (must match training)
8. **Output directory base** - Where fine-tuned models will be saved

### Parsing the "Evaluation Tasks" Table

```markdown
| Task Name | Script | Dataset | Description |
|-----------|--------|---------|-------------|
| capitalization | `path/to/cap_task.py` | `path/to/test.json` | Tests capitalization |
```

Extract:
- Task name (for naming evaluation outputs)
- Script path (inspect-ai task file)
- Dataset path (if specified and different from training)
- Description (for documentation)

### Parsing the "Evaluation Plan" Section

Determine:
- **Epochs to evaluate**: "last", "all", or specific list (e.g., "0,2")
- **Evaluation matrix**: Which runs evaluate on which tasks
- **Base model evaluations**: Control runs that need evaluation

## Reading claude.local.md

Extract environment-specific settings:
- `conda_env` - Which conda environment to use
- `scratch_dir` - User's scratch directory
- `account` - SLURM account to use (OPTIONAL)

## Verifying Inspect-AI Tasks

For each evaluation task in the experiment:

1. **Check if task script exists:**
   ```bash
   ls {task_script_path}
   ```

2. **If task doesn't exist:**
   - Note in log that task needs to be created
   - Suggest running `create-inspect-task` skill first
   - Continue with other tasks (don't fail completely)

3. **Verify task is compatible with experiment:**
   - Can it accept `config_dir` parameter? (for fine-tuned models)
   - Can it accept `dataset_path` parameter? (for base models)
   - Check docstring/parameters if accessible

## Generating Inspect SLURM Scripts

For each evaluation to be performed, generate an `inspect.slurm` script.

### Evaluation Naming Convention

Organize evaluations within run directories:

**For fine-tuned models:**
```
{experiment_dir}/{run_dir}/
├── finetune.slurm
├── finetune.yaml
├── setup_finetune.yaml
└── eval/
    ├── {task_name}_epoch0.slurm
    ├── {task_name}_epoch1.slurm
    └── ...
```

**For base models (controls):**
```
{experiment_dir}/{run_dir}_base/
└── eval/
    └── {task_name}_base.slurm
```

### SLURM Script Template

Generate a SLURM script for each evaluation:

```bash
#!/bin/bash
#SBATCH --job-name=eval-{task_name}-{run_id}
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --gpus=1
{optional: #SBATCH --account={account}}
{optional: #SBATCH --constraint=gpu80}

# Load environment
module load anaconda3/2025.6
conda activate {conda_env}

# Set model and config paths
{if fine-tuned:}
MODEL_PATH="{output_dir_base}/ck-out-{run_name}/epoch_{N}"
CONFIG_DIR="$MODEL_PATH"
{if base model:}
MODEL_PATH="{base_model_path}"
CONFIG_DIR=""

# Run inspect-ai evaluation
cd {experiment_dir}/{run_dir}/eval

{if fine-tuned with config_dir:}
inspect eval {task_script_path} \\
  --model hf/local \\
  -M model_path="$MODEL_PATH" \\
  -T config_dir="$CONFIG_DIR" \\
  --log-dir ./logs \\
  --log-level info

{if base model or direct dataset path:}
inspect eval {task_script_path} \\
  --model hf/local \\
  -M model_path="$MODEL_PATH" \\
  -T dataset_path="{eval_dataset_path}" \\
  -T system_prompt="{system_prompt}" \\
  --log-dir ./logs \\
  --log-level info

echo "Evaluation complete"
```

### Script Configuration

**SLURM parameters:**
- Time: Default to 30 minutes (adjust based on experiment estimates if available)
- GPUs: 1 (evaluation is typically single-GPU)
- Memory: 32G (adjust based on model size if known)
- Account/constraint: Use from claude.local.md if specified

**Model paths:**
- Fine-tuned: `{output_dir_base}/ck-out-{run_name}/epoch_{N}`
- Base model: Original model path from experiment_summary.md

**Task parameters:**
- `config_dir`: For fine-tuned models (reads from setup_finetune.yaml)
- `dataset_path`: For base models or when explicit path needed
- `system_prompt`: Must match training configuration
- `temperature`: Typically 0.0 (may be task-specific)

**Output location:**
- Log directory: `{run_dir}/eval/logs/`
- SLURM output: `{run_dir}/eval/slurm-{job_id}.out`

## Handling Different Evaluation Scenarios

### Scenario 1: Fine-tuned Model with Config Integration

Task supports `config_dir` parameter:
```bash
inspect eval cap_task.py \\
  --model hf/local \\
  -M model_path="/path/to/epoch_0" \\
  -T config_dir="/path/to/epoch_0"
```

The task reads dataset path and system prompt from `../setup_finetune.yaml`

### Scenario 2: Base Model Evaluation

No config_dir, explicit parameters:
```bash
inspect eval cap_task.py \\
  --model hf/local \\
  -M model_path="/path/to/base/model" \\
  -T dataset_path="/path/to/test_data.json" \\
  -T system_prompt=""
```

### Scenario 3: Custom Evaluation Dataset

Fine-tuned model but different eval dataset:
```bash
inspect eval cap_task.py \\
  --model hf/local \\
  -M model_path="/path/to/epoch_0" \\
  -T dataset_path="/path/to/generalization_test.json" \\
  -T system_prompt="{from_training_config}"
```

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

## Error Handling

**If experiment_summary.md not found:**
- Ask user for experiment directory path
- Verify file exists before proceeding

**If evaluation task information missing:**
- Report what's missing (task script path, dataset, etc.)
- Ask user to update experiment_summary.md
- Don't proceed without complete information

**If inspect-ai task script doesn't exist:**
- Log warning for that task
- Continue with other tasks
- Note in summary that some tasks need creation
- Suggest running `create-inspect-task` skill

**If unclear which evaluation approach to use:**
- Check if task file has `config_dir` parameter (preferred for experiments)
- Fall back to `dataset_path` + `system_prompt` approach
- Log the decision

## Logging

Create a detailed log file at `{experiment_dir}/scaffold-inspect.log`:

### Log Format

```
[YYYY-MM-DD HH:MM:SS] ACTION: Description
Details: {specifics}
Result: {outcome}

```

### What to Log

- Experiment discovery and validation
- Parsing experiment_summary.md evaluation sections
- Verification of inspect-ai task scripts
- Evaluation matrix analysis (which runs, which epochs, which tasks)
- Directory creation
- SLURM script generation for each evaluation
- Any errors or warnings
- Final summary of created evaluation configs

### Example Log Entries

```
[2025-10-24 17:00:00] DISCOVER_EXPERIMENT: Found experiment
Details: /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/experiment_summary.md
Result: Successfully read experiment plan (8 runs, 1 evaluation task)

[2025-10-24 17:00:05] PARSE_EVAL_TASKS: Extracting evaluation configuration
Details: Found 1 task: capitalization (cap_task.py)
Result: Task script path verified: /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/tasks/capitalization/cap_task.py

[2025-10-24 17:00:10] PARSE_EVAL_PLAN: Determining evaluation matrix
Details: Evaluate last epoch only, all runs on all tasks
Result: Will generate 8 evaluations (8 runs × 1 task × 1 epoch)

[2025-10-24 17:00:15] VERIFY_TASK: capitalization task
Command: ls /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/tasks/capitalization/cap_task.py
Result: Task script exists and is accessible
Note: Task supports config_dir parameter for experiment integration

[2025-10-24 17:00:20] CREATE_EVAL_DIR: rank8_lr1e-5/eval
Details: mkdir -p rank8_lr1e-5/eval/logs
Result: Directory created successfully

[2025-10-24 17:00:25] GENERATE_SLURM: rank8_lr1e-5/eval/capitalization_epoch0.slurm
Details: Fine-tuned model evaluation with config_dir integration
Model path: /scratch/gpfs/MSALGANIK/niznik/ck-outputs/ck-out-rank8_lr1e-5/epoch_0
Result: SLURM script created (45 lines)

[2025-10-24 17:01:30] COMPLETE: All evaluation configs generated
Summary: 8 evaluation scripts created successfully, 0 failures
Next: User can proceed with run-inspect skill after fine-tuning completes
```

## Output Summary

After completing all evaluation configurations, provide a summary:

```markdown
## Scaffold Inspect Complete

Successfully created 8 evaluation configurations in:
`/scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/`

### Created Evaluations

**Fine-tuned runs (8 evaluations):**
✓ rank8_lr1e-5/eval/capitalization_epoch0.slurm
✓ rank8_lr5e-5/eval/capitalization_epoch0.slurm
✓ rank16_lr1e-5/eval/capitalization_epoch0.slurm
✓ rank16_lr5e-5/eval/capitalization_epoch0.slurm
✓ rank32_lr1e-5/eval/capitalization_epoch0.slurm
✓ rank32_lr5e-5/eval/capitalization_epoch0.slurm
✓ rank64_lr1e-5/eval/capitalization_epoch0.slurm
✓ rank64_lr5e-5/eval/capitalization_epoch0.slurm

Each evaluation directory contains:
- {task_name}_epoch{N}.slurm (SLURM script)
- logs/ (for inspect-ai output)

### Evaluation Tasks

✓ **capitalization**: `/path/to/cap_task.py`
  - Dataset: Reads from fine-tuning config
  - System prompt: Matches training configuration
  - Epochs evaluated: Last epoch only (epoch 0)

### Next Steps

1. **Fine-tune models first:**
   Run `run-torchtune` skill to train all models

2. **After fine-tuning completes:**
   Run `run-inspect` skill to execute all evaluations

3. **Manual evaluation submission (alternative):**
   ```bash
   cd /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22
   # After fine-tuning completes for a run:
   cd rank8_lr1e-5/eval
   sbatch capitalization_epoch0.slurm
   ```

4. **View evaluation results:**
   ```bash
   inspect view --port=$(get_free_port)
   ```

See `scaffold-inspect.log` for detailed creation log.
```

## Validation Before Completion

Before reporting success, verify:
- ✓ All eval directories created
- ✓ Each evaluation has corresponding SLURM script
- ✓ Scripts reference correct model paths
- ✓ Scripts reference correct task scripts
- ✓ System prompts match training configuration
- ✓ Log directory paths are correct
- ✓ No errors in log
- ✓ Log file created

## Important Notes

- This skill generates evaluation configs for both fine-tuned and base models
- Evaluation scripts should not be submitted until fine-tuning completes
- System prompt consistency between training and evaluation is critical
- Model paths reference fine-tuning output directories that don't exist yet (created during training)
- inspect-ai task scripts must exist before scaffolding (or note as prerequisite)
- Base model evaluations use original model paths, not fine-tuned checkpoints
- This skill is typically called by `scaffold-experiment` orchestrator but can be run standalone
- Evaluation logs will be written to `{run_dir}/eval/logs/` subdirectories

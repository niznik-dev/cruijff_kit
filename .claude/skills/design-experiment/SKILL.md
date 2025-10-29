# Design Experiment

You help users plan experiments for fine-tuning and evaluating LLMs. Create a self-documenting plan that specifies all runs, verifies resources, estimates compute requirements, and documents the complete workflow from training through evaluation.

## Your Task

Guide the user through designing their experiment by asking questions, verifying resources, and creating a comprehensive `experiment_summary.md` file that documents the plan and tracks execution status.

## Workflow Tools

This skill documents a complete experimental workflow that uses:

1. **Model preparation: torchtune (current)**
   - Used by: `scaffold-torchtune` and `run-torchtune` skills
   - Generates: `finetune.yaml`, `finetune.slurm`
   - Produces: Model checkpoints in `output_dir_base`

2. **Evaluation: inspect-ai (current)**
   - Used by: `scaffold-inspect` and `run-inspect` skills
   - Generates: `inspect.slurm` and/or inspect task scripts
   - Produces: Evaluation logs (`.eval` files)

3. **Analysis: (future)**
   - Used by: `analyze-experiment` skill (planned)
   - Produces: Comparison tables, plots, reports

## Workflow

1. **Determine experiment type and location** - Auto-detect sanity_check vs experiment and set base directory
2. **Understand the experiment** - What variables are being tested? What's the scientific question?
3. **Confirm tool choices** - Ask which preparation and evaluation tools to use (currently only torchtune and inspect-ai)
4. **Verify resources** - Do models, datasets, and eval scripts exist? (log all checks)
5. **Plan evaluation** - What metrics, which epochs, what eval datasets/tasks?
6. **Establish naming** - Help choose a clear, descriptive name for the experiment
7. **Estimate resources** - Calculate time and disk space for BOTH training and evaluation (log all calculations)
8. **Create summary** - Write `experiment_summary.md` with complete plan including tool choices and evaluation workflow
9. **Create log** - Write `design-experiment.log` with all verification steps and decisions
10. **Get approval** - Review with user, adjust if needed

## Determining Experiment Location

**Auto-detect based on current working directory:**

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

**Directory structure:**
- **Experiments** (research tasks): `/scratch/gpfs/MSALGANIK/niznik/ck-experiments/{experiment_name}/`
- **Sanity checks** (workflow validation): `/scratch/gpfs/MSALGANIK/niznik/ck-sanity-checks/{sanity_check_name}/`

**Outputs are automatically grouped:**
- Output directory: `/scratch/gpfs/MSALGANIK/niznik/ck-outputs/{experiment_or_sanity_check_name}/ck-out-{run_name}/`

**When logging:**
- Log the detected experiment type (sanity_check vs experiment)
- Log the full path where the experiment will be created
- Note in experiment_summary.md that outputs will be grouped under the same name in ck-outputs/

## Logging

**IMPORTANT:** Create a detailed log file at `{experiment_name}/design-experiment.log` that records all verification steps, calculations, and decisions made during planning.

### Log Format

```
[YYYY-MM-DD HH:MM:SS] ACTION: Description
Command: {actual_command_run}
Result: {output_or_outcome}
Explanation: {why_this_was_done}

```

### What to Log

**DO log:**
- ✓ Resource verification commands (ls, du, df)
- ✓ Prior run searches and data extraction (find, grep)
- ✓ Calculations (time estimates, batch sizes, disk space)
- ✓ Decisions made (naming choices, recipe selection, configuration)
- ✓ File creation (experiment_summary.md, directories)

**DON'T log:**
- ✗ Job status checks (squeue, sacct)
- ✗ Simple read operations that don't affect the plan

### Example Log Entries

```
[2025-10-22 14:23:15] VERIFY_MODEL: Checking Llama-3.2-1B-Instruct
Command: ls /scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-1B-Instruct
Result: Directory exists with 15 files (config.json, model.safetensors, etc.)
Explanation: Verifying base model exists before creating experiment plan

[2025-10-22 14:23:42] VERIFY_DATASET: Checking capitalization dataset
Command: ls -lh /scratch/gpfs/MSALGANIK/niznik/GitHub/cruijff_kit/experiments/capitalization/input/words_8L_80P_10000.json
Result: File exists, 655KB
Explanation: Verifying training dataset exists and checking size

[2025-10-22 14:24:01] SEARCH_PRIOR_RUNS: Looking for similar experiments
Command: find /scratch/gpfs/MSALGANIK/mjs3 -name "slurm-*.out" -path "*/ck-out-*" -size +100k | head -5
Result: Found 3 similar runs: ck-out-happy-narwhal, ck-out-bright-horizon, ck-out-calm-dolphin
Explanation: Searching for prior SLURM logs to extract training speed data for estimates

[2025-10-22 14:24:15] EXTRACT_SPEED: Analyzing prior run for training speed
Command: grep -E "[0-9.]+it/s" /scratch/gpfs/MSALGANIK/mjs3/ck-out-happy-narwhal/slurm-12345.out | tail -20
Result: Average speed after warmup: 4.34 it/s
Explanation: Extracting iteration speed from similar prior run with same model and batch size

[2025-10-22 14:24:30] CALCULATE_TIME: Training time estimate
Input: 8000 samples, batch_size=4, speed=4.34 it/s, epochs=2
Calculation: steps_per_epoch = 8000/4 = 2000, time_per_epoch = 2000/4.34 ≈ 461s ≈ 8min
Result: Estimated 16 minutes total (8 min × 2 epochs)
Explanation: Calculated training time based on actual iteration speed from prior run

[2025-10-22 14:25:00] CHECK_DISK: Verifying available disk space
Command: df -h /scratch/gpfs/MSALGANIK/niznik
Result: 2.1T available
Explanation: Ensuring sufficient space for ~40 GiB of checkpoints

[2025-10-22 14:25:30] DECIDE_NAMING: Experiment name chosen
Decision: cap_8L_lora_comparison_2025-10-22
Reasoning: Capitalization task (cap), 8-letter words (8L), comparing LoRA ranks, dated
Explanation: User confirmed this naming follows convention and is descriptive

[2025-10-22 14:26:00] CREATE_SUMMARY: Writing experiment plan
Command: Created /scratch/gpfs/MSALGANIK/niznik/cap_8L_lora_comparison_2025-10-22/experiment_summary.md
Result: File created with 4 runs (2 fine-tuned × 2 ranks + 2 controls)
Explanation: Comprehensive experiment plan with all configurations and status tracking

[2025-10-22 14:26:15] COMPLETE: Experiment design finished
Status: Ready for user review and approval
Next Steps: User should review experiment_summary.md, then proceed with manual setup
Explanation: Planning phase complete, documented in summary and this log
```

### Purpose of the Log

The log enables:
1. **Debugging:** If estimates are wrong, check what commands were run and what data was used
2. **Reproducibility:** Another person (or Claude) can understand exactly what was done
3. **Improvement:** Review logs to identify better approaches or missing steps
4. **Auditing:** Verify that all resources were properly checked before committing to the experiment

## Questions to Ask

### 1. Tool Selection

**Which tools will you use for this experiment?**

**Model preparation:**
- torchtune (currently the only option)
- *Future:* Other fine-tuning frameworks may be supported

**Evaluation:**
- inspect-ai (currently the only option)
- *Future:* Other evaluation frameworks may be supported

**Note:** While these are currently the only options, explicitly confirming and documenting tool choices now will make it easier to support multiple tools in future iterations.

### 2. Experiment Design

**What do you want to test?**
- Different model sizes? (e.g., 1B, 3B, 8B)
- Different LoRA ranks?
- Different datasets or data sizes?
- Different hyperparameters?
- Combinations of the above?

**Should we include base model controls?**
- Controls evaluate base models without fine-tuning to measure the effect of fine-tuning

### 3. Resources (use `claude.local.md` for defaults)

**Models:**
- Which models? (check `{models_dir}` from `claude.local.md`)
- Verify they exist: `ls {models_dir}/{model_name}`

**Dataset:**
- Which dataset and where is it located?
- Verify it exists and has required splits (train/validation/test)
- Check size: `ls -lh {dataset_path}`

**Evaluation tasks (inspect-ai):**
- Which evaluation tasks/benchmarks?
- For each task: name, inspect-ai task script path, dataset path (if different from training), description
- Does the task already exist or need to be created? (use `create-inspect-task` skill if needed)
- Verify task scripts exist: `ls {eval_script_path}`

### 4. Training Configuration

**Basic settings:**
- How many epochs? (default: 1-2)
- How many GPUs per job? (default: 1)
- Should validation run during training? (default: no)
- System prompt for training and evaluation? (default: "")

**Advanced settings (calculate from prior runs if available):**
- Batch sizes - estimate from GPU memory usage in prior runs
- Dataset packing - enabled by default, affects batch size
- For help estimating: check `{scratch_dir}/*/slurm-*.out` for similar runs

### 5. Evaluation Strategy (inspect-ai)

**Evaluation tool:** inspect-ai will be used for model evaluation after fine-tuning.

**Which epochs to evaluate?**
- Last epoch only (default, most efficient)
- All epochs (compare training progression)
- Specific epochs (e.g., epoch 0 and final)
- Best by validation loss (requires validation during training)

**Which runs get which evaluations?**
- All runs on all tasks (typical)
- Selective (e.g., only large models on expensive evals)
- If selective, create evaluation matrix

**Evaluation datasets:**
- Same as training dataset (typical for overfitting checks)
- Different test set (typical for generalization evaluation)
- Multiple evaluation datasets (comprehensive assessment)

**Important:** Base models evaluate once per task (no epoch suffix), fine-tuned models evaluate per epoch.

**Evaluation configuration:**
- System prompt must match training for consistency
- Temperature typically 0.0 for deterministic evaluation
- Scorer selection (exact match, includes, model-graded, etc.)

### 6. Experiment Naming

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

## Resource Verification

Before finalizing the plan, verify:

1. **Models exist:**
   ```bash
   ls {models_dir}/{model_name}
   ```

2. **Dataset exists with splits:**
   ```bash
   ls -lh {dataset_path}
   # Check for train/validation/test splits
   ```

3. **Evaluation scripts exist:**
   ```bash
   ls {eval_script_path}
   ```

4. **Disk space available:**
   ```bash
   df -h {scratch_dir}
   ```

**If resources missing:**
- Model: Suggest downloading with appropriate tool
- Dataset: Offer to help create it (if known task like capitalization)
- Eval script: Note as prerequisite, proceed with plan anyway
- Disk space: Warn user, suggest cleanup or alternative location

## Estimation Guidelines

### Time Estimates

**From prior runs (preferred):**
1. Find similar runs in `{scratch_dir}/ck-out-*/`
2. Extract iteration speed from SLURM logs: `grep -E "it/s" {log_path}`
3. Calculate: `time = (samples / batch_size / speed) * epochs`

**If no prior runs:**
- Use conservative estimates based on model size and GPU type
- Clearly mark as "preliminary - verify with test run"
- Typical ranges:
  - 1B models: 30-60 min/epoch
  - 3B models: 1-2 hours/epoch
  - 7B+ models: 3-5 hours/epoch

**Evaluation time:**
- Inference-only: ~2-3x faster than training
- Typically 1-5 minutes per evaluation
- Multiply by (num_runs × num_tasks × num_epochs)

### Disk Space Estimates

**From prior runs:**
```bash
du -sh {prior_run_dir}/epoch_0
```

**Typical checkpoint sizes:**
- 1B: ~2-3 GiB per epoch
- 3B: ~6-7 GiB per epoch
- 7B: ~14-20 GiB per epoch

**Total:** `num_runs × num_epochs × checkpoint_size + 20% buffer`

### Batch Size Guidance

**From prior runs:**
1. Find GPU memory usage: `grep "GPU peak memory" {log_path}`
2. Calculate headroom: `available_memory / peak_memory`
3. Scale conservatively: `max_batch = headroom × 0.7`

**If dataset packing enabled (default):**
- Reduces effective batch size by 2-4x
- Start conservative: batch_size=4 (1B), batch_size=2 (3B)

**No prior data:**
- 80GB GPU: batch_size=4-8 (1B), 2-4 (3B)
- 40GB GPU: batch_size=2-4 (1B), 2 (3B)
- Start small, monitor first run, adjust

## Output: experiment_summary.md

Create a comprehensive document in `{experiment_name}/experiment_summary.md` with:

### Required Sections

1. **Overview** - Experiment type, total runs, scientific question, created date
2. **Tools** - Which preparation and evaluation tools are used
3. **Variables** - Table of factors and levels being tested
4. **All Runs** - Complete table with run names, configurations, estimated time
5. **Resources** - Verified paths to models, datasets, eval scripts
6. **Evaluation Plan** - Which tasks, which runs, which epochs
7. **Configuration** - Recipe, epochs, batch sizes, hyperparameters, system prompt
8. **Compute Estimates** - Training time, eval time, disk space, GPU hours
9. **Naming Conventions** - How runs are named and organized
10. **Status Tracking** - Embedded status for each run and evaluation

### Status Tracking Format

Include a status section that can be updated as work progresses:

```markdown
## Status

### Fine-tuning
| Run Name | Status | Job ID | Started | Completed | Notes |
|----------|--------|--------|---------|-----------|-------|
| {run1} | pending | - | - | - | - |
| {run2} | pending | - | - | - | - |

Status values: `pending`, `running`, `completed`, `failed`

### Evaluations
| Run Name | Task | Epoch | Status | Job ID | Completed | Notes |
|----------|------|-------|--------|--------|-----------|-------|
| {run1} | {task1} | 0 | pending | - | - | - |
| {run1} | {task1} | 1 | pending | - | - | - |
| {run1_base} | {task1} | - | pending | - | - | Base model |
```

### Quick Reference Section

Include experiment-specific quick reference:
```markdown
## Quick Reference

**Paths:**
- Experiment: `{full_path_to_experiment_dir}`
- Models: `{models_dir}/{model_names}`
- Dataset: `{dataset_path}`

**Common Commands:**
- Check jobs: `squeue -u $USER`
- Monitor training: `tail -f {experiment_dir}/{run_name}/slurm-*.out`
- Check disk: `df -h {scratch_dir}`

**Next Steps:**
1. [Manual step or placeholder for future skill]
2. Generate configs for each run
3. Submit fine-tuning jobs
4. Monitor progress
5. Submit evaluation jobs
6. Analyze results
```

## Template Structure

Use this outline (adapt details based on user's experiment):

```markdown
# Experiment: {experiment_name}

## Overview
- **Type:** {design_type} (e.g., 2×2 factorial, parameter sweep, ablation)
- **Total Runs:** {count} ({finetuned} fine-tuned + {control} controls)
- **Scientific Question:** {research_question}
- **Created:** {timestamp}

## Tools

This experiment uses the following tools:

- **Model Preparation:** torchtune
  - *Current status:* Only option available
  - *Purpose:* Fine-tuning LLMs with LoRA
  - *Used by:* `scaffold-torchtune` and `run-torchtune` skills

- **Evaluation:** inspect-ai
  - *Current status:* Only option available
  - *Purpose:* Evaluating LLMs on custom tasks
  - *Used by:* `scaffold-inspect` and `run-inspect` skills

*Note:* Future iterations of cruijff_kit may support additional preparation and evaluation frameworks. Tool choices are documented here to facilitate that transition.

## Variables
| Factor | Levels |
|--------|--------|
| {factor1} | {level1}, {level2} |

## All Runs
| Run Name | Model | LoRA Rank | Batch Size | Type | Est. Time |
|----------|-------|-----------|------------|------|-----------|
| ... | ... | ... | ... | ... | ... |

## Resources

### Models
- **Location:** `{models_dir}`
- **Models Used:**
  - {model1}: `{full_path}` ✓ verified
  - {model2}: `{full_path}` ✓ verified

### Dataset
- **Path:** `{dataset_path}` ✓ verified
- **Size:** {file_size}
- **Splits:** train ({count}), validation ({count}), test ({count})

### Evaluation Tasks
| Task Name | Script | Dataset | Description |
|-----------|--------|---------|-------------|
| {task1} | `{path}` ✓ | `{dataset}` ✓ | {desc} |

## Evaluation Plan

### Configuration
- **Epochs to evaluate:** {which_epochs}
- **Total evaluations:** {count}

### Evaluation Matrix
| Run Name | {task1} | {task2} | Notes |
|----------|---------|---------|-------|
| {run1} | ✓ epoch 0,1 | ✓ epoch 0,1 | All evals |
| {run1_base} | ✓ | - | Primary only |

## Configuration
- **Recipe:** `{recipe_path}`
- **GPUs:** {count}
- **Epochs:** {count}
- **Batch sizes:** {details}
- **Dataset packing:** {true/false} - {impact_note}
- **LoRA alpha:** Auto-set to 2 × rank
- **System prompt:** "{prompt}" (consistent across train and eval)
- **Validation during training:** {yes/no}

## Compute Estimates

### Training
- **{Model1}:** ~{time} per run × {count} runs = {total}
- **{Model2}:** ~{time} per run × {count} runs = {total}
- **Total GPU hours:** {hours}
- **Basis:** {from_prior_runs or estimated}

### Evaluation
- **Per evaluation:** ~{time}
- **Total evaluations:** {count}
- **Total eval time:** {time}

### Disk Space
- **Checkpoints:** ~{size} GiB
- **Available:** {size} GiB ✓

## Naming Conventions
- **Experiment:** `{experiment_name}`
- **Fine-tuned runs:** `{model_name}_{factors}_rank{N}`
- **Control runs:** `{model_name}_base`

## Status

### Fine-tuning
| Run Name | Status | Job ID | Started | Completed | Notes |
|----------|--------|--------|---------|-----------|-------|
| ... | pending | - | - | - | - |

### Evaluations
| Run Name | Task | Epoch | Status | Job ID | Completed | Notes |
|----------|------|-------|--------|--------|-----------|-------|
| ... | ... | ... | pending | - | - | - |

## Quick Reference

**Paths:**
- Experiment: `{full_path}`
- Models: `{models_dir}`
- Dataset: `{dataset_path}`

**Common Commands:**
- Check jobs: `squeue -u $USER`
- Monitor: `tail -f {path}/slurm-*.out`

**Next Steps:**
1. Run `scaffold-experiment` skill to generate all configs (torchtune + inspect-ai)
2. Run `run-experiment` skill to execute the full workflow:
   - Submits and monitors fine-tuning jobs (via `run-torchtune`)
   - Submits and monitors evaluation jobs (via `run-inspect`)
3. Run `analyze-experiment` skill to interpret results (planned)

## Notes
{Any additional notes, assumptions, or considerations}
```

## Validation Before Completion

Verify the plan is complete:
- ✓ All models verified
- ✓ Dataset verified with correct splits
- ✓ Evaluation scripts verified (or noted as prerequisites)
- ✓ Time estimates calculated or clearly marked as preliminary
- ✓ Disk space checked
- ✓ All run names follow convention
- ✓ Evaluation matrix is consistent
- ✓ Status tables initialized

## Next Steps

After creating `experiment_summary.md`:

1. **Ask user for approval:**
   - "I've created the experiment plan at `{path}/experiment_summary.md`. Please review it."
   - "Would you like me to proceed with scaffolding? I can run `scaffold-experiment` to generate all configs."

2. **Automated workflow (recommended):**
   - Run `scaffold-experiment` skill to generate:
     - Fine-tuning configs via `scaffold-torchtune` (finetune.yaml, finetune.slurm)
     - Evaluation configs via `scaffold-inspect` (inspect.slurm, task scripts)
   - Run `run-experiment` skill to execute:
     - Fine-tuning via `run-torchtune` (submit jobs, monitor progress)
     - Evaluation via `run-inspect` (submit jobs after training completes, monitor progress)
   - Run `analyze-experiment` skill to interpret results (planned)

3. **Manual workflow (if needed):**
   - User can manually create directories and configs
   - Follow the experiment plan as documented in experiment_summary.md

## Important Notes

- Use paths from `claude.local.md` for models, datasets, scratch directories
- Always verify resources exist before finalizing plan
- Be conservative with estimates if no prior run data available
- **System prompt must be consistent between training and evaluation** (critical for inspect-ai)
- Base models evaluate once (no epoch), fine-tuned models evaluate per epoch
- Status tables track both fine-tuning and evaluation progress
- Document which tool is used at each stage (torchtune for training, inspect-ai for evaluation)
- Evaluation datasets may differ from training datasets (document clearly)
- If inspect-ai task doesn't exist, note that `create-inspect-task` skill should be run first

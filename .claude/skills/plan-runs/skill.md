# Plan Runs

You are helping the user plan a series of runs that will fine-tune and evaluate LLMs. These runs are designed for science, so everything should be clear, well-documented, and computationally reproducible. 

When finished you should write a document called runs_plan.md that summaries the set of runs and runs_status.yaml that will be updated as the different steps in the pipelines are completed.

## Your Task

Guide the user through defining the structure of their experiments by asking questions, verifying resources, and clarifying their goals.

## Workflow Overview

Follow these steps sequentially to create a complete run plan:

1. **Gather Requirements** (Questions to Ask section)
   - **First**: Check if `claude.local.md` exists in repo root for user's environment paths
   - Understand what variables they want to test
   - Determine control conditions
   - Confirm resource and training configuration

2. **Verify Resources** (Required Ingredients section)
   - Use paths from `claude.local.md` (models_dir, scratch_dir, etc.) or ask user
   - Check that base models exist at specified paths
   - Verify dataset exists with proper splits
   - Confirm evaluation script is available and compatible

3. **Estimate Resources** (Resource Estimation sections)
   - Calculate time estimates based on prior run logs (or make educated estimates)
   - Determine appropriate batch sizes based on GPU memory
   - Compute total GPU hours needed

4. **Generate Plan Summary** (Run Summary section)
   - Create comprehensive runs table
   - Document all verified resources with paths
   - Show naming conventions and directory structure
   - Include all configuration details
   - Add Quick Reference section with experiment-specific paths and essential commands

5. **Get User Approval**
   - Present the complete plan
   - Confirm all details are correct
   - Address any final questions or adjustments

6. **Create Deliverables & Hand Off**
   - Write `runs_plan.md` to the run group directory
   - Initialize `runs_status.yaml` with all runs in `pending` state
   - Suggest next skill for directory setup and script generation

**Note:** If resources are missing or estimates can't be calculated, pause and work with the user to resolve before proceeding.

## Output Deliverables

At the end of this skill, you will create two files in the `{run_group_name}/` directory:

### 1. `runs_plan.md`
This is the complete plan document that describes all runs. It should contain:
- Run design overview (type, total count)
- Variables being tested (factors and levels)
- Table of all runs with configurations
- Verified resources (model paths, dataset paths, eval script paths)
- Configuration details (recipe, epochs, batch sizes, hyperparameters)
- Naming conventions
- Directory structure preview
- Estimated compute requirements (time, GPU hours)

**Purpose:** This file should be detailed enough that a later skill can parse it to automatically generate directory structures and SLURM scripts.

**Template:**
```markdown
# Run Plan: {run_group_name}

## Overview
- **Type**: {e.g., 2×2 Factorial Design}
- **Total Runs**: {total_count} ({finetune_count} fine-tuned + {control_count} controls)
- **Task**: {task_name}
- **Created**: {timestamp}

## Variables
| Factor | Levels |
|--------|--------|
| {factor1} | {levels} |
| {factor2} | {levels} |

## All Runs
| Run Name | Model | LoRA Rank | Batch Size | Type | Est. Time |
|----------|-------|-----------|------------|------|-----------|
| {name1} | {model} | {rank} | {batch_size} | Fine-tuned | {time} |
| ... | ... | ... | ... | ... | ... |

## Resources
### Models
- **Location**: {base_path}
- **Models Used**:
  - {model1}: `{full_path1}`
  - {model2}: `{full_path2}`

### Dataset
- **Path**: `{dataset_path}`
- **Size**: {file_size}
- **Splits**: train ({train_count}), validation ({val_count}), test ({test_count})

## Evaluations

### Evaluation Configuration
- **Epochs to evaluate**:
  - Fine-tuned models: {which_epochs} (e.g., "epoch_1 only", "all epochs (0, 1)", "best epoch by validation loss")
  - Base models: Single evaluation per task (no epochs - base models don't change)
- **Total evaluation jobs**:
  - Fine-tuned: {num_finetuned_runs} runs × {num_tasks} tasks × {num_epochs} epochs = {finetuned_eval_jobs} evaluations
  - Base models: {num_base_runs} runs × {num_tasks} tasks × 1 eval = {base_eval_jobs} evaluations
  - **Total: {total_eval_jobs} evaluations**

### Evaluation Tasks
| Task Name | Script | Dataset | Description |
|-----------|--------|---------|-------------|
| {task1_name} | `{task1_script}` | `{task1_dataset}` | {task1_description} |
| {task2_name} | `{task2_script}` | `{task2_dataset}` | {task2_description} |
| ... | ... | ... | ... |

### Evaluation Matrix
| Run Name | {task1_name} | {task2_name} | {task3_name} | Notes |
|----------|--------------|--------------|--------------|-------|
| {run1} | ✓ | ✓ | - | {notes} |
| {run2} | ✓ | - | - | {notes} |
| ... | ... | ... | ... | ... |

**Legend**: ✓ = will evaluate, - = skip

## Configuration
- **Recipe**: `{recipe_name}`
- **GPUs per job**: {gpu_count}
- **Epochs**: {epochs}
- **Dataset packing**: {True/False} - {explanation}
  - Packing density: {avg_examples_per_sequence} (from prior runs / estimated)
  - Impact on memory: ~{estimated_multiplier}x compared to unpacked
- **Batch sizes**: {batch_size_details} (adjusted for packing)
- **Gradient accumulation**: {steps} (effective batch size: {effective_batch})
- **LoRA alpha**: {alpha_value} (auto-set to 2 × rank)
- **Validation during training**: {yes/no}
- **System prompt**: "{system_prompt}" (used during both training and evaluation)

## Directory Structure
```
{run_group_name}/
  {run1_name}/
    setup_finetune.yaml
    finetune.yaml
    finetune.slurm
  {run2_name}/
    ...
  submit_all.sh
  runs_plan.md (this file)
  runs_status.yaml
```

## Compute Estimates

### Training Time Estimates
Based on actual prior runs from `{prior_run_directory}`:

**{Model1} models:**
- Prior run data: {samples} samples, batch_size={bs}, speed={it/s} it/s
- Steps per epoch: {samples}/{bs} = {steps} steps
- Time per epoch: ~{minutes} minutes (includes validation + checkpointing overhead)
- **{epochs} epochs × {num_runs} runs**: ~{total_min} minutes total (can run in parallel)
- **Sequential**: {num_runs} runs × {time_per_run} min = {sequential_time} minutes
- **GPU hours**: {gpu_hours} hours

**{Model2} models:**
- Prior run data: {samples} samples, batch_size={bs}, speed={it/s} it/s
- Steps per epoch: {samples}/{bs} = {steps} steps
- Time per epoch: ~{minutes} minutes (includes validation + checkpointing)
- **{epochs} epochs × {num_runs} runs**: ~{total_min} minutes total
- **GPU hours**: {gpu_hours} hours

**Total training compute:**
- **Parallel** (if {N} GPUs available): ~{max_job_time} minutes wall-clock time
- **Sequential** (1 GPU): ~{total_sequential} minutes wall-clock time
- **Total GPU hours**: {total_gpu_hours} GPU hours

### Evaluation Time Estimates
Evaluations are inference-only (no backprop), significantly faster than training:

**Per evaluation job estimate:**
- Test set: {test_samples} samples
- Inference speed: ~2-3x faster than training speed
- **{Model1}**: ~{eval_time_1} minutes per eval
- **{Model2}**: ~{eval_time_2} minutes per eval

**Total evaluation compute:**
- {breakdown_by_model_and_task}
- **Total**: {total_eval_time} minutes = {total_eval_hours} hours wall-clock time (sequential)
- **GPU hours**: {eval_gpu_hours} GPU hours

### Disk Space Estimates
Based on checkpoint sizes from prior runs:

**Per checkpoint:**
- {Model1}: {size_1} GiB per epoch
- {Model2}: {size_2} GiB per epoch

**Total checkpoint storage:**
- {Model1}: {num_runs} runs × {epochs} epochs × {size_1} GiB = {total_1} GiB
- {Model2}: {num_runs} runs × {epochs} epochs × {size_2} GiB = {total_2} GiB
- **Total**: ~{total_disk} GiB for all checkpoints

**Evaluation logs** (inspect-ai output):
- Estimated: ~10 MB per evaluation × {num_evals} evals = ~{eval_disk} MB

**Total disk space needed**: ~{grand_total_disk} GiB

### Wall-Clock Time Scenarios

**Scenario 1: Fully Parallel ({N} GPUs)**
- Training: {max_training_time} minutes (longest job)
- Evaluations: {max_eval_time} minutes (longest eval)
- **Total**: ~{total_parallel} minutes

**Scenario 2: Sequential (1 GPU)**
- Training: {sequential_training} minutes
- Evaluations: {sequential_evals} minutes
- **Total**: ~{total_sequential_hours} hours

**Scenario 3: Mixed ({M} GPUs)**
- Training: {batches} batches × {time_per_batch} min = {mixed_training} minutes
- Evaluations: {eval_batches} batches × {time_per_eval_batch} min = {mixed_evals} minutes
- **Total**: ~{total_mixed_hours} hours

### Summary
| Resource | Training | Evaluations | Total |
|----------|----------|-------------|-------|
| **GPU Hours** | {train_gpu_hrs} | {eval_gpu_hrs} | **{total_gpu_hrs}** |
| **Wall-Clock (Sequential)** | {train_seq_min} min | {eval_seq_min} min | **{total_seq_time}** |
| **Wall-Clock (Parallel, {N} GPUs)** | {train_par_min} min | {eval_par_min} min | **{total_par_min} min** |
| **Disk Space** | {train_disk} GiB | {eval_disk} GiB | **{total_disk} GiB** |

**Note**: Estimates based on actual SLURM logs from prior runs with identical configurations. Actual times may vary by ±10% depending on cluster load and GPU type.

## Quick Reference

**Paths:**
- **Experiment**: `{experiment_base_path}`
- **Models**: `{models_dir}/{model_names}`
- **Datasets**: `{dataset_paths}`
- **Eval scripts**: `{eval_script_paths}`

**Common Commands:**
- **Check jobs**: `squeue -u $USER`
- **Monitor log**: `tail -f slurm-{job_id}.out`
- **View results**: `inspect view --port=$(get_free_port)`
- **Check disk space**: `df -h {scratch_dir}`

**Workflow:**
1. Use `create-torchtune-config` skill to generate configs for all runs
2. Use `launch-runs` skill to submit fine-tuning jobs
3. Use `monitor-jobs` skill to track progress
4. Use `launch-runs` skill again to submit evaluation jobs after training completes
5. Use `summarize-experiments` skill to analyze results

## Notes
{Any additional notes, assumptions, or special considerations}
```

### 2. `runs_status.yaml`
This file tracks the progress of each run through the pipeline. It is automatically created and updated by the `update-run-status` skill.

**Purpose:** Provides a programmatically-accessible status tracker that allows recovery and restart if any part of the pipeline breaks.

**Template:**
```yaml
# Run Status Tracker
# Auto-generated and updated by update-run-status skill
# Last updated: {timestamp}

run_group: {run_group_name}
runs:
  {finetuned_run_name}:
    finetune:
      status: pending  # pending|submitted|running|completed|failed
      job_id: null
      output: null
      last_updated: null
    evaluations:
      # For fine-tuned runs: include epoch suffix
      {task1_name}_epoch_0:
        status: pending
        job_id: null
        output: null
        last_updated: null
      {task1_name}_epoch_1:
        status: pending
        job_id: null
        output: null
        last_updated: null
      {task2_name}_epoch_0:
        status: pending
        job_id: null
        output: null
        last_updated: null
  {base_model_name}:
    finetune:
      status: skipped  # Base models don't need fine-tuning
      job_id: null
      output: null
      last_updated: null
      note: "Base model - no fine-tuning required"
    evaluations:
      # For base models: NO epoch suffix (evaluated once)
      {task1_name}:
        status: pending
        job_id: null
        output: null
        last_updated: null
      {task2_name}:
        status: pending
        job_id: null
        output: null
        last_updated: null
```

**Note:** The initial `runs_status.yaml` will be created by the skill that sets up the directory structure, with all runs in `pending` state. The `update-run-status` skill will update it as jobs progress.

## Concrete Example

Here's a typical example to illustrate what we're planning:

**Note:** This example uses paths from Princeton's della cluster. Your environment will have different paths (from `claude.local.md` or user input).

### Scenario: 2×2 Factorial Design
**Research Question:** Does LoRA rank affect model performance across different model sizes?

**Variables:**
- Model size: 1B-Instruct, 3B-Instruct
- LoRA rank: 4, 64

**Design Matrix:**
```
           rank4    rank64
1B-Instruct  ✓        ✓
3B-Instruct  ✓        ✓
```

**Result:** 4 fine-tuning runs + 2 control runs = 6 total runs

**Directory Structure:**
```
cap_8L_llama32_lora_comparison_2025-10-18/
├── Llama-3.2-1B-Instruct_rank4/
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   └── finetune.slurm
├── Llama-3.2-1B-Instruct_rank64/
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   └── finetune.slurm
├── Llama-3.2-3B-Instruct_rank4/
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   └── finetune.slurm
├── Llama-3.2-3B-Instruct_rank64/
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   └── finetune.slurm
├── submit_all.sh
├── runs_plan.md
└── runs_status.yaml
```

**Key Configuration:**
- Task: Capitalization (8-letter words)
- Dataset: 10,000 words with train/val/test splits
- Recipe: `lora_finetune_single_device_val` (1 GPU, with validation)
- Epochs: 2
- Batch size: 16 (1B models), 8 (3B models)
- LoRA alpha: Auto-set to 2 × rank (8 or 128)

**Evaluations:**

| Task Name | Script | Dataset |
|-----------|--------|---------|
| capitalization_8L | tasks/capitalization/eval.py | Same as training |
| reasoning | tasks/reasoning/eval.py | tasks/reasoning/test.json |

**Evaluation Matrix:**

| Run Name | capitalization_8L | reasoning | Notes |
|----------|-------------------|-----------|-------|
| Llama-3.2-1B-Instruct_rank4 | ✓ | ✓ | All evals |
| Llama-3.2-1B-Instruct_rank64 | ✓ | ✓ | All evals |
| Llama-3.2-3B-Instruct_rank4 | ✓ | ✓ | All evals |
| Llama-3.2-3B-Instruct_rank64 | ✓ | ✓ | All evals |
| Llama-3.2-1B-Instruct_base | ✓ | - | Primary only |
| Llama-3.2-3B-Instruct_base | ✓ | ✓ | All evals |

**Estimated Time:**
- 1B models: ~20 minutes each
- 3B models: ~45 minutes each
- Total compute: ~2.5 GPU hours

This example shows how answering the questions in the next section translates into a concrete, executable plan.

## Questions to Ask

Guide the user through these questions in a natural conversation flow. Ask questions sequentially and clarify based on their responses.

### 1. Experiment Design (Required)

**What do you want to vary in these runs?**
- Different model sizes? (e.g., 1B-Instruct, 3B-Instruct, 8B-Instruct, 70B-Instruct)
- Different LoRA ranks?
- Different datasets?
- Different hyperparameters?
- Any combination of the above?

**Should we include control runs?**
- Control runs evaluate base models without fine-tuning
- They help measure the effect of fine-tuning vs. base model performance
- Recommended for scientific experiments

### 2. Resource Configuration (Has defaults, ask if not specified)

**How many GPUs per job?**
- Default: 1 GPU → uses `lora_finetune_single_device` recipe
- >1 GPU → uses `lora_finetune_distributed_v1` recipe

**What batch size should we use?**
- Should be calculated from prior runs and GPU memory
- See "Batch Size Estimation" section below for methodology
- If no prior runs available, start conservative and ask user's GPU type

### 3. Training Options (Has defaults, confirm with user)

**Do you want to track validation loss during training?**
- Default: No (simpler, faster)
- If yes: Requires `_val` recipe variant (e.g., `lora_finetune_single_device_val`)
- Validation requires dataset with validation split

**Should dataset packing be enabled?**
- Default: Yes (packed=True) for efficiency
- Packing concatenates multiple examples into sequences up to max_seq_len
- More efficient (no padding waste) but requires smaller batch sizes
- If unsure: Start with packed=True and batch_size=4 (conservative)
- If have prior runs with packing disabled: Need to recalculate batch sizes (see Batch Size Estimation section)
- Consider packed=False if:
  - Examples are already near max_seq_len (little benefit from packing)
  - Debugging training issues (simpler to reason about)
  - Have well-tuned batch sizes from unpacked runs

**How many epochs?**
- Default suggestion: 1-3 epochs (depends on task complexity)
- Ask user if they have preference based on prior work

**What system prompt should be used during training and evaluation?**
- System prompts guide the model's behavior and should be consistent across training and evaluation
- Default: Empty string "" (no system prompt)
- Common examples:
  - "" (empty) - Let the model use its default instruction-following behavior
  - "You are a helpful assistant." - General assistance
  - "You are an expert at capitalizing words correctly." - Task-specific guidance
- IMPORTANT: The same system prompt used during training MUST be used during evaluation for fair comparison
- Ask user to specify the exact system prompt they want to use
- Document this clearly in the runs_plan.md as it's critical for reproducibility

**LoRA hyperparameters:**
- LoRA rank: User must specify (or varies as experimental factor)
- LoRA alpha: Auto-set to 2 × rank by `setup_finetune.py` (no need to ask)

### 4. Evaluation Strategy (Required)

**What evaluation tasks do you want to run?**
- Ask user to list all evaluation tasks/benchmarks they want to use
- For each evaluation task, gather:
  - Task name/identifier (e.g., "capitalization_8L", "reasoning", "robustness")
  - Script path (e.g., "tasks/capitalization/eval.py")
  - Dataset path (if different from training data)
  - Brief description of what it evaluates

**Which epochs should be evaluated?**
- Options:
  - **Last epoch only** (most common): Evaluate only the final checkpoint
  - **First epoch only**: Evaluate early training checkpoint
  - **All epochs**: Compare performance across training progression
  - **Specific epochs**: Choose particular checkpoints (e.g., epochs 0 and 2)
  - **Best epoch**: Based on validation loss (requires validation during training)
- Default suggestion: Last epoch only (most efficient)
- Note: Evaluating multiple epochs increases total evaluation jobs proportionally
  - Example: 12 fine-tuned runs × 3 tasks × 2 epochs = 72 evaluation jobs

**IMPORTANT - Base Model Evaluations:**
- Base models (control runs) should be evaluated **once per task** (no epochs)
- Base models don't have training epochs since they're not fine-tuned
- In `runs_status.yaml`, base model evaluation tasks should NOT have epoch suffixes
  - Fine-tuned: `cap_5L_epoch_0`, `cap_5L_epoch_1`
  - Base models: `cap_5L` (no epoch suffix)
- This reduces compute and is scientifically correct (base models don't change)

**Should all runs be evaluated on all tasks?**
- **If yes**: Simple - all runs get all evaluation tasks
  - Good for fair comparison across all runs
  - Typical scientific use case
- **If no**: Ask which runs get which evaluations
  - Create an evaluation matrix showing which runs get which evals
  - Useful when:
    - Only evaluating larger models on expensive benchmarks
    - Testing transfer learning on subset of models
    - Controls get fewer evaluations than fine-tuned models
  - Will generate a visual matrix for approval

**Example evaluation scenarios:**
- **Single eval**: All runs evaluated on same primary task
- **Multiple evals, all runs**: All runs on [task1, task2, task3]
- **Selective evals**: 1B models on [task1], 3B models on [task1, task2, task3]
- **Staged evals**: Controls on [task1], fine-tuned on [task1, task2]

## Required Ingredients

For each run, verify and identify:

### 1. Base Models
- **Location**: `{models_dir}` (from `claude.local.md` or ask user)
- **Action**: Check that model paths exist using `ls` before proceeding

**If models not found:**
- Verify the `{models_dir}` path with the user
- Suggest using the `download-model-from-hf` skill to download missing models
- Provide estimated download time and size if known
- Pause planning until models are available or user confirms they'll download later

### 2. Dataset
- **Check**: Does the dataset exist?
- **Verify**: Does it have train/validation/test splits?
- **Show**: Dataset size and split information

Example for capitalization task:
```bash
# Check if exists
ls tasks/capitalization/input/words_{word_len}L_80P_{num_words}.json

# If not, create it
cd tasks/capitalization/input
python sample_words.py --word-len 8 --num-words 10000
```

**If dataset not found:**
- Verify the dataset path with the user
- For known tasks (e.g., capitalization), offer to create the dataset using existing scripts
- Ask user if they need help creating a custom dataset
- Document in the plan that dataset needs to be created before execution

**If dataset missing required splits:**
- Check if validation is truly needed (can disable validation in recipe choice)
- Ask user if they want to re-generate dataset with proper splits
- For test-only evaluation, confirm test split exists
- Update plan to reflect actual available splits

### 3. Evaluation Tasks
- **Action**: Verify all evaluation task scripts exist
- **Check**: For each evaluation task specified by the user:
  - Verify the script path exists
  - Verify the dataset path exists (if specified)
  - Document the task details

**For each evaluation task:**
```bash
# Check if eval script exists
ls tasks/capitalization/eval.py

# Check if eval dataset exists (if different from training data)
ls tasks/capitalization/test_set.json
```

**If evaluation script not found:**
- Check both `inspect.py` and `eval.py` in the task directory
- Suggest using the `create-evaluation` skill to create an evaluation script
- Ask user if they plan to create it later (document as prerequisite)
- For existing tasks, verify the script is in the expected location

**If evaluation dataset not found:**
- Verify the path with the user
- Check if it's the same as the test split from training data
- Ask user if they need help locating or creating it

## Recipe Selection Logic

Automatically determine the recipe based on GPU count and validation preference:

- **1 GPU:**
  - Without validation: `lora_finetune_single_device.py`
  - With validation: `lora_finetune_single_device_val.py`
- **Multiple GPUs:**
  - Without validation: `lora_finetune_distributed_v1.py`
  - With validation: `lora_finetune_distributed_val.py` (if available)

## Resource Estimation

**IMPORTANT**: Use actual prior run logs to estimate time, not assumptions!

The more prior runs exist, the better your estimates will be. This section shows how to gather data and make increasingly accurate estimates over time.

### Estimation Workflow

**Step 1: Search for prior runs**
```bash
# Find all prior experiment directories
ls {scratch_dir}/mjs3/ | grep -E "(cap|llama|lora)" | sort -r | head -10

# Look for similar configurations (same models, similar datasets)
```

**Step 2: Identify the most relevant prior runs**
- Prioritize runs with the SAME models you're planning
- Look for similar dataset sizes (±20% is fine)
- Check batch sizes and LoRA ranks (should match if possible)
- More recent runs are better (cluster performance may have changed)

**Step 3: Extract training data**
For each relevant prior run directory:
```bash
# Find the training log (usually the largest SLURM output file)
find {prior_run_dir} -name "slurm-*.out" -type f -size +100k | head -1

# Extract iteration speed from the log
tail -100 {slurm_log} | grep -E "it/s|s/it" | tail -20

# Check configuration
head -100 {slurm_log} | grep -E "batch_size|epochs|dataset|model"
```

**Step 4: Extract checkpoint sizes**
```bash
# Check disk space for each epoch
du -sh {prior_run_dir}/epoch_*

# Get average checkpoint size
du -sh {prior_run_dir}/epoch_0
```

**Step 5: Calculate estimates**
Use the formulas in the "Steps to Estimate Time" section below.

### Learning From Prior Runs

**Keep a mental model of typical values:**
- 1B models: typically 2-3 GiB per checkpoint, 5-15 it/s
- 3B models: typically 6-7 GiB per checkpoint, 3-7 it/s
- 7B+ models: typically 14-20 GiB per checkpoint, 1-3 it/s

**As you plan more experiments:**
1. **First experiment**: Make conservative estimates, clearly note they're preliminary
2. **Second experiment**: Use data from first experiment to refine
3. **Third+ experiments**: Average across multiple prior runs for more accuracy

**Track performance over time:**
- Note if speeds change (cluster upgrades, different GPU types)
- Update baseline estimates as new data becomes available
- Document any significant deviations and their causes

### Steps to Estimate Time:

1. **Find similar prior runs:**
   ```bash
   # Look for prior runs in output directory
   ls {output_dir} | grep "ck-out"

   # Find SLURM output logs
   find {output_dir}/ck-out-*/slurm-*.out -type f
   ```

2. **Extract training speed from logs:**
   ```bash
   # Look for iteration speed (e.g., "4.34it/s" or "8.38s/it")
   grep -E "[0-9.]+it/s|[0-9.]+s/it" /path/to/slurm-*.out | tail -20

   # Check model, batch size, dataset size, epochs
   grep -E "model:|batch_size:|epochs:|dataset_label:" /path/to/slurm-*.out | head -10
   ```

3. **Calculate time estimate:**
   - **Steps per epoch** = (dataset_size × train_fraction) ÷ batch_size
   - **Seconds per epoch** = steps_per_epoch ÷ iterations_per_second
   - **Total time** = seconds_per_epoch × epochs

4. **Scale for model size differences:**
   - If no prior run for exact model, use scaling factors:
     - 3B is ~2x slower than 1B
     - 7B is ~4-5x slower than 1B
     - 8B is ~5-6x slower than 1B

### Example Calculation:

**Prior run data** (from actual log):
- Model: Llama-3.2-1B
- Dataset: 5,000 words
- Batch size: 1
- Speed: 4.34 it/s (after warmup)
- Steps: 109
- Time: ~25 seconds for 1 epoch

**New experiment**:
- Model: Llama-3.2-1B
- Dataset: 10,000 words × 0.8 = 8,000 training samples
- Batch size: 4
- Epochs: 2
- Steps per epoch: 8,000 ÷ 4 = 2,000
- Time per epoch: 2,000 ÷ 4.34 ≈ 460s ≈ 8 minutes
- Total time: 8 min × 2 epochs = **16 minutes**

**DO NOT use generic assumptions** like "1B = 2 hours" - always base estimates on actual logs!

**If no prior runs available:**
- Ask user about their GPU type (A100 80GB, V100 32GB, etc.)
- Use conservative estimates based on model size and known benchmarks:
  - 1B models: Start with 30-60 min per epoch estimate
  - 3B models: Start with 1-2 hours per epoch estimate
  - 7B+ models: Start with 3-5 hours per epoch estimate
- Clearly document in the plan: "Estimates are preliminary; actual time will vary"
- Suggest running a short test job (1 epoch) to get accurate timing
- Update estimates after first run completes

## Evaluation Time Estimation

Evaluations are significantly faster than training because:
- Inference only (no backpropagation)
- No gradient computation
- Typically smaller test sets

### Steps to Estimate Evaluation Time:

1. **Use training speed as baseline:**
   - Evaluation is typically 2-3x faster than training
   - Same model, same GPU, but simpler computation

2. **Account for test set size:**
   ```
   Steps = test_set_size ÷ inference_batch_size
   Inference_speed = training_speed × 2.5  (middle of 2-3x range)
   Time = steps ÷ inference_speed
   ```

3. **Add overhead:**
   - Model loading: ~30 seconds
   - Result logging: ~10 seconds
   - Total overhead: ~1 minute per evaluation

### Example Calculation:

**Given:**
- Training speed: 10 it/s (from prior run)
- Test set: 1000 samples
- Inference batch size: 16 (often same as training batch size)

**Calculation:**
```
Inference speed: 10 × 2.5 = 25 it/s
Steps: 1000 ÷ 16 = 63 steps
Pure inference time: 63 ÷ 25 = 2.5 seconds
With overhead: 2.5 + 60 = 62.5 seconds ≈ 1 minute
```

For most tasks: **1-5 minutes per evaluation is typical**

### Evaluation-Specific Considerations:

1. **Multiple epochs:**
   - If evaluating both epoch_0 and epoch_1, multiply eval time × 2
   - Each epoch requires reloading the model

2. **Multiple tasks:**
   - Each task runs separately
   - Total time = num_tasks × time_per_task

3. **Base models:**
   - Evaluated once (not per epoch)
   - Faster to load (no adapter merging)

### Conservative Estimates (if no training data):

**Per evaluation job:**
- **1B models**: 2-3 minutes
- **3B models**: 4-5 minutes
- **7B models**: 8-10 minutes
- **70B+ models**: 20-30 minutes

**Total evaluation time formula:**
```
Total = (num_finetuned_runs × num_tasks × num_epochs × avg_eval_time) +
        (num_base_runs × num_tasks × avg_eval_time)
```

### Wall-Clock vs GPU Hours:

**Sequential execution (1 GPU):**
- Wall-clock time = total evaluation time
- GPU hours = total evaluation time

**Parallel execution (N GPUs):**
- Wall-clock time = longest_single_eval_time
- GPU hours = total evaluation time (same as sequential)

### Including in the Plan:

Always provide:
1. **Per-evaluation estimates** (how long each eval takes)
2. **Total sequential time** (if running one at a time)
3. **Total parallel time** (if running all simultaneously)
4. **Total GPU hours** (cost metric)
5. **Breakdown by model size** (1B vs 3B vs 7B, etc.)

## Dataset Packing Considerations

**Packing** is a memory optimization that concatenates multiple training examples into single sequences up to `max_seq_len`. It's more efficient but requires careful batch size tuning.

### When to Use Packing

**Use packing (`packed: True`) when**:
- Dataset has short examples (< 50% of max_seq_len on average)
- Training on instruction-following tasks with varied response lengths
- Want to maximize GPU utilization and training speed

**Avoid packing (`packed: False`) when**:
- Examples are already near max_seq_len (little to pack)
- Unsure about memory constraints (safer to start without)
- Debugging training issues (simpler to reason about batch sizes)

### Packing Density Analysis

Before choosing batch size, analyze packing density from prior runs.

#### How to Extract Packing Density from Logs

**Step 1: Locate the training log**
```bash
# Find SLURM output file in your run directory
ls /path/to/run_dir/slurm-*.out

# Example: /scratch/gpfs/username/experiments/run1/slurm-12345678.out
```

**Step 2: Search for packing information**
```bash
# Search for dataset packing messages
grep -i "packing dataset" /path/to/slurm-*.out

# Or look at the start of training
head -100 /path/to/slurm-*.out | grep -A10 "dataset"
```

**Step 3: Interpret the output**

Torchtune may show packing info in several ways:

**Example A: Explicit packing stats** (if recipe logs them)
```
Packing dataset: 100%|██████████| 8000/8000 [00:01<00:00]
INFO: Packed 8000 examples into 2857 sequences
INFO: Average packing density: 2.80 examples/sequence
INFO: Packing efficiency: 87% non-padding tokens
```
→ **Packing density = 2.8** (explicitly stated)

**Example B: Calculate from dataset info**
```
Loading dataset from /path/to/data/train.parquet...
Loaded 8000 examples
Packing dataset with max_seq_len=2048...
Dataset packed: 2857 total sequences
```
→ **Packing density = 8000 ÷ 2857 = 2.80**

**Example C: No explicit output** (older torchtune versions)
```
Dataset loaded: 8000 examples
Training started...
```
→ **Estimate based on task:**
  - Short prompts/responses (< 30% max_seq_len): density ≈ 3-4
  - Medium prompts/responses (30-60% max_seq_len): density ≈ 2-3
  - Long prompts/responses (> 60% max_seq_len): density ≈ 1.5-2

**Step 4: Document for future reference**

Add to your experiment notes or runs_plan.md:
```markdown
## Packing Analysis (from run: bright_horizon)
- Dataset: capitalization_5letter
- Examples: 8000
- Packed sequences: 2857
- **Packing density: 2.8 examples/sequence**
- Implication: batch_size with packing ≈ batch_size_unpacked ÷ 2.8
```

**Key metrics**:
- **Packing density**: Average examples per packed sequence (most important for memory estimation)
- **Packing efficiency**: % of tokens that are actual data vs padding (for throughput estimation)

**Typical values by task type**:
- 5-letter words (capitalization): ~3-4 examples/sequence (high density)
- 13-letter words (capitalization): ~2-3 examples/sequence (medium density)
- Long-form generation: ~1-2 examples/sequence (low density)
- Chat conversations: ~1.5-2.5 examples/sequence (variable density)

### Memory Impact of Packing

**Formula for packed memory estimation**:
```
Memory_packed ≈ Memory_unpacked × packing_density × 0.9
```

Where:
- `packing_density` = avg examples per sequence (e.g., 2.8)
- 0.9 = efficiency factor (packed sequences process more efficiently)

**Example**:
- Unpacked: batch_size=16, 1 example/seq → 16 examples, 2.4 GB
- Packed: batch_size=16, 2.8 examples/seq → 44.8 examples, ~6.0 GB

**This is why batch_size=16 with packed=True caused OOM!**

## Batch Size Estimation

**IMPORTANT**: Estimate maximum batch size from prior runs and GPU memory!

### Steps to Estimate Batch Size:

1. **Find GPU memory usage from prior runs:**
   ```bash
   # Look for GPU memory info in logs
   grep -E "GPU peak memory" {output_dir}/ck-out-*/slurm-*.out | head -5

   # Example output:
   # GPU peak memory allocation: 2.41 GiB
   # GPU peak memory reserved: 2.42 GiB
   ```

2. **Check configuration of that run:**
   ```bash
   grep -E "batch_size:|lora_rank:|model:" /path/to/slurm-*.out | head -10
   ```

3. **Calculate maximum batch size:**
   - **Formula**: max_batch_size ≈ (GPU_memory_available ÷ peak_memory_per_batch) × safety_factor
   - **Safety factor**: Use 0.6-0.7 to leave headroom

4. **Scale for different models:**
   - If changing batch size: memory scales roughly linearly
   - If changing model size:
     - 3B uses ~2-3x memory of 1B
     - 7B uses ~4-5x memory of 1B

### Example Calculation:

**Prior run data:**
- Model: Llama-3.2-1B
- Batch size: 1
- LoRA rank: 64
- GPU peak memory: 2.4 GB

**For 80GB GPU:**
- Headroom: 80 ÷ 2.4 ≈ 33x
- Conservative max (70% utilization): 33 × 0.7 ≈ 23
- **Recommended for 1B**: batch_size = 16-20

**For 3B model (estimated 2.5x more memory):**
- Base memory: 2.4 × 2.5 = 6 GB per batch
- Headroom: 80 ÷ 6 ≈ 13x
- Conservative max (70% utilization): 13 × 0.7 ≈ 9
- **Recommended for 3B**: batch_size = 8

### Accounting for Packing

If using `packed: True`, adjust batch size calculation:

1. **Find packing density from prior runs:**
   ```bash
   # Check logs for packing info
   tail -200 {prior_run_log} | grep -i "packing\|average"

   # If no prior data, estimate based on task:
   # - Short examples (< 512 tokens): density ≈ 3-4
   # - Medium examples (512-1024 tokens): density ≈ 2-3
   # - Long examples (> 1024 tokens): density ≈ 1.5-2
   ```

2. **Adjust batch size for packing:**
   ```
   max_batch_size_packed = max_batch_size_unpacked ÷ packing_density
   ```

3. **Apply safety factor:**
   ```
   recommended_batch_size = max_batch_size_packed × 0.6
   ```

**Example Calculation**:
- Prior run: batch_size=16, packed=False, memory=2.4 GB, GPU=80 GB
- Unpacked headroom: 80 ÷ 2.4 = 33x
- Conservative max unpacked: 33 × 0.7 = 23
- **If enabling packing** with density=2.8:
  - Max packed: 23 ÷ 2.8 ≈ 8
  - Recommended: 8 × 0.6 ≈ 5
  - **Use batch_size=4 or batch_size=8 (test 8 first)**

**If no packing data available**:
- Start with batch_size=4 (1B), batch_size=2 (3B) when using packed=True
- Monitor first run and scale up if memory allows
- Document actual packing density for future reference

### Important Notes:

- **LoRA rank has minimal impact** on memory (LoRA adds very little compared to base model)
- **Sequence length matters**: Longer sequences use more memory
- **Packing has MAJOR impact**: Reduces batch size by 2-4x compared to unpacked
- **Always test**: Start conservative, monitor actual usage, adjust if needed
- **Different batch sizes per model**: It's fine to use different batch sizes for different model sizes (e.g., 16 for 1B, 8 for 3B)

**If no GPU memory data available:**
- Ask user about their GPU type (A100 80GB, V100 32GB, H100, etc.)
- Use conservative defaults based on GPU memory AND packing setting:

**With packed=True (default)**:
  - **80GB GPUs**: 1B → batch_size=4-8, 3B → batch_size=2-4, 7B+ → batch_size=2
  - **40GB GPUs**: 1B → batch_size=2-4, 3B → batch_size=2, 7B+ → batch_size=1
  - **32GB GPUs or less**: Start with batch_size=1 for all models

**With packed=False (if using unpacked)**:
  - **80GB GPUs**: 1B → batch_size=16, 3B → batch_size=8, 7B+ → batch_size=4
  - **40GB GPUs**: 1B → batch_size=8, 3B → batch_size=4, 7B+ → batch_size=2
  - **32GB GPUs or less**: Start with batch_size=2 for 1B, batch_size=1 for 3B+

- Document in the plan that batch sizes are conservative estimates
- Document whether packing is enabled and estimated density
- Recommend monitoring first run to optimize batch size
- Better to start small and scale up than OOM (out of memory) errors

## Disk Space Estimation

**IMPORTANT**: Calculate disk space requirements to avoid running out of storage mid-experiment!

### Steps to Estimate Disk Space:

1. **Find checkpoint sizes from prior runs:**
   ```bash
   # Check actual checkpoint sizes
   du -sh {prior_run_dir}/epoch_0
   du -sh {prior_run_dir}/epoch_1

   # Get average size
   du -sh {prior_run_dir}/epoch_* | awk '{sum+=$1} END {print sum/NR}'
   ```

2. **Calculate total checkpoint storage:**
   ```
   Total = num_runs × num_epochs × checkpoint_size_per_epoch
   ```

3. **Add evaluation log overhead:**
   - Inspect-AI logs are small: ~5-20 MB per evaluation
   - Total eval logs: num_evaluations × 10 MB (conservative estimate)

4. **Add safety margin:**
   - Add 10-20% buffer for temporary files and logs
   - Torchtune may create intermediate checkpoints

### Example Calculation:

**For a 2×2 factorial design:**
- 4 fine-tuned runs (1B and 3B, rank 4 and 64)
- 2 epochs each
- Prior data shows:
  - 1B checkpoints: 2.4 GiB each
  - 3B checkpoints: 6.1 GiB each

**Calculation:**
```
1B storage: 2 runs × 2 epochs × 2.4 GiB = 9.6 GiB
3B storage: 2 runs × 2 epochs × 6.1 GiB = 24.4 GiB
Eval logs: 20 evals × 0.01 GiB = 0.2 GiB
Safety margin (15%): (9.6 + 24.4 + 0.2) × 0.15 = 5.1 GiB

Total: 9.6 + 24.4 + 0.2 + 5.1 = 39.3 GiB ≈ 40 GiB
```

### Checkpoint Size Estimates (if no prior data):

**Typical checkpoint sizes by model size:**
- **1B models**:
  - Base model: ~2 GiB
  - With LoRA adapters: ~2.3-2.5 GiB
- **3B models**:
  - Base model: ~6 GiB
  - With LoRA adapters: ~6.1-6.3 GiB
- **7B models**:
  - Base model: ~14 GiB
  - With LoRA adapters: ~14.2-14.5 GiB
- **70B models**:
  - Base model: ~140 GiB
  - With LoRA adapters: ~141-143 GiB

**Note**: LoRA adds minimal overhead because it only saves adapter weights (typically <100 MB for rank 64).

### Disk Space Best Practices:

1. **Check available space before starting:**
   ```bash
   df -h {scratch_dir}
   ```

2. **Plan for cleanup:**
   - Keep only necessary epochs (use `epochs_to_save` in config)
   - Delete intermediate checkpoints after training
   - Archive completed experiments if space is tight

3. **Monitor during execution:**
   - Check disk usage if jobs fail unexpectedly
   - Out-of-space errors can corrupt checkpoints

4. **Include in plan:**
   - Total disk space needed
   - Available space verification
   - Cleanup strategy if needed

## Run Summary

Once you understand the run structure, create a summary table:

### Example Output Format

```markdown
## Run Plan Summary

### Run Design
**Type**: 2×2 Factorial Design
**Total Runs**: 6 (4 fine-tuned + 2 controls)

### Variables
| Factor | Levels |
|--------|--------|
| Model | 1B-Instruct, 3B-Instruct |
| LoRA Rank | 4, 64 |

### All Runs
| # | Model | LoRA Rank | Type | Est. Time |
|---|-------|-----------|------|-----------|
| 1 | 1B | 4 | Fine-tuned | 2h |
| 2 | 1B | 64 | Fine-tuned | 2h |
| 3 | 3B | 4 | Fine-tuned | 4h |
| 4 | 3B | 64 | Fine-tuned | 4h |
| 5 | 1B | - | Control | eval only |
| 6 | 3B | - | Control | eval only |

### Resources Verified ✓
- Models: `{models_dir}/Llama-3.2-{1B,3B}-Instruct`
- Dataset: `tasks/capitalization/input/words_8L_80P_10000.json` (655KB)
- Evaluation: `tasks/capitalization/inspect.py`

### Configuration
- Recipe: `lora_finetune_single_device.py` (1 GPU)
- Epochs: 1
- Batch size: 4 (1B), 2 (3B)
- Total GPU hours: ~12 hours
```

## Naming Convention

**IMPORTANT**: Run names must use the full HuggingFace model naming convention (without the organization prefix) to ensure clarity and traceability.

**Pattern for fine-tuned runs**: `{full_model_name}_{experimental_factors}_rank{lora_rank}`
**Pattern for control runs**: `{full_model_name}_base`

**Examples**:
- `Llama-3.2-1B-Instruct_5L_rank4` (1B model, 5-letter words, rank 4)
- `Llama-3.2-1B-Instruct_5L_rank64` (1B model, 5-letter words, rank 64)
- `Llama-3.2-3B-Instruct_9L_rank4` (3B model, 9-letter words, rank 4)
- `Llama-3.2-3B-Instruct_13L_rank64` (3B model, 13-letter words, rank 64)
- `Llama-3.2-1B-Instruct_base` (1B control run)
- `Llama-3.2-3B-Instruct_base` (3B control run)

**Rationale**:
- Uses exact HuggingFace model names (e.g., `Llama-3.2-1B-Instruct` from `meta-llama/Llama-3.2-1B-Instruct`)
- Makes run names self-documenting and unambiguous
- Facilitates searching and organization
- Matches directory names in model storage

## SLURM Job Organization Strategy

Once runs are planned, they will be organized as **individual scripts with sequential submission**:

### Directory Structure
Each run gets its own directory with complete configuration:
```
{run_group_name}/
  Llama-3.2-1B-Instruct_5L_rank4/
    setup_finetune.yaml    # Input config for this run
    finetune.yaml          # Generated torchtune config
    finetune.slurm         # Generated SLURM script
  Llama-3.2-1B-Instruct_5L_rank64/
    setup_finetune.yaml
    finetune.yaml
    finetune.slurm
  Llama-3.2-3B-Instruct_9L_rank4/
    ...
  Llama-3.2-3B-Instruct_9L_rank64/
    ...
  submit_all.sh            # Master submission script
  runs_plan.md             # Run plan documentation (this file)
  runs_status.yaml         # Status tracker
```

### Master Submission Script
The `submit_all.sh` script submits all jobs sequentially:
```bash
#!/bin/bash
# Submit all fine-tuning jobs

cd Llama-3.2-1B-Instruct_5L_rank4 && sbatch finetune.slurm && cd ..
cd Llama-3.2-1B-Instruct_5L_rank64 && sbatch finetune.slurm && cd ..
cd Llama-3.2-3B-Instruct_9L_rank4 && sbatch finetune.slurm && cd ..
cd Llama-3.2-3B-Instruct_9L_rank64 && sbatch finetune.slurm && cd ..

echo "All jobs submitted!"
echo "Check status: squeue -u \$USER"
```

### Benefits
- Each run is self-contained and reproducible
- Easy to modify individual runs
- Separate logs per run (`slurm-{job_id}.out`)
- Can re-run individual runs without affecting others
- Works seamlessly with existing `setup_finetune.py` workflow
- Master script provides convenience for bulk submission

## Validation Checklist

Before presenting the final plan to the user, validate that all critical requirements are met:

### Resource Verification
- [ ] **All base models exist** at specified paths
  - Used `ls {models_dir}/model_name` to verify each model
  - All model paths are accessible
- [ ] **Dataset exists** with complete splits
  - Dataset file/directory found at specified path
  - Has train, validation (if needed), and test splits
  - Split sizes documented and reasonable
- [ ] **All evaluation tasks verified**
  - All evaluation scripts exist at specified paths
  - All evaluation datasets exist (if different from training data)
  - Evaluation matrix is complete and consistent
  - Each run has at least one evaluation assigned

### Estimate Validation
- [ ] **Time estimates based on actual logs** (not assumptions)
  - Found similar prior runs in `{output_dir}/ck-out-*/`
  - Extracted actual iteration speed from SLURM logs
  - Calculated estimates using documented formula
  - OR: If no prior runs, clearly noted as "estimated, not verified"
- [ ] **Batch sizes fit in GPU memory**
  - Calculated from prior GPU memory usage
  - Scaled appropriately for different model sizes
  - Conservative safety factor applied (0.6-0.7)
  - OR: User confirmed GPU type and batch sizes are conservative

### Plan Completeness
- [ ] **All runs properly named** following convention
  - Names clearly indicate experimental factors
  - No duplicate run names
- [ ] **Configuration consistent** across runs
  - Same epochs, same recipe type (except for experimental variables)
  - LoRA alpha correctly calculated (2 × rank)
  - Validation settings match recipe choice
- [ ] **Compute estimates calculated**
  - Total training time summed
  - Total GPU hours estimated
  - Estimates clearly documented in summary

### Summary Quality
- [ ] **runs_plan.md template filled completely**
  - All placeholders replaced with actual values
  - All paths are absolute and verified
  - Table includes all runs (experimental + controls)
  - Evaluation tasks table is complete
  - Evaluation matrix shows which runs get which evals
  - Naming conventions documented
  - Quick Reference section includes experiment-specific paths and workflow
- [ ] **runs_status.yaml structure prepared**
  - All run names match between plan and status files
  - Each run has finetune and evaluations sections
  - Evaluation tasks match the evaluation matrix from plan
  - Initial state is `pending` for all finetune and evaluation tasks

**If any checklist item fails:** Fix the issue before proceeding to user approval. If something is missing or unclear, ask the user for clarification.

## Next Steps

Once the validation checklist is complete and the plan is finalized:

1. **Create the deliverables:**
   - Write `runs_plan.md` to `{run_group_name}/runs_plan.md`
   - Initialize `runs_status.yaml` in the same directory with all runs in `pending` state

2. **Ask the user if they want to proceed:**
   - "I've created the run plan. Would you like me to set up the directory structure and generate the configuration files for each run?"
   - If yes: Invoke the `setup-experiment-dirs` skill to create directories and configs
   - If no: Inform them they can use the `setup-experiment-dirs` skill later when ready

3. **Handoff context to next skill:**
   - The `setup-experiment-dirs` skill will read `runs_plan.md` to create:
     - Individual run directories
     - `setup_finetune.yaml` configs for each run
     - `submit_all.sh` master script
     - Any other necessary files

## Important Notes

- Training, validation, and evaluation should be random splits of one data file
- All base LLMs follow HuggingFace naming conventions
- Output location: `{output_dir}` (from `claude.local.md` or user's scratch directory)
- LoRA alpha is automatically set to 2 × rank by setup_finetune.py
- Model location: `{models_dir}` (from `claude.local.md` or user's models directory)

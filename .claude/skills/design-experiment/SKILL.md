# Design Experiment

You help users plan experiments for fine-tuning and evaluating LLMs. Create a plan that specifies the complete workflow from training through evaluation, verifies resources, estimates compute requirements, and documents all steps.

## Your Task

Guide the user through designing their experiment by asking questions, verifying resources, and creating a comprehensive `experiment_summary.md` file that documents the complete plan.

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

1. **Understand the experiment** - What variables are being tested? What's the scientific question?
2. **Confirm tool choices** - Ask which preparation and evaluation tools to use (currently only torchtune and inspect-ai)
3. **Design training runs** - Which models? Which datasets? What hyperparameters? What LoRA ranks?
4. **Design evaluation runs** - Which trained models on which tasks? Which epochs to evaluate?
5. **Establish naming** - Choose descriptive names for the experiment and runs
6. **Verify resources** - Check that models, datasets, and eval scripts exist (log all checks)
7. **Estimate resources** - Calculate time and disk space for BOTH training and evaluation (log all calculations)
8. **Get approval** - Present the complete plan to user, adjust if needed
9. **Create files** - After approval, write `experiment_summary.md` and `design-experiment.log`

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
Explanation: Comprehensive experiment plan with all configurations documented

[2025-10-22 14:26:15] COMPLETE: Experiment design finished
Status: Plan approved by user, files written
Next Steps: User can now run scaffold-experiment to generate configs
Explanation: Planning phase complete, documented in summary and this log
```

### Purpose of the Log

The log enables:
1. **Debugging:** If estimates are wrong, check what commands were run and what data was used
2. **Reproducibility:** Another person (or Claude) can understand exactly what was done
3. **Improvement:** Review logs to identify better approaches or missing steps
4. **Auditing:** Verify that all resources were properly checked before committing to the experiment

## Questions to Ask (Follow Workflow Order)

These questions map directly to the workflow steps above.

### 1. Understand the Experiment

**What is the scientific question?**
- What are you trying to learn?
- What variables are you testing?
- What are the experimental factors and levels?

**Should we include base model controls?**
- Controls evaluate base models without fine-tuning to measure the effect of fine-tuning

### 2. Tool Selection

**Which tools will you use for this experiment?**

**Model preparation:**
- torchtune (currently the only option)
- *Future:* Other fine-tuning frameworks may be supported

**Evaluation:**
- inspect-ai (currently the only option)
- *Future:* Other evaluation frameworks may be supported

**Note:** While these are currently the only options, explicitly confirming and documenting tool choices now will make it easier to support multiple tools in future iterations.

### 3. Design Training Runs

**Which models?**
- Which model(s) to fine-tune? (e.g., 1B, 3B, 8B)
- Check `{models_dir}` from `claude.local.md`

**Which dataset?**
- Training dataset location and format
- Required splits: train, validation (optional), test (optional)

**What variables are you testing?**
- Different model sizes?
- Different LoRA ranks?
- Different datasets or data sizes?
- Different hyperparameters?
- Combinations of the above?

**Training configuration:**

**Basic settings:**
- How many epochs? (default: 1-2)
- How many GPUs per job? (default: 1)
- Should validation run during training? (default: yes)
- System prompt for training and evaluation? (default: "")

**Advanced settings (calculate from prior runs if available):**
- Batch sizes - estimate from GPU memory usage in prior runs
- Dataset packing - enabled by default, affects batch size
- For help estimating: check `{scratch_dir}/*/slurm-*.out` for similar runs

### 4. Design Evaluation Runs

**Which evaluation tasks?**
- Which inspect-ai task(s) to run?
- For each task: name, script path, dataset path (if different from training), description
- Does the task exist or need to be created? (use `create-inspect-task` skill if needed)

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

**Evaluation configuration:**
- System prompt must match training for consistency
- Temperature typically 0.0 for deterministic evaluation
- Scorer selection (exact match, includes, model-graded, etc.)

**Important:** Base models evaluate once per task (no epoch suffix), fine-tuned models evaluate per epoch.

### 5. Establish Naming

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

### 6. Verify Resources

Now that the design is complete, verify all resources exist (use `claude.local.md` for default paths):

**Models:** `ls {models_dir}/{model_name}`
- Verify each model directory exists

**Training dataset:** `ls -lh {dataset_path}`
- Check file exists and note size
- Verify required splits (train, validation if needed, test if needed)

**Evaluation task scripts:** `ls {eval_script_path}`
- Verify each inspect-ai task script exists
- If missing, note as prerequisite (may need `create-inspect-task` skill first)

**Disk space:** `df -h {scratch_dir}`
- Ensure sufficient space for checkpoints

**If resources missing:**
- Model: Suggest downloading with appropriate tool
- Dataset: Offer to help create it (if known task like capitalization)
- Eval script: Note as prerequisite, proceed with plan anyway
- Disk space: Warn user, suggest cleanup or alternative location

### 7. Estimate Resources

Calculate compute requirements for the complete experiment (training + evaluation):

**Training time:** Estimate per-run and total training time (see Estimation Guidelines below)
**Evaluation time:** Estimate total evaluation time across all runs and tasks
**Disk space:** Calculate checkpoint storage requirements
**GPU hours:** Sum total GPU time needed

Use the **Estimation Guidelines** section below for methods and formulas.

## Estimation Guidelines

This section provides detailed methods for making estimates requested in step 7 above.

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

### Quick Reference Section

Include experiment-specific quick reference:
```markdown
## Quick Reference

**Paths:**
- Experiment: `{full_path_to_experiment_dir}`
- Models: `{models_dir}/{model_names}`
- Dataset: `{dataset_path}`

**Common Commands:**
- List available models: `ls {models_dir}`
- Check dataset: `ls -lh {dataset_path}`
- Find prior runs: `find {scratch_dir} -name "slurm-*.out" -path "*/ck-out-*" | head -10`
- Extract training speed: `grep -E "[0-9.]+it/s" {prior_run_path}/slurm-*.out | tail -20`
- Check disk space: `df -h {scratch_dir}`

**Next Steps:**
1. [Manual step or placeholder for future skill]
2. Generate configs for each run
3. Submit fine-tuning jobs
4. Monitor progress
5. Submit evaluation jobs
6. Analyze results
```

## Template Structure

The experiment_summary.md file should follow the section order listed in "Required Sections" above. Here are examples of complex sections:

**Evaluation Matrix Example** (when runs have different evaluation plans):
```markdown
## Evaluation Plan

### Evaluation Matrix
| Run Name | capitalization_task | reasoning_task | Notes |
|----------|---------------------|----------------|-------|
| Llama-3.2-1B_rank4 | ✓ epoch 0,1 | ✓ epoch 0,1 | All evals |
| Llama-3.2-3B_rank4 | ✓ epoch 0,1 | - | Cap only |
| Llama-3.2-1B_base | ✓ | ✓ | Base control |
```

**Tools Section Example** (documents which preparation and evaluation tools are used):
```markdown
## Tools

- **Model Preparation:** torchtune
  - *Purpose:* Fine-tuning LLMs with LoRA
  - *Used by:* `scaffold-torchtune` and `run-torchtune` skills

- **Evaluation:** inspect-ai
  - *Purpose:* Evaluating LLMs on custom tasks
  - *Used by:* `scaffold-inspect` and `run-inspect` skills
```

For complete examples, refer to existing experiment_summary.md files in `experiments/*/experiment_summary.md`.

## Validation Before Presenting to User

Before presenting the plan for approval (step 8 of workflow), verify it's complete:
- ✓ All models verified
- ✓ Dataset verified with correct splits
- ✓ Evaluation scripts verified (or noted as prerequisites)
- ✓ Time estimates calculated or clearly marked as preliminary
- ✓ Disk space checked
- ✓ All run names follow convention
- ✓ Evaluation matrix is consistent

## After User Approval (Step 9 of Workflow)

Once the user approves the plan:

1. **Create the files:**
   - Write `experiment_summary.md` with the approved plan
   - Write `design-experiment.log` with all verification steps and decisions

2. **Ask about next steps:**
   - "I've created the experiment plan at `{path}/experiment_summary.md`."
   - "Would you like me to proceed with scaffolding? I can run `scaffold-experiment` to generate all configs."

3. **Automated workflow (recommended):**
   - Run `scaffold-experiment` skill to generate:
     - Fine-tuning configs via `scaffold-torchtune` (finetune.yaml, finetune.slurm)
     - Evaluation configs via `scaffold-inspect` (inspect.slurm, task scripts)
   - Run `run-experiment` skill to execute:
     - Fine-tuning via `run-torchtune` (submit jobs, monitor progress)
     - Evaluation via `run-inspect` (submit jobs after training completes, monitor progress)
   - Run `analyze-experiment` skill to interpret results (planned)

4. **Manual workflow (if needed):**
   - User can manually create directories and configs
   - Follow the experiment plan as documented in experiment_summary.md

## Important Notes

- Use paths from `claude.local.md` for models, datasets, scratch directories
- Always verify resources exist before finalizing plan
- Be conservative with estimates if no prior run data available
- **System prompt must be consistent between training and evaluation** (critical for inspect-ai)
- Base models evaluate once (no epoch), fine-tuned models evaluate per epoch
- Document which tool is used at each stage (torchtune for training, inspect-ai for evaluation)
- Evaluation datasets may differ from training datasets (document clearly)
- If inspect-ai task doesn't exist, note that `create-inspect-task` skill should be run first

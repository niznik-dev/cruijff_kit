# Parameter Selection

Guide the user through the 10-step interactive workflow to gather all experiment parameters.

## Workflow Overview

1. **Determine experiment type and location** - Auto-detect sanity_check vs experiment
2. **Understand the experiment** - What variables? What's the scientific question?
3. **Confirm tool choices** - Which optimizer and evaluator to use
4. **Design training runs** - Models, datasets, hyperparameters
5. **Design evaluation runs** - Tasks, epochs, evaluation matrix
6. **Establish naming** - Experiment and run names
7. **Verify resources** - Check models, datasets, eval scripts exist
8. **Estimate resources** - Calculate time, disk space, GPU hours
9. **Get approval** - Present complete plan (validate first via `validation.md`)
10. **Create files** - Proceed to `experiment_generation.md`

---

## Step 1: Determine Experiment Type and Location

### Auto-Detect Based on Working Directory

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

### Directory Structure

- **Experiments** (research tasks): `/scratch/gpfs/MSALGANIK/niznik/ck-experiments/{experiment_name}/`
- **Sanity checks** (workflow validation): `/scratch/gpfs/MSALGANIK/niznik/ck-sanity-checks/{sanity_check_name}/`

**Outputs are automatically grouped:**
- Output directory: `/scratch/gpfs/MSALGANIK/niznik/ck-outputs/{experiment_or_sanity_check_name}/ck-out-{run_name}/`

### Confirm with User

**Are you working on a sanity check or a research experiment?**
- Log the detected type and path for user confirmation
- Note in experiment_summary.md that outputs will be grouped under the same name in ck-outputs/

---

## Step 2: Understand the Experiment

### Key Questions

**What is the scientific question?**
- What are you trying to learn?
- What variables are you testing?
- What are the experimental factors and levels?

**Should we include base model controls?**
- Controls evaluate base models without fine-tuning to measure the effect of fine-tuning

---

## Step 3: Confirm Tool Choices

### Available Tools

**Model Preparation (Optimizer):**
- **torchtune** (currently the only option)
  - Used by: `scaffold-experiment` and `run-experiment` skills
  - Generates: `finetune.yaml`, `finetune.slurm`
  - Produces: Model checkpoints in `output_dir_base`
- *Future:* Other fine-tuning frameworks may be supported

**Evaluation (Evaluator):**
- **inspect-ai** (currently the only option)
  - Used by: `scaffold-experiment` and `run-experiment` skills
  - Generates: `inspect.slurm` and/or inspect task scripts
  - Produces: Evaluation logs (`.eval` files)
- *Future:* Other evaluation frameworks may be supported

### Document in experiment_summary.md

```markdown
## Tools

- **Model Preparation:** torchtune
  - *Purpose:* Fine-tuning LLMs with LoRA
  - *Used by:* `scaffold-experiment` and `run-experiment` skills

- **Evaluation:** inspect-ai
  - *Purpose:* Evaluating LLMs on custom tasks
  - *Used by:* `scaffold-experiment` and `run-experiment` skills
```

**Note:** While these are currently the only options, explicitly confirming and documenting tool choices now will make it easier to support multiple tools in future iterations.

---

## Step 4: Design Training Runs

### Which Models?
- Which model(s) to fine-tune? (e.g., 1B, 3B, 8B)
- Check `{models_dir}` from `claude.local.md`

### Which Dataset?
- Training dataset location and format
- Required splits: train, validation (optional), test (optional)

**Note:** If the dataset needs to be created, check `experiments/{task}/README.md` for:
- Dataset naming conventions (e.g., parameter-based filenames)
- Preprocessing script details and usage
- Expected output formats and locations

### What Variables Are You Testing?
- Different model sizes?
- Different LoRA ranks?
- Different datasets or data sizes?
- Different hyperparameters?
- Combinations of the above?

### Should We Include Base Model Controls?
- Controls evaluate base models without fine-tuning to measure the effect of fine-tuning

### Training Configuration

**Basic settings:**
- How many epochs? (default: 1-2)
- How many GPUs per job? (default: 1)
- Should validation run during training? (default: yes)
- System prompt for training and evaluation? (default: "")

**Advanced settings (calculate from prior runs if available):**
- Batch sizes - estimate from GPU memory usage in prior runs
- Dataset packing - enabled by default, affects batch size
- For help estimating: check `{scratch_dir}/*/slurm-*.out` for similar runs

### Document All Runs Table

Create a table documenting all fine-tuned and control runs:

```markdown
## All Runs

| Run Name | Model | LoRA Rank | Learning Rate | Batch Size | Type | Est. Time |
|----------|-------|-----------|---------------|------------|------|-----------|
| Llama-3.2-1B_rank8_lr1e-5 | Llama-3.2-1B-Instruct | 8 | 1e-5 | 4 | Fine-tuned | ~10min |
| Llama-3.2-1B_rank8_lr5e-5 | Llama-3.2-1B-Instruct | 8 | 5e-5 | 4 | Fine-tuned | ~10min |
| Llama-3.2-1B_base | Llama-3.2-1B-Instruct | - | - | - | Control | N/A |

**Notes**:
- **Type**: "Fine-tuned" for runs requiring training, "Control" for base model evaluation only
- **Run Name**: Should match directory structure (varying parameters only)
- Include all parameters that vary across runs as separate columns
- Use `-` for non-applicable parameters (like LoRA rank for control runs)
```

---

## Step 5: Design Evaluation Runs

### Which Evaluation Tasks?
- Which inspect-ai task(s) to run?
- For each task: name, script path, dataset path (if different from training), description
- Does the task exist or need to be created? (use `create-inspect-task` skill if needed)

### Which Epochs to Evaluate?
- **NOTE:** Epochs are 0-indexed. Training for 1 epoch produces `epoch_0`, training for 2 epochs produces `epoch_0` and `epoch_1`, etc.
- Last epoch only (default, most efficient)
  - After 1 epoch of training, this is `epoch_0`
  - After 2 epochs of training, this is `epoch_1`
- All epochs (compare training progression)
- Specific epochs (e.g., epoch 0 and final)
- Best by validation loss (requires validation during training)

### Which Runs Get Which Evaluations?
- All runs on all tasks (typical)
- Selective (e.g., only large models on expensive evals)
- If selective, create evaluation matrix

### Evaluation Datasets
- Same as training dataset (typical for overfitting checks)
- Different test set (typical for generalization evaluation)
- Multiple evaluation datasets (comprehensive assessment)

### Evaluation Configuration
- System prompt must match training for consistency
- Temperature typically 0.0 for deterministic evaluation
- Scorer selection (exact match, includes, model-graded, etc.)

**Important:** Base models evaluate once per task (no epoch suffix), fine-tuned models evaluate per epoch.

### Document Evaluation Plan

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

---

## Step 6: Establish Naming

### Experiment Name

Help the user choose a descriptive experiment name that includes:
- Task/dataset indicator (e.g., `cap_8L` for capitalization 8-letter)
- Key experimental factor (e.g., `lora_comparison`, `model_sizes`)
- Date (YYYY-MM-DD format)

**Example patterns:**
- `cap_8L_lora_comparison_2025-10-18` (capitalization, varying LoRA rank)
- `twins_model_sizes_2025-10-22` (synthetic twins, varying model size)
- `reasoning_ablation_2025-11-01` (reasoning task, ablation study)

### Run Names

Use full model names with experimental factors:
- `Llama-3.2-1B-Instruct_rank4`
- `Llama-3.2-3B-Instruct_rank64`
- `Llama-3.2-1B-Instruct_base` (control)

---

## Step 7: Verify Resources

Now that the design is complete, verify all resources exist (use `claude.local.md` for default paths).

### Models
**Command:** `ls {models_dir}/{model_name}`
- Verify each model directory exists
- Note approximate size

### Training Dataset
**Command:** `ls -lh {dataset_path}`
- Check file exists and note size
- Verify required splits (train, validation if needed, test if needed)

**If dataset was just created:**
- Verify the actual filename matches what was documented in experiment_summary.md
- Generated datasets often include parameters in filename (e.g., `words_7L_80P_1000.json`)
- Update experiment_summary.md if filename differs from initial expectations

### Evaluation Task Scripts
**Command:** `ls {eval_script_path}`
- Verify each inspect-ai task script exists
- If missing, note as prerequisite (may need `create-inspect-task` skill first)

### Disk Space
**Command:** `df -h {scratch_dir}`
- Ensure sufficient space for checkpoints

### If Resources Missing

**Model:** Suggest downloading with appropriate tool

**Dataset:**
1. **Check for existing preprocessing scripts:** Look in `experiments/{task}/` for scripts like `preprocess_*.py`
2. **Consult task README:** Check `experiments/{task}/README.md` for dataset creation instructions
3. **Use the proper tool:** Run the task-specific preprocessing script with appropriate parameters
4. **DON'T:** Write ad-hoc Python code or copy dataset creation code from previous experiments
5. **WHY:** Task-specific scripts ensure correct format, naming conventions, and reproducibility

**Eval script:** Note as prerequisite, proceed with plan anyway

**Disk space:** Warn user, suggest cleanup or alternative location

### Document in experiment_summary.md

```markdown
## Resources

### Models
- **Llama-3.2-1B-Instruct**: `{models_dir}/Llama-3.2-1B-Instruct`
  - Verified: ✓ (2025-10-22)
  - Size: ~2.5 GB

### Dataset
- **Path**: `{repo_dir}/data/green/capitalization/words_8L_80P_10000.json`
- **Format**: JSON
- **Size**: 655KB
- **Splits**: train (8000 samples), validation (1000 samples), test (1000 samples)
- **Verified**: ✓ (2025-10-22)

### Evaluation Tasks
| Task Name | Script | Dataset | Description |
|-----------|--------|---------|-------------|
| capitalization | `{repo_dir}/experiments/capitalization/inspect_task_capitalization.py` | Same as training | Tests word capitalization accuracy |

**Note**: All paths verified during design phase. Evaluation task scripts must exist before scaffolding.
```

### Log All Verification Steps

All resource verification commands should be logged in `design-experiment.log` (see `logging.md`).

---

## Step 8: Estimate Resources

Calculate compute requirements for the complete experiment (training + evaluation).

### What to Estimate

- **Training time:** Per-run and total training time
- **Evaluation time:** Total evaluation time across all runs and tasks
- **Disk space:** Checkpoint storage requirements
- **GPU hours:** Sum total GPU time needed

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

### Document in experiment_summary.md

```markdown
## Compute Estimates

### Training
- **Per-run time:** ~10 minutes/epoch
- **Total runs:** 4 fine-tuned runs
- **Total training time:** ~80 minutes (4 runs × 2 epochs × 10 min)

### Evaluation
- **Per-eval time:** ~2 minutes
- **Total evals:** 16 (4 runs × 2 tasks × 2 epochs)
- **Total eval time:** ~32 minutes

### Disk Space
- **Per-epoch checkpoint:** ~2.5 GiB
- **Total checkpoints:** ~40 GiB (4 runs × 2 epochs × 2.5 GiB + 20% buffer)
- **Available space:** 2.1T

### Total GPU Hours
- **Training:** ~1.3 hours
- **Evaluation:** ~0.5 hours
- **Total:** ~1.8 GPU hours
```

### Log All Calculations

All estimation calculations should be logged in `design-experiment.log` (see `logging.md`).

---

## Step 9: Get Approval

Before presenting the plan to the user, validate completeness using `validation.md`.

### Present Complete Plan

Show the user:
- Overview (experiment type, total runs, scientific question)
- All runs table
- Evaluation plan
- Resource estimates (training time, eval time, disk space, GPU hours)

### Adjust if Needed

Based on user feedback:
- Modify configurations
- Add/remove runs
- Change evaluation strategy
- Adjust estimates

### After Approval

Proceed to `experiment_generation.md` to create the files.

---

## Conversation Patterns

### Opening
```
I'll help you design this experiment. Let me start by understanding what you want to test.

I see you're working in [directory]. This looks like a [sanity check / research experiment].
I'll create the experiment in: {base_dir}{experiment_name}/

What scientific question are you trying to answer?
```

### During Design
```
Great! So you're testing [variable] across [levels].

Let me verify the models exist...
[checks and logs]

Now let's estimate how long this will take. I'll look for similar prior runs...
[searches, extracts, calculates, logs]
```

### Before Approval
```
Here's the complete experiment plan:

**Overview:**
- X fine-tuning runs (varying [factors])
- Y evaluation tasks
- Z total evaluations

**Estimated resources:**
- Training time: ~X hours
- Eval time: ~Y minutes
- Disk space: ~Z GiB
- Total GPU hours: ~W

Does this look correct? Any adjustments needed?
```

---

## Key Principles

1. **Ask, don't assume** - Even when you can auto-detect, confirm with user
2. **Log everything** - All verifications and calculations go in design-experiment.log (see `logging.md`)
3. **Validate before presenting** - Use `validation.md` to ensure plan is complete
4. **Be conservative** - When estimating without prior data, give ranges and mark as preliminary
5. **Handle missing resources gracefully** - Note as prerequisites, don't block the plan
6. **System prompt consistency** - Critical for inspect-ai, verify it matches between training and eval

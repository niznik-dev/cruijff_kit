# Parameter Selection

Guide the user through the 9-step interactive workflow to gather all experiment parameters.

## Workflow Overview

1. **Determine experiment location** - Auto-detect sanity_check vs experiment
2. **Understand the experiment** - What variables? What's the scientific question?
3. **Confirm tool choices** - Which optimizer and evaluator to use
4. **Design training runs** - Models, datasets, hyperparameters
5. **Design evaluation runs** - Tasks, epochs, evaluation matrix
6. **Establish naming** - Experiment and run names
7. **Verify resources** - Check models, datasets, eval scripts exist
8. **Get approval** - Present complete plan (validate first via `validation.md`)
9. **Create files** - Proceed to `experiment_generation.md`

---

## Step 1: Determine Experiment Location

### Derive Paths from claude.local.md

1. Read the **Scratch directory** field from `claude.local.md`
2. Determine experiment type based on user intent or working directory context:
   - If the user mentions "sanity check" or is working in a sanity-checks directory → `experiment_type = "sanity_check"`
   - Otherwise → `experiment_type = "experiment"`
3. Derive the experiment directory:
   - **Experiments**: `{scratch_dir}/ck-experiments/{experiment_name}/`
   - **Sanity checks**: `{scratch_dir}/ck-sanity-checks/{experiment_name}/`
4. Derive the output directory:
   - `{scratch_dir}/ck-outputs/{experiment_name}/`

### Directory Structure

- **Experiments** (research tasks): `{scratch_dir}/ck-experiments/{experiment_name}/`
- **Sanity checks** (simple fine-tuning verification): `{scratch_dir}/ck-sanity-checks/{sanity_check_name}/`

**Outputs are automatically grouped:**
- Output directory: `{scratch_dir}/ck-outputs/{experiment_or_sanity_check_name}/ck-out-{run_name}/`

### Confirm with User

**Are you working on a sanity check or a research experiment?**
- Log the detected path for user confirmation
- Note that outputs will be grouped under the same name in ck-outputs/

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

### Workflow Architecture

The experiment workflow uses an **orchestrator → worker** pattern:

- **Scaffolding (current):** `scaffold-experiment` reads experiment_summary.yaml and launches:
  - `scaffold-torchtune` agent → generates torchtune configs
  - `scaffold-inspect` agent → generates inspect-ai configs

- **Execution (planned):** `run-experiment` will launch:
  - `run-torchtune` agent → submit and monitor fine-tuning jobs
  - `run-inspect` agent → submit and monitor evaluation jobs (after training)

**Why document tools:** Orchestrators use tool specifications to route to the correct worker agents. This architecture enables future support for multiple tool options (e.g., axolotl, lm-eval-harness).

**Note:** While these are currently the only options, explicitly confirming and documenting tool choices now will make it easier to support multiple tools in future iterations. These will be documented in the `tools` section of experiment_summary.yaml.

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
- Prompt with {input} placeholder (default: "{input}\n"; e.g., "Capitalize: {input}\n")

### Available Hyperparameters for Torchtune

When designing experiments, you can vary any of these parameters. Add varied parameters to `variables` and constant parameters to `controls` in experiment_summary.yaml.

**Core Training Parameters:**
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `lora_rank` | LoRA adapter rank (higher = more capacity, more memory) | 4, 8, 16, 32, 64 |
| `lr` | Learning rate | 1e-5, 5e-5, 1e-4, 3e-4 |
| `batch_size` | Batch size per GPU | 1, 2, 4, 8 |
| `epochs` | Number of training epochs | 1, 2, 3 |

**Additional Training Parameters:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `gradient_accumulation_steps` | Effective batch = batch_size × this | 1 |
| `weight_decay` | Optimizer regularization | 0.01 |
| `lora_dropout` | Dropout for LoRA layers | 0.0 |
| `num_warmup_steps` | LR scheduler warmup steps | 100 |
| `max_seq_len` | Maximum sequence length | 2048 |

**GPU Allocation (small models only):**

For models that fit on a GPU partition smaller than a full GPU (1B models require ~20GB VRAM), ask:
> "Use a full dedicated GPU for complete utilization metrics? (Recommended for experiments, optional for sanity checks)"

- **Yes (default for experiments):** Uses the full-GPU constraint/partition from `claude.local.md` — guarantees a dedicated GPU with complete nvidia-smi metrics (GPU utilization, memory, power). This is the standard choice for experiments where compute observability matters.
- **No (acceptable for sanity checks):** May land on a shared or MIG-partitioned GPU where GPU utilization reports as `[N/A]`. Memory and power metrics are still available. Faster queue times.

If user says "yes" (default), add the constraint/partition values from `claude.local.md` SLURM Defaults to `experiment_summary.yaml` under the run's `slurm_overrides` section (e.g., `slurm_overrides: {constraint: "gpu80"}`). If "no", omit `slurm_overrides` — scaffold will not pass constraint/partition, and the SLURM lines stay commented out.

**Note:** Only ask this for models where `min_gpu_vram_gb` (from model_configs.py) is less than a full GPU's VRAM. Larger models always need full dedicated GPUs — read the constraint/partition from `claude.local.md` and include it automatically.

**Advanced settings (calculate from prior runs if available):**
- Batch sizes - estimate from GPU memory usage in prior runs
- Dataset packing - enabled by default, affects batch size
- For help estimating: check `{scratch_dir}/*/slurm-*.out` for similar runs
- **Consult past compute utilization analyses** - If previous experiments have `analysis/compute_metrics.json` or a compute section in `report.md`, use that data to inform time limits, memory allocations, and GPU resource requests for new runs

### Generate Runs List

Create the runs list in experiment_summary.yaml:
- For fine-tuned runs: Include `name`, `type: "fine-tuned"`, `model`, and `parameters` dict with varied values
- For control runs: Include `name`, `type: "control"`, `model`, and empty `parameters: {}`
- Run names should include model and varying parameter values (e.g., `Llama-3.2-1B-Instruct_rank4`)
- Parameters dict should only include values that vary across runs (e.g., `lora_rank: 4`)

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

### Scorer Selection

**Ask the user:** "Which scorers should be used for evaluation?"

See `references/scorers.md` for the full list of available scorers, their parameters, design-time considerations, and common combinations.

**Important:** Base models evaluate once per task (no epoch suffix), fine-tuned models evaluate per epoch.

### Create Evaluation Matrix

Generate the evaluation matrix in experiment_summary.yaml:
- For each run, specify which tasks to run and which epochs to evaluate
- Fine-tuned runs: Use `epochs: [0, 1]` list for which epochs to evaluate
- Control runs: Use `epochs: null` (no epoch suffix)
- Tasks list should reference task names defined in `evaluation.tasks`

### Visualization Labels (vis_label)

**Ask the user:** "How should runs be labeled in visualizations? The default is to use the full run name (e.g., `1B_1K`), but you can specify shorter labels."

Each matrix entry can include an optional `vis_label` field that controls how the run appears in inspect-viz plots. This becomes a suffix on the task name (e.g., `acs_employment_1K` instead of `acs_employment_1B_1K`).

```yaml
matrix:
  - run: 1B_1K
    vis_label: "1K"  # shorter label for visualization
    tasks: [acs_employment]
    epochs: [0, 1, 2]
```

**Default behavior:** If `vis_label` is not specified, scaffold-inspect uses the run name.

**When to customize:** Use shorter labels when:
- Run names are verbose (e.g., `Llama-3.2-1B-Instruct_rank4` → `rank4`)
- You want to group by a specific dimension (e.g., sample size only, not model size)

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
- Verify the actual filename matches what was planned
- Generated datasets often include parameters in filename (e.g., `words_7L_80P_1000.json`)
- Update your plan if filename differs from initial expectations

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

### Document Resources

Record verified resources in experiment_summary.yaml:
- `models.base`: List with `name`, `path`, `size_gb` for each model
- `data.training`: Include `path`, `label`, `format`, `size_kb`, and `splits` with train/validation/test counts
- `evaluation.tasks`: List with `name`, `script`, optional `dataset`, and `description` for each task

### Log All Verification Steps

All resource verification commands should be logged in `design-experiment.log` (see `logging.md`).

---

## Step 8: Get Approval

Before presenting the plan to the user, validate completeness using `validation.md`.

### Present Complete Plan

Show the user:
- Overview (experiment overview, total runs, scientific question)
- Summary of runs (number of fine-tuned runs, controls, varying parameters)
- Evaluation plan (tasks, epochs)

### Adjust if Needed

Based on user feedback:
- Modify configurations
- Add/remove runs
- Change evaluation strategy

### After Approval
Proceed to `experiment_generation.md` to create the files.

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
```

### Before Approval
```
Here's the complete experiment plan:

**Overview:**
- X fine-tuning runs (varying [factors])
- Y evaluation tasks
- Z total evaluations

Does this look correct? Any adjustments needed?
```

---

## Key Principles

1. **Ask, don't assume** - Even when you can auto-detect, confirm with user
2. **Log everything** - All verifications go in design-experiment.log (see `logging.md`)
3. **Validate before presenting** - Use `validation.md` to ensure plan is complete
4. **Handle missing resources gracefully** - Note as prerequisites, don't block the plan
5. **System prompt consistency** - Critical for inspect-ai, verify it matches between training and eval

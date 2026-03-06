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

**Advanced settings (informed by prior runs when available):**
- **Batch sizes** - Use GPU memory data from Step 4b to recommend safe batch sizes (see batch size estimation below)
- Dataset packing - enabled by default, affects batch size
- If no prior data is available, start conservative (batch_size=4 for 1B, 2 for 3B, 1 for 8B+)

### Generate Runs List

Create the runs list in experiment_summary.yaml:
- For fine-tuned runs: Include `name`, `type: "fine-tuned"`, `model`, and `parameters` dict with varied values
- For control runs: Include `name`, `type: "control"`, `model`, and empty `parameters: {}`
- Run names should include model and varying parameter values (e.g., `Llama-3.2-1B-Instruct_rank4`)
- Parameters dict should only include values that vary across runs (e.g., `lora_rank: 4`)

### Step 4b: Compute Estimation (from prior runs)

Estimate SLURM time limits, GPU counts, and memory allocations using compute data from prior experiments. This step is **optional** — if no prior data is found, skip silently and omit `compute` blocks from the YAML (scaffold will use its defaults).

#### 1. Discovery

Search for `compute_metrics.json` files from past experiments:
- `{scratch_dir}/ck-experiments/*/analysis/compute_metrics.json`
- `{scratch_dir}/ck-sanity-checks/*/analysis/compute_metrics.json`

**Note:** `compute_metrics.json` lives in `{experiment_dir}/analysis/`, NOT in the output directory.

If none found, skip this entire step silently.

#### 2. Read & filter

Load all discovered `compute_metrics.json` files. For each, read the metadata envelope (model name, dataset size, epochs, job types). Separate jobs by `job_type`:
- `"finetune"` jobs → inform fine-tuning time estimates
- `"eval"` jobs → inform evaluation time estimates

**Selecting relevant data:** For each run in the new experiment, find the most relevant prior data:
1. **Same model** is the strongest match — prefer prior runs with the same model name
2. **Same model family/size** is next best (e.g., any 1B model for a 1B run)
3. **Different model size** can still inform estimates with parameter-count scaling

If multiple prior experiments are equally relevant (e.g., same model), average their wall times before scaling. If none are relevant (completely different model family, no overlap), skip compute estimation silently.

#### 3. Scale + reason (fine-tuning)

For each fine-tuning run in the new experiment:

1. Start with the best-matching prior finetune job's `wall_time` (or average if multiple are equally relevant)
2. Scale by dataset and epoch differences:
   ```
   scaled_time = prior_wall_time * (new_epochs / old_epochs) * (new_dataset_size / old_dataset_size)
   ```
3. Apply 1.5x safety margin: `estimated_time = scaled_time * 1.5`
4. Apply Claude judgment for additional factors:
   - Model size changes (larger model = slower per step)
   - Batch size differences (larger batch = fewer steps but more compute per step)
   - LoRA rank changes (higher rank = slightly more compute)
5. Round up to nearest 5-minute increment
6. Use prior job's GPU count and memory allocation as baseline (adjust for model size changes)

#### 3c. Batch size recommendation (from GPU memory)

When prior `compute_metrics.json` data is available, use GPU memory utilization to recommend a safe batch size for new runs. This addresses the common problem of OOM failures requiring manual batch size reduction and resubmission.

**Data needed from prior run:**
- `gpu_mem_used_mean_gb` and `gpu_mem_total_gb` from finetune jobs
- `batch_size` from the compute_metrics.json envelope
- Model name from the envelope

**Estimation logic:**

1. Calculate the prior run's memory utilization ratio:
   ```
   mem_ratio = gpu_mem_used_mean_gb / gpu_mem_total_gb
   ```

2. **Same model → same or larger batch size:**
   - If `mem_ratio < 0.7`: prior batch size has headroom; suggest increasing (e.g., double it)
   - If `0.7 ≤ mem_ratio < 0.9`: prior batch size is well-fitted; reuse it
   - If `mem_ratio ≥ 0.9`: prior batch size is near the limit; reuse but warn about OOM risk

3. **Scaling to a larger model:**
   - GPU memory for LoRA fine-tuning scales roughly with model parameter count
   - Estimate: `new_mem ≈ prior_mem * (new_model_params / prior_model_params)`
   - If `new_mem > 0.85 * gpu_mem_total_gb`: halve the batch size
   - If still too large: halve again (minimum batch_size = 1)
   - Example: 1B model used 12.5GB/80GB with batch_size=8 → 3B model (~3x params) would use ~37.5GB → batch_size=8 still fits. But 8B model (~8x params) would use ~100GB → reduce to batch_size=4 or 2.

4. **Present recommendation** alongside the compute estimates table:
   ```
   Batch size recommendation based on prior GPU memory usage:
   - Prior: Llama-3.2-1B-Instruct, batch_size=8, used 12.5/80.0 GB (16%)
   - Recommended for Llama-3.2-1B-Instruct: batch_size=8 (same model, headroom available)
   - Recommended for Llama-3.2-3B-Instruct: batch_size=4 (estimated ~37 GB usage)
   ```

5. **User confirmation:** Present as a suggestion, not a mandate. The user may override.

**If batch_size is not in the compute_metrics.json envelope** (older data without this field): skip batch size recommendation and fall back to the conservative defaults (batch_size=4 for 1B, 2 for 3B, 1 for 8B+).

#### 4. Scale + reason (evaluation)

Take the prior eval job's `wall_time` and apply the same scaling logic:
- Scale by dataset size differences
- Apply 1.5x safety margin
- Round up to nearest 5-minute increment
- Use prior eval job's GPU count and memory

#### 5. Present to user

Show estimated compute as a table for confirmation. Note which prior experiments informed the estimates (for transparency), but this is explanatory context, not configuration:

```
## Compute Estimates (informed by 2 prior experiments)

| Run | Time | GPUs | Memory |
|-----|------|------|--------|
| Llama-3.2-1B-Instruct_rank4 | 0:15:00 | 1 | 80G |
| Llama-3.2-1B-Instruct_rank8 | 0:15:00 | 1 | 80G |
| Eval jobs | 0:05:00 | 1 | 80G |

Based on: cap_4L_2025-10-22, cap_8L_2025-11-05
These estimates include a 1.5x safety margin. Adjust if needed.
```

#### 6. Write compute blocks

If confirmed, add `compute` blocks to:
- Each fine-tuned run in the `runs` section
- The `evaluation` section (shared across all eval jobs)

**Do not** include provenance metadata (which prior experiments informed the estimate) in the YAML. That information is logged in `design-experiment.log` (see `logging.md`).

Example:
```yaml
runs:
  - name: "Llama-3.2-1B-Instruct_rank4"
    type: "fine-tuned"
    model: "Llama-3.2-1B-Instruct"
    parameters:
      lora_rank: 4
    compute:
      time: "0:15:00"
      gpus: 1
      mem: "80G"

evaluation:
  # ... existing fields ...
  compute:
    time: "0:05:00"
    gpus: 1
    mem: "80G"
```

If the user declines estimates or no prior data exists, omit `compute` blocks entirely.

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

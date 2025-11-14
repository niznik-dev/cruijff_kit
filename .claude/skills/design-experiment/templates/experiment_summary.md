# Experiment Summary Template

Use this template structure when creating `experiment_summary.md` files.

## Required Sections (in order)

### 1. Overview
- Experiment type (sanity check or research experiment)
- Total number of runs
- Scientific question being addressed
- Created date

### 2. Tools
- Which preparation tool (torchtune)
- Which evaluation tool (inspect-ai)
- Brief purpose of each

### 3. Variables
- Table of factors and levels being tested
- Example: LoRA rank (4, 8, 16) × Learning rate (1e-5, 5e-5)

### 4. All Runs
- Complete table with columns:
  - Run Name
  - Model
  - Parameters that vary (LoRA rank, learning rate, etc.)
  - Type (Fine-tuned or Control)
  - Estimated time
- Include both fine-tuned runs and controls

### 5. Resources
- **Models:** Paths, verification status, sizes
- **Dataset:** Path, format, size, splits, verification status
- **Evaluation Tasks:** Table with task name, script path, dataset, description

### 6. Evaluation Plan
- Which tasks
- Which epochs to evaluate (0-indexed: epoch_0, epoch_1, etc.)
- Evaluation matrix if selective (runs × tasks)
- Base models vs fine-tuned models handling

### 7. Configuration
- **Training:** Recipe, epochs, batch size, LoRA rank, learning rate, system prompt
- **Evaluation:** Temperature, scorer, system prompt (must match training)
- **Output:** Checkpoint directory pattern, naming pattern, example path

### 8. Compute Estimates
- **Training:** Per-run time, total runs, total training time
- **Evaluation:** Per-eval time, total evals, total eval time
- **Disk Space:** Per-checkpoint size, total size, available space
- **Total GPU Hours:** Training + evaluation

### 9. Naming Conventions
- How runs are named
- How outputs are organized
- Examples

### 10. Quick Reference
- **Paths:** Experiment dir, models dir, dataset path
- **Common Commands:** Model listing, dataset check, prior run search, speed extraction, disk check
- **Next Steps:** Numbered list (scaffold → submit → monitor → evaluate → analyze)

## Template Example

```markdown
# {Experiment Name}

## Overview
- **Type:** {sanity_check or experiment}
- **Runs:** {N} fine-tuned + {M} controls = {N+M} total
- **Question:** {Scientific question}
- **Created:** {YYYY-MM-DD}

## Tools
- **Model Preparation:** torchtune
  - *Purpose:* Fine-tuning LLMs with LoRA
  - *Used by:* `scaffold-torchtune` and `run-torchtune` skills

- **Evaluation:** inspect-ai
  - *Purpose:* Evaluating LLMs on custom tasks
  - *Used by:* `scaffold-inspect` and `run-inspect` skills

## Variables
| Factor | Levels |
|--------|--------|
| {Factor1} | {Level1, Level2, Level3} |
| {Factor2} | {LevelA, LevelB} |

## All Runs
| Run Name | Model | {Varying Param 1} | {Varying Param 2} | Type | Est. Time |
|----------|-------|-------------------|-------------------|------|-----------|
| {run_1} | {model} | {value} | {value} | Fine-tuned | ~{X}min |
| {run_2} | {model} | {value} | {value} | Fine-tuned | ~{X}min |
| {run_base} | {model} | - | - | Control | N/A |

## Resources

### Models
- **{Model Name}**: `{path}`
  - Verified: ✓ ({date})
  - Size: ~{X} GB

### Dataset
- **Path**: `{dataset_path}`
- **Format:** {JSON/CSV/etc}
- **Size:** {XXX}KB
- **Splits:** train ({N} samples), validation ({M} samples), test ({K} samples)
- **Verified:** ✓ ({date})

### Evaluation Tasks
| Task Name | Script | Dataset | Description |
|-----------|--------|---------|-------------|
| {task1} | `{script_path}` | {dataset_name} | {description} |

## Evaluation Plan

{If all runs evaluated the same:}
All runs will be evaluated on {task1, task2} at {epoch selection}.

{If selective evaluation:}
### Evaluation Matrix
| Run Name | {task1} | {task2} | Notes |
|----------|---------|---------|-------|
| {run1} | ✓ epoch X,Y | ✓ epoch X,Y | {notes} |

## Configuration

### Training
- **Recipe:** {recipe_name}
- **Epochs:** {N}
- **Batch size:** {B}
- **LoRA rank:** {R}
- **Learning rate:** {LR}
- **System prompt:** "{prompt}" or "" (blank)

### Evaluation
- **Temperature:** {T}
- **Scorer:** {scorer_type}
- **System prompt:** "{prompt}" or "" (blank - must match training)

### Output
- **Checkpoint directory:** `{output_dir_base}`
- **Naming pattern:** `ck-out-{run_name}/epoch_{N}`
- **Example:** `{output_dir_base}/ck-out-{example_run}/epoch_0`

## Compute Estimates

### Training
- **Per-run time:** ~{X} minutes/epoch
- **Total runs:** {N} fine-tuned runs
- **Total training time:** ~{Y} minutes ({calculation})

### Evaluation
- **Per-eval time:** ~{X} minutes
- **Total evals:** {N} ({runs} × {tasks} × {epochs})
- **Total eval time:** ~{Y} minutes

### Disk Space
- **Per-epoch checkpoint:** ~{X} GiB
- **Total checkpoints:** ~{Y} GiB ({calculation})
- **Available space:** {Z}T

### Total GPU Hours
- **Training:** ~{X} hours
- **Evaluation:** ~{Y} hours
- **Total:** ~{Z} GPU hours

## Naming Conventions
{Describe naming pattern and provide examples}

## Quick Reference

**Paths:**
- Experiment: `{experiment_dir}`
- Models: `{models_dir}/{model_names}`
- Dataset: `{dataset_path}`

**Common Commands:**
- List models: `ls {models_dir}`
- Check dataset: `ls -lh {dataset_path}`
- Find prior runs: `find {scratch_dir} -name "slurm-*.out" -path "*/ck-out-*" | head -10`
- Extract speed: `grep -E "[0-9.]+it/s" {prior_run}/slurm-*.out | tail -20`
- Check disk: `df -h {scratch_dir}`

**Next Steps:**
1. Generate configs: Run `scaffold-experiment`
2. Submit jobs: Run `run-experiment` (or manual submission)
3. Monitor progress
4. Evaluate results
5. Analyze: Run `analyze-experiment` (when available)
```

## Important Notes

### Path Placeholders
All paths should use actual values from `claude.local.md`, not placeholders like `{models_dir}`.

### System Prompt Consistency
**Critical for inspect-ai:** System prompt must be identical between training and evaluation. Document explicitly, even if blank ("").

### Epoch Indexing
Always use 0-indexed epochs: epoch_0, epoch_1, etc. Document clearly in Evaluation Plan.

### Base vs Fine-tuned
- Base models: Evaluate once per task (no epoch suffix)
- Fine-tuned models: Evaluate per specified epoch (epoch_0, epoch_1, etc.)

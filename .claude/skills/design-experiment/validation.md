# Validation

Before presenting the plan to user (step 8 of param_selection workflow), validate that it's complete and feasible.

## Complete Checklist

Run through this checklist before presenting the plan:

### Experiment Metadata Validation
- ✓ Experiment name is descriptive
- ✓ Scientific question is clearly stated
- ✓ Date is in YYYY-MM-DD format
- ✓ Experiment directory path is specified

### Tools Validation
- ✓ Preparation tool specified ("torchtune")
- ✓ Evaluation tool specified ("inspect-ai")

### Variables and Controls Validation
- ✓ Variables section lists all parameters that vary (or omitted if none)
- ✓ Controls section lists all constant hyperparameters
- ✓ System prompt is defined in controls (even if blank "")
- ✓ Prompt is defined in controls with {input} placeholder
- ✓ Epochs, batch size, GPUs specified
- ✓ LoRA parameters specified if not varied

### Training Steps Validation
- ✓ Compute: `steps_per_epoch = ceil(training_samples / (batch_size * gradient_accumulation_steps))`
- ✓ Compute: `total_steps = steps_per_epoch * epochs`
- ✓ Warn if `total_steps < num_warmup_steps` (warmup never completes)
- ✓ Warn if `total_steps < 50` (too few steps for meaningful training)
- ✓ If either warning triggers, flag for user and suggest reducing batch size/gradient accumulation or increasing epochs

### Runs Validation
- ✓ All run names follow established convention
- ✓ Names include model and varying parameters
- ✓ Control runs clearly marked with "base" in name
- ✓ All fine-tuned runs have `type: "fine-tuned"`
- ✓ All control runs have `type: "control"`
- ✓ Fine-tuned runs have parameters dict with varied values
- ✓ Control runs have empty parameters dict `{}`
- ✓ All runs specify correct model name (matches models.base[].name)
- ✓ Control runs included if requested

### Models and Data Validation
- ✓ All models have name, path, and size_gb
- ✓ Model paths exist (verified and logged)
- ✓ Training data has path, label, format, size_kb
- ✓ Dataset file exists (verified and logged)
- ✓ Splits section has train, validation, test counts
- ✓ Format is "json" or "parquet"

### Output Configuration Validation
- ✓ Base directory specified
- ✓ Checkpoint pattern specified (contains {run_name} placeholder)
- ✓ Wandb project name specified

### Evaluation Validation
- ✓ System prompt matches training system prompt (controls.system_prompt)
- ✓ **CRITICAL:** System prompt consistency is essential for inspect-ai
- ✓ Temperature specified (typically 0.0)
- ✓ Scorer specified and appropriate for task
- ✓ All evaluation tasks listed with name, script, description
- ✓ Task scripts exist OR missing scripts noted as prerequisites
- ✓ Evaluation matrix includes all runs
- ✓ Fine-tuned runs specify epochs list (e.g., `[0, 1]`)
- ✓ Control runs use `epochs: null`
- ✓ Epochs are 0-indexed in matrix
- ✓ Task names in matrix match tasks defined in evaluation.tasks

### Resources Validation
- ✓ All verifications logged in design-experiment.log
- ✓ Available disk space checked
- ✓ Sufficient space for checkpoints
- ✓ Warning issued if disk space is tight

### YAML Structure Validation
- ✓ All required top-level sections present (experiment, tools, controls, models, data, output, runs, evaluation)
- ✓ Proper nesting and indentation
- ✓ Lists use proper YAML syntax
- ✓ No placeholder values (use actual paths from claude.local.md)

---

## Common Issues to Check

### System Prompt Mismatch
**Problem:** Training uses one system prompt, evaluation uses another (or blank)
**Impact:** inspect-ai evaluations may fail or give invalid results
**Fix:** Ensure `evaluation.system_prompt` exactly matches `controls.system_prompt`

**How it works:** scaffold-inspect generates eval scripts that set `CONFIG_PATH` to point at `setup_finetune.yaml`. The inspect-ai task reads the system prompt from this config, ensuring train/eval parity automatically. However, the experiment_summary.yaml must still document both for validation purposes.

### 1-Indexed Epochs
**Problem:** Documenting epochs as [1, 2, 3] instead of [0, 1, 2]
**Impact:** Evaluation scripts will fail or evaluate wrong checkpoints
**Fix:** Always use 0-indexed: [0, 1, 2]

### Base Model Epochs
**Problem:** Assigning epoch list to base model evaluations
**Impact:** Base models don't have epochs - will cause errors
**Fix:** Base models use `epochs: null` in evaluation matrix

### Missing Prerequisites
**Problem:** Evaluation task scripts don't exist yet
**Impact:** scaffold-experiment will fail
**Fix:** Don't block - clearly document as prerequisite with instructions

### Too Few Training Steps
**Problem:** Large effective batch size (batch_size * gradient_accumulation_steps) relative to dataset size collapses total training steps
**Impact:** Warmup never completes (e.g., 100 warmup steps > 14 total steps), model barely trains
**Formula:** `total_steps = ceil(training_samples / (batch_size * gradient_accumulation_steps)) * epochs`
**Fix:** Reduce batch_size or gradient_accumulation_steps, or increase epochs. Flag if total_steps < 50 or < num_warmup_steps.

### Unrealistic Batch Sizes
**Problem:** Batch size too large for GPU memory
**Impact:** Training jobs will crash with OOM errors
**Fix:** Check prior runs or start conservative (batch_size=4 for 1B, 2 for 3B)

### Empty Parameters Dict
**Problem:** Control runs have parameters with values instead of empty dict
**Impact:** Downstream parsing may incorrectly treat control as fine-tuned
**Fix:** Control runs must have `parameters: {}`

---

## If Validation Fails

### Missing Information
Don't present plan - go back to `param_selection.md` to gather missing details

### Inconsistencies
Fix them before presenting, or clearly flag for user approval

### Missing Resources
Note as prerequisites - don't block the entire plan

**Example prerequisite note for user:**
```
Before running scaffold-experiment, you must:

1. Create evaluation task: Run create-inspect-task to create task script
2. Download model: (if model missing) Use appropriate download tool

Once prerequisites are complete, you can proceed with scaffolding.
```

---

## After Validation Passes

Proceed to step 8 of `param_selection.md` to present the complete plan to user for approval.

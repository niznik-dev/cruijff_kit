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
- ✓ Warn if `total_steps < 3 * num_warmup_steps` (most of training spent in LR warmup; matches setup_finetune.py's scaffold-time check)
- ✓ Warn if `total_steps < 50` (too few steps for meaningful training)
- ✓ If either warning triggers, flag for user and suggest reducing batch size/gradient accumulation or increasing epochs

### Runs Validation
- ✓ All run names follow established convention
- ✓ Names include model and varying parameters
- ✓ Control runs clearly marked with "base" in name
- ✓ Every run's `type` is one of `"fine-tuned"`, `"control"`, or `"eval-only"`
- ✓ All fine-tuned runs have `type: "fine-tuned"`
- ✓ All control runs have `type: "control"`
- ✓ All eval-only runs have `type: "eval-only"` and a `parameters.checkpoint_path` pointing at a pre-existing checkpoint directory
- ✓ Fine-tuned runs have parameters dict with varied values
- ✓ Control runs have empty parameters dict `{}`
- ✓ Eval-only and control runs use `epochs: null` in the evaluation matrix (no training epochs)
- ✓ All runs specify correct model name (matches models.base[].name)
- ✓ Control runs included if requested

### Models and Data Validation
- ✓ All models have name, path, and size_gb
- ✓ Model paths exist (verified and logged)
- ✓ Training data has path, dataset_label, format, size_kb
- ✓ `controls.dataset_type` is present and is exactly `"chat_completion"` or `"text_completion"` (REQUIRED — read by torchtune and drives eval's chat-template choice; a missing or wrong value silently corrupts evaluation, so treat absence as a hard validation failure, not a default-and-continue)
- ✓ Dataset file exists (verified and logged)
- ✓ Splits section has train, validation, test counts
- ✓ Format is "json"

### Output Configuration Validation
- ✓ Wandb project name specified

### Evaluation Validation
- ✓ `controls.system_prompt` is set — the single source for training and eval (propagated to both, so parity is automatic; there is no separate `evaluation.system_prompt` to match). Per-task variations live at `evaluation.tasks[].system_prompt`.
- ✓ Temperature specified (typically 0.0)
- ✓ Scorer specified and appropriate for task
- ✓ All evaluation tasks listed with name, script, description
- ✓ Task scripts exist OR missing scripts noted as prerequisites
- ✓ Evaluation matrix includes all runs
- ✓ Fine-tuned runs specify epochs list (e.g., `[0, 1]`)
- ✓ Control runs use `epochs: null`
- ✓ Epochs are 0-indexed in matrix
- ✓ Task names in matrix match tasks defined in evaluation.tasks
- ✓ Optional per-task overrides (`prompt`, `system_prompt`, `assistant_prefix`) are strings if present
- ✓ If `prompt` varies (per-task `evaluation.tasks[].prompt` or per-run `runs[].parameters.prompt`), each value contains the `{input}` placeholder

### Resources Validation
- ✓ All verifications logged in design-experiment.log
- ✓ Available disk space checked
- ✓ Sufficient space for checkpoints
- ✓ Warning issued if disk space is tight

### Compute Estimates Validation (if present)
- ✓ If any `runs[].compute` block exists, verify:
  - `time` is in HH:MM:SS format (e.g., "0:15:00")
  - `gpus` is a positive integer
  - `mem` matches pattern like "80G" (number followed by unit)
- ✓ If `evaluation.compute` block exists, apply same checks
- ✓ Compute blocks are optional — their absence is NOT an error

### YAML Structure Validation
- ✓ All required top-level sections present (experiment, tools, controls, models, data, output, runs, evaluation)
- ✓ Proper nesting and indentation
- ✓ Lists use proper YAML syntax
- ✓ No placeholder values (use actual paths from claude.local.md)

---

## Common Issues to Check

### System Prompt (single source — mismatch prevented by construction)
**Design:** the system prompt has one home, `controls.system_prompt`. It is propagated to both training and eval (`eval.yaml`) by `propagate.py`, so train/eval parity is guaranteed — there is no separate `evaluation.system_prompt` that could drift out of sync.
**Per-task variation:** when a task legitimately needs a different prompt (e.g. a cue vs. no-cue ablation), set `evaluation.tasks[].system_prompt`; the per-task override beats the propagated default for that cell only.
**Check:** confirm `controls.system_prompt` is present and correct. A stray top-level `evaluation.system_prompt` is not read — remove it.

### Prompt Variation (per-task vs per-run)
**Design:** the user prompt has one experiment-wide home, `controls.prompt`, propagated to both training and eval. Two override surfaces let it vary:
- **Per-task** `evaluation.tasks[].prompt` — the eval-only prompt-sweep lever (N tasks, one script, a different appended sentence each). The canonical zero-finetuning prompt-engineering pattern.
- **Per-run** `runs[].parameters.prompt` — a run trains *and* evaluates on its own prompt (flows to `setup_finetune.yaml` and the run's eval cells). A per-task prompt still wins over it for that cell.

**Check:** every prompt (controls + any override) keeps the `{input}` placeholder. When prompt is the swept axis across runs, give runs short meaningful names — don't fold the prompt string into a directory name.

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

# Validation

Before presenting the plan to user (step 9 of param_selection workflow), validate that it's complete and feasible.

## Complete Checklist

Run through this checklist before presenting the plan:

### Preparation Validation
- ✓ All run names follow established convention
- ✓ Names include varying parameters
- ✓ Control runs clearly marked as "base"
- ✓ All fine-tuned runs have specified: model, LoRA rank, learning rate, batch size, epochs
- ✓ Control runs properly marked with `-` for non-applicable parameters
- ✓ All runs use compatible configurations
- ✓ System prompt is defined (even if blank "")
- ✓ Dataset splits are appropriate for validation settings
- ✓ Batch sizes are appropriate for GPU memory
- ✓ Epochs are reasonable for the task
- ✓ Time estimates are calculated (actual or preliminary)
- ✓ Control runs included if requested

### Evaluation Validation
- ✓ All requested evaluation tasks are listed
- ✓ Task scripts exist (or noted as prerequisites)
- ✓ Evaluation datasets are defined
- ✓ Epochs are 0-indexed (epoch_0, epoch_1, etc.)
- ✓ Epoch selection is appropriate (last epoch, all epochs, or specific)
- ✓ Base models don't have epoch suffixes
- ✓ Fine-tuned models specify which epochs to evaluate
- ✓ All runs have evaluation assignments
- ✓ Selective evaluations are documented clearly in matrix
- ✓ **CRITICAL:** System prompt matches between training and evaluation
- ✓ Temperature is specified (typically 0.0 for deterministic eval)
- ✓ Scorer is appropriate for the task
- ✓ Evaluation datasets are appropriate (same as training, or different test set)

### Resources Validation
- ✓ All model directories exist (verified and logged)
- ✓ Paths logged in design-experiment.log
- ✓ Approximate sizes noted
- ✓ Dataset file exists (verified and logged)
- ✓ Dataset size checked and logged
- ✓ Required splits present (train, validation if needed, test if needed)
- ✓ Format appropriate for torchtune
- ✓ All inspect-ai task scripts exist OR missing scripts noted as prerequisites
- ✓ If missing, user knows to run `create-inspect-task` first
- ✓ Available disk space checked
- ✓ Sufficient space for estimated checkpoint sizes
- ✓ Warning issued if disk space is tight

---

## Common Issues to Check

### System Prompt Mismatch
**Problem:** Training uses one system prompt, evaluation uses another (or blank)
**Impact:** inspect-ai evaluations may fail or give invalid results
**Fix:** Ensure system prompt is identical in both Configuration sections

### 1-Indexed Epochs
**Problem:** Documenting epochs as 1, 2, 3 instead of 0, 1, 2
**Impact:** Job scripts will fail or evaluate wrong checkpoints
**Fix:** Always use 0-indexed: epoch_0, epoch_1, epoch_2

### Base Model Epochs
**Problem:** Assigning epoch suffixes to base model evaluations
**Impact:** Base models don't have epochs - will cause errors
**Fix:** Base models evaluate once per task (no epoch suffix)

### Missing Prerequisites
**Problem:** Evaluation task scripts don't exist yet
**Impact:** scaffold-experiment will fail
**Fix:** Don't block - clearly document as prerequisite with instructions

### Unrealistic Batch Sizes
**Problem:** Batch size too large for GPU memory
**Impact:** Training jobs will crash with OOM errors
**Fix:** Check prior runs or start conservative (batch_size=4 for 1B, 2 for 3B)

### Missing Estimates
**Problem:** No time or disk space estimates provided
**Impact:** User can't plan resources
**Fix:** Provide at least preliminary estimates, mark as "verify with test run"

---

## If Validation Fails

### Missing Information
Don't present plan - go back to `param_selection.md` to gather missing details

### Inconsistencies
Fix them before presenting, or clearly flag for user approval

### Missing Resources
Document as prerequisites in experiment_summary.md - don't block the entire plan

**Example:**
```markdown
## Prerequisites

Before running `scaffold-experiment`, you must:

1. **Create evaluation task:** Run `create-inspect-task` to create task script
2. **Download model:** (if model missing) Use appropriate download tool

Once prerequisites are complete, you can proceed with scaffolding.
```

---

## After Validation Passes

Proceed to step 9 of `param_selection.md` to present the complete plan to user for approval.

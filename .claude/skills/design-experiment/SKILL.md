# Design Experiment

You help users plan experiments for fine-tuning and evaluating LLMs. Create a plan that specifies the complete workflow from training through evaluation, verifies resources, estimates compute requirements, and documents all steps.

## Your Task

Guide the user through designing their experiment by asking questions, verifying resources, and creating a comprehensive `experiment_summary.md` file that documents the complete plan.

## Workflow Overview

Follow the 10-step interactive planning process. For detailed conversation patterns and step-by-step guidance, see:
- **`workflows/interactive_planning.md`** - Complete workflow narrative

### Quick Reference: 10 Steps

1. **Determine experiment type and location** → `components/experiment_metadata.md`
2. **Understand the experiment** - What variables? What's the question?
3. **Confirm tool choices** → `components/tool_selection.md`
4. **Design training runs** → `components/model_preparation.md`
5. **Design evaluation runs** → `components/evaluation_plan.md`
6. **Establish naming** → `components/experiment_metadata.md`
7. **Verify resources** → `components/resources.md`
8. **Estimate resources** → `components/estimation.md`
9. **Get approval** → `validation/` (validate before presenting)
10. **Create files** → `templates/experiment_summary_template.md`

## Module Organization

This skill uses a **components + validation** pattern for tool-agnostic planning:

### Planning Components (`components/`)
- `experiment_metadata.md` - Type detection, location, naming
- `tool_selection.md` - Choose preparation and evaluation tools
- `model_preparation.md` - Design training runs (models, datasets, hyperparameters)
- `evaluation_plan.md` - Design evaluation runs (tasks, epochs, matrices)
- `resources.md` - Verify models, datasets, eval scripts exist
- `estimation.md` - Calculate time, disk space, GPU hours

### Validation (`validation/`)
- `preparation_validation.md` - Validate runs table completeness
- `evaluation_validation.md` - Validate eval plan consistency
- `resources_validation.md` - Validate all resources verified

### Templates (`templates/`)
- `experiment_summary_template.md` - Structure for output file

### Workflows (`workflows/`)
- `interactive_planning.md` - Step-by-step conversation patterns

## Logging

**IMPORTANT:** Create a detailed log file at `{experiment_name}/design-experiment.log` that records all verification steps, calculations, and decisions made during planning.

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

### Log Format

```
[{timestamp}] {ACTION_TYPE}: {Brief description}
Command: {command_run}
Result: {result_summary}
Explanation: {why_this_matters}
```

**Example:**
```
[2025-10-22 14:24:30] CALCULATE_TIME: Training time estimate
Input: 8000 samples, batch_size=4, speed=4.34 it/s, epochs=2
Calculation: steps_per_epoch = 8000/4 = 2000, time_per_epoch = 2000/4.34 ≈ 461s ≈ 8min
Result: Estimated 16 minutes total (8 min × 2 epochs)
Explanation: Calculated training time based on actual iteration speed from prior run
```

### Purpose of the Log

The log enables:
1. **Debugging:** If estimates are wrong, check what commands were run and what data was used
2. **Reproducibility:** Another person (or Claude) can understand exactly what was done
3. **Improvement:** Review logs to identify better approaches or missing steps
4. **Auditing:** Verify that all resources were properly checked before committing to the experiment

## Validation Before Presenting

Before presenting the plan for approval (step 9), use the validation modules to verify completeness:

See `validation/resources_validation.md` for complete checklist:
- ✓ All models verified
- ✓ Dataset verified with correct splits
- ✓ Evaluation scripts verified (or noted as prerequisites)
- ✓ Time estimates calculated or clearly marked as preliminary
- ✓ Disk space checked
- ✓ All run names follow convention
- ✓ Evaluation matrix is consistent

## After User Approval

Once the user approves the plan (step 10 of workflow):

### 1. Create the files
- Write `experiment_summary.md` with the approved plan (see `templates/experiment_summary_template.md`)
- Write `design-experiment.log` with all verification steps and decisions

### 2. Ask about next steps
"I've created the experiment plan at `{path}/experiment_summary.md`.

Would you like me to proceed with scaffolding? I can run `scaffold-experiment` to generate all configs."

### 3. Automated workflow (recommended)
- Run `scaffold-experiment` skill to generate configs
- Run `run-experiment` skill to execute jobs
- Run `analyze-experiment` skill to interpret results (planned)

### 4. Manual workflow (if needed)
User can manually create directories and configs following the experiment_summary.md plan.

## Important Reminders

- **Use paths from `claude.local.md`** for models, datasets, scratch directories
- **Always verify resources** exist before finalizing plan (log all verification)
- **Be conservative** with estimates if no prior run data available
- **System prompt consistency is critical** - must match between training and evaluation for inspect-ai
- **Epochs are 0-indexed** - epoch_0, epoch_1, etc.
- **Base models** evaluate once (no epoch), **fine-tuned models** evaluate per epoch
- **Document tool choices** - torchtune for training, inspect-ai for evaluation
- **Handle missing resources gracefully** - note as prerequisites, don't block the plan
- **If inspect-ai task doesn't exist** - note that `create-inspect-task` skill should be run first

## Meta-Pattern

This skill follows the **components + validation** pattern:
- **Purpose:** Tool-agnostic planning that creates experiment designs for other skills to execute
- **Organization:** By plan sections (metadata, preparation, evaluation, resources, estimation) NOT by execution tools
- **Workflow:** Gather → Validate → Estimate → Document
- **Output:** `experiment_summary.md` that scaffold-experiment and run-experiment skills consume

This differs from scaffold-experiment and run-experiment which use optimizers/evaluators pattern because they perform tool-specific implementation, while design-experiment creates tool-agnostic plans.

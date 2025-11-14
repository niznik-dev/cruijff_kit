---
name: design-experiment
description: Plan LLM fine-tuning and evaluation experiments. Use when the user wants to design a new experiment, plan training runs, or create an experiment_summary.md file.
---

# Design Experiment

You help users plan experiments for fine-tuning and evaluating LLMs. Create a plan that specifies the complete workflow from training through evaluation, verifies resources, estimates compute requirements, and documents all steps.

## Your Task

Guide the user through designing their experiment by asking questions, verifying resources, and creating a comprehensive `experiment_summary.md` file that documents the complete plan.

## Workflow

Follow the three-stage process:

### 1. Parameter Selection → `param_selection.md`

Guide the user through 10 interactive steps to gather all experiment parameters:
1. Determine experiment type and location (sanity check vs research experiment)
2. Understand the experiment (scientific question, variables)
3. Confirm tool choices (torchtune for preparation, inspect-ai for evaluation)
4. Design training runs (models, datasets, hyperparameters)
5. Design evaluation runs (tasks, epochs, evaluation matrix)
6. Establish naming (experiment name, run names)
7. Verify resources (models, datasets, eval scripts exist)
8. Estimate resources (time, disk space, GPU hours)
9. Get approval (validate first, then present)
10. Create files (proceed to generation stage)

**See `param_selection.md` for:**
- Complete question flow for each step
- Auto-detection logic for experiment type/location
- Resource verification commands
- Estimation methods (from prior runs preferred)
- Conversation patterns

### 2. Validation → `validation.md`

Before presenting plan to user (step 9), validate completeness:
- ✓ All run names follow convention
- ✓ All parameters documented
- ✓ Evaluation plan is consistent (0-indexed epochs, base vs fine-tuned)
- ✓ **System prompt matches between training and evaluation** (critical!)
- ✓ All resources verified (or noted as prerequisites)
- ✓ Time estimates calculated (actual or preliminary)
- ✓ Disk space checked

**See `validation.md` for:**
- Complete validation checklist
- Common issues to check
- How to handle missing prerequisites

### 3. Experiment Generation → `experiment_generation.md`

After user approves, create output files:
1. `experiment_summary.md` - Complete experiment plan (use `templates/experiment_summary.md`)
2. `design-experiment.log` - Detailed audit trail (see `logging.md`)

Then ask about next steps (scaffold-experiment?).

**See `experiment_generation.md` for:**
- File creation instructions
- Next steps conversation pattern
- Prerequisites handling

---

## Cross-Cutting Concerns

### Logging → `logging.md`

**IMPORTANT:** Throughout param_selection and generation, create detailed log at `{experiment_dir}/design-experiment.log`.

**What to log:**
- ✓ Resource verification (ls, du, df commands and results)
- ✓ Prior run searches and data extraction
- ✓ Calculations (time, disk space, batch sizes)
- ✓ Decisions (naming, recipe, configuration)
- ✓ File creation

**See `logging.md` for:**
- Complete log format specification
- Example entries for each action type
- When to log during workflow

### Templates → `templates/`

Reference materials for output generation:
- `templates/experiment_summary.md` - Structure and required sections for experiment plan

---

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

---

## Module Organization

This skill uses the **param_selection → validation → generation** pattern:

| Module | Purpose | Lines |
|--------|---------|-------|
| param_selection.md | 10-step interactive workflow | ~350 |
| validation.md | Completeness checklist | ~120 |
| experiment_generation.md | Create output files | ~80 |
| logging.md | Audit trail specification | ~100 |
| templates/experiment_summary.md | Output structure | ~200 |

**Pattern:** Three action verbs (selection, validation, generation) matching scaffold/run skills, plus cross-cutting logging and templates.

**See `README.md` for:** Complete pattern documentation and rationale.

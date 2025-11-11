# Experiment Generation

After user approves the plan, create the output files and suggest next steps.

## Files to Create

### 1. experiment_summary.md

Create comprehensive documentation in `{experiment_dir}/experiment_summary.md` using the template structure from `templates/experiment_summary.md`.

**Required sections (in order):**
1. **Overview** - Experiment type, total runs, scientific question, created date
2. **Tools** - Which preparation and evaluation tools are used
3. **Variables** - Table of factors and levels being tested
4. **All Runs** - Complete table with run names, configurations, estimated time
5. **Resources** - Verified paths to models, datasets, eval scripts
6. **Evaluation Plan** - Which tasks, which runs, which epochs
7. **Configuration** - Recipe, epochs, batch sizes, hyperparameters, system prompt
8. **Compute Estimates** - Training time, eval time, disk space, GPU hours
9. **Naming Conventions** - How runs are named and organized
10. **Quick Reference** - Paths, common commands, next steps

**Important notes:**
- Use actual paths from `claude.local.md`, not placeholders
- **System prompt consistency is critical** - must match between training and evaluation for inspect-ai
- **Epochs are 0-indexed** - epoch_0, epoch_1, etc.
- Base models evaluate once (no epoch suffix), fine-tuned models evaluate per specified epoch

For complete template structure and examples, see `templates/experiment_summary.md`.

### 2. design-experiment.log

Create detailed audit trail in `{experiment_dir}/design-experiment.log` that records all verification steps, calculations, and decisions.

For log format and examples, see `logging.md`.

**Key logging points:**
- Resource verification commands and results
- Prior run searches and data extraction
- All calculations (time estimates, batch sizes, disk space)
- Decisions made (naming, recipe selection, configuration)
- File creation

---

## After Creation

### 1. Confirm Files Created

```
I've created the experiment plan at `{experiment_dir}/experiment_summary.md`.

All verification steps and calculations have been logged in `{experiment_dir}/design-experiment.log`.
```

### 2. Ask About Next Steps

```
Would you like me to proceed with scaffolding? I can run `scaffold-experiment` to generate all configs.
```

### 3. Explain Workflow Options

**Automated workflow (recommended):**
- Run `scaffold-experiment` skill to generate:
  - Fine-tuning configs via `scaffold-torchtune` (finetune.yaml, finetune.slurm)
  - Evaluation configs via `scaffold-inspect` (inspect.slurm, task scripts)
- Run `run-experiment` skill to execute:
  - Fine-tuning via `run-torchtune` (submit jobs, monitor progress)
  - Evaluation via `run-inspect` (submit jobs after training completes, monitor progress)
- Run `analyze-experiment` skill to interpret results (planned)

**Manual workflow (if needed):**
- User can manually create directories and configs
- Follow the experiment plan as documented in experiment_summary.md

---

## Conversation Pattern

```
Perfect! I've created:
- experiment_summary.md with the complete plan
- design-experiment.log with all verification steps and calculations

Would you like me to proceed with scaffolding? I can run `scaffold-experiment` to generate all the configs and SLURM scripts for you.
```

---

## Prerequisites Handling

If some resources were missing during verification (e.g., evaluation task scripts don't exist yet):

```markdown
## Prerequisites

Before running `scaffold-experiment`, you must:

1. **Create evaluation task:** Run `create-inspect-task` to create the capitalization task script
   - Task name: capitalization
   - Expected location: `{repo_dir}/experiments/capitalization/cap_task.py`

Once prerequisites are complete, you can proceed with scaffolding.
```

Don't block the plan - document prerequisites clearly so user knows what to do next.

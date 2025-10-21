# Skill Separation: Changes Summary

**Date**: 2025-10-20
**Goal**: Clarify the distinct roles of setup-experiment-dirs, create-torchtune-config, and generate-slurm-script

---

## Rationale

**Why keep these skills separate?**
- **Modularity**: Each skill does one thing well
- **Framework flexibility**: Can swap torchtune for other fine-tuning frameworks
- **Clear dependencies**: Easy to understand what comes before/after
- **Reusability**: Can skip/repeat individual steps without affecting others

---

## Changes Made

### 1. setup-experiment-dirs (.claude/skills/setup-experiment-dirs/skill.md)

**SCOPE**: Creates ONLY directory structure (framework-agnostic)

**Changes:**
- ✓ Added clear scope statement at top
- ✓ Added workflow position (AFTER plan-runs, BEFORE framework configs)
- ✓ Clarified that it does NOT create config files
- ✓ Updated steps to read from runs_plan.md
- ✓ Added verification step to check directories were created
- ✓ Updated "Next Steps" to suggest create-torchtune-config (not claim to do it)
- ✓ Made it clear this is framework-agnostic (works with any fine-tuning tool)

**What it creates:**
```
experiment_dir/
├── run1/
│   └── evaluations/
├── run2/
│   └── evaluations/
└── README.md
```

**Does NOT create:**
- finetune.yaml (that's create-torchtune-config's job)
- finetune.slurm (that's generate-slurm-script's job)

---

### 2. create-torchtune-config (.claude/skills/create-torchtune-config/skill.md)

**SCOPE**: Creates ONLY torchtune configuration files (torchtune-specific)

**Changes:**
- ✓ Added clear scope statement (torchtune-specific, not framework-agnostic)
- ✓ Added workflow position (AFTER setup-experiment-dirs, BEFORE generate-slurm-script)
- ✓ Added prerequisites checklist
- ✓ Removed any mention of creating SLURM scripts
- ✓ Added step to verify directories exist before proceeding
- ✓ Added step to read and parse runs_plan.md
- ✓ Clarified it skips base model runs (no fine-tuning needed)
- ✓ Updated "Next Steps" to suggest generate-slurm-script

**What it creates:**
```
experiment_dir/
├── run1/
│   ├── finetune.yaml  ← CREATES THIS
│   └── evaluations/
├── run2/
│   ├── finetune.yaml  ← CREATES THIS
│   └── evaluations/
└── README.md
```

**Does NOT create:**
- Directories (that's setup-experiment-dirs's job)
- finetune.slurm (that's generate-slurm-script's job)

---

### 3. generate-slurm-script (.claude/skills/generate-slurm-script/skill.md)

**SCOPE**: Creates ONLY SLURM batch scripts (framework-agnostic)

**Changes:**
- ✓ Added clear scope statement at top
- ✓ Added workflow position (AFTER create-torchtune-config, BEFORE launch-runs)
- ✓ Added prerequisites checklist
- ✓ Clarified it reads finetune.yaml to determine resources
- ✓ Updated description to be cluster-agnostic (not just "della cluster")

**What it creates:**
```
experiment_dir/
├── run1/
│   ├── finetune.yaml
│   ├── finetune.slurm  ← CREATES THIS
│   └── evaluations/
├── run2/
│   ├── finetune.yaml
│   ├── finetune.slurm  ← CREATES THIS
│   └── evaluations/
└── README.md
```

**Does NOT create:**
- Directories (that's setup-experiment-dirs's job)
- finetune.yaml (that's create-torchtune-config's job)

---

### 4. CLAUDE.md

**Changes:**
- ✓ Added new "Skill Workflow" section showing the complete sequence
- ✓ Added "Key Separation of Concerns" explaining each skill's scope
- ✓ Documented why this separation exists
- ✓ Made it clear that setup-experiment-dirs is framework-agnostic
- ✓ Made it clear that create-torchtune-config is torchtune-specific

**New section shows:**
```
1. plan-runs → creates runs_plan.md
2. setup-experiment-dirs → creates directories
3. create-torchtune-config → creates finetune.yaml
4. generate-slurm-script → creates finetune.slurm
5. launch-runs → submits jobs
6. monitor-jobs → tracks progress
```

---

## Workflow Clarity

**Before** (ambiguous):
- "setup-experiment-dirs creates configs" vs "create-torchtune-config creates configs" → confusing!
- Not clear which one to use when

**After** (clear):
```
setup-experiment-dirs:
  - Creates: Directories only
  - Scope: Framework-agnostic
  - Always required

create-torchtune-config:
  - Creates: finetune.yaml files
  - Scope: Torchtune-specific
  - Only if using torchtune

generate-slurm-script:
  - Creates: finetune.slurm files
  - Scope: Framework-agnostic
  - Reads configs to determine resources
```

---

## Current Experiment Status

For your current experiment (`cap_cross_eval_5_9_13L_2025-10-20`):

**Completed:**
- ✓ plan-runs (runs_plan.md and runs_status.yaml exist)

**Not yet done:**
- ✗ setup-experiment-dirs (no run directories created yet)
- ✗ create-torchtune-config (no finetune.yaml files)
- ✗ generate-slurm-script (no finetune.slurm files)
- ✗ launch-runs (no jobs submitted)

**Next steps when you return:**
1. Run setup-experiment-dirs to create all 14 run directories
2. Run create-torchtune-config to generate all finetune.yaml files
3. Run generate-slurm-script to generate all finetune.slurm files
4. Run launch-runs to submit all jobs to SLURM

---

## Benefits of This Approach

1. **Clear responsibilities**: Each skill has one job
2. **Easier debugging**: If something fails, you know which skill to check
3. **Flexible workflow**: Can skip steps (e.g., use different SLURM generator)
4. **Framework independence**: Could swap torchtune for Axolotl/Hugging Face/etc.
5. **Better error messages**: Each skill can validate its own prerequisites
6. **Easier testing**: Can test each skill independently

---

## Files Modified

1. `.claude/skills/setup-experiment-dirs/skill.md` - Clarified scope (directories only)
2. `.claude/skills/create-torchtune-config/skill.md` - Clarified scope (torchtune configs only)
3. `.claude/skills/generate-slurm-script/skill.md` - Clarified scope (SLURM scripts only)
4. `CLAUDE.md` - Added workflow diagram and separation rationale

---

## What Didn't Change

- The Python tools themselves (`setup_finetune.py`, `generate_slurm_scripts.py`) - no code changes
- The templates (`finetune_template.yaml`, `finetune_template.slurm`) - no changes
- The other skills (plan-runs, launch-runs, etc.) - still TODO for future improvements
- Functionality - all skills still do what they did before, just with clearer documentation

---

## Next Improvements (from original list)

These are still on the TODO list:

**HIGH PRIORITY:**
3. Add pre-flight validation to launch-runs
4. Add consistent terminology across all skills

**MEDIUM PRIORITY:**
5. Move evaluation TODO to separate doc in launch-runs
6. Add packing validation code to create-torchtune-config
7. Add resume support to launch-runs
8. Add validation for runs_plan.md parsing

**LOW PRIORITY:**
9. Add Quick Start examples to each skill
10. Improve error message formatting
11. Add ETA calculations to job submission
12. Add plan validation before directory creation

# Setup Experiment Directories

You are helping the user create a well-organized directory structure for their experiments.

**SCOPE**: This skill creates ONLY the directory structure (framework-agnostic). It does NOT generate configuration files.

**WORKFLOW POSITION**: Run this skill AFTER `plan-runs` and BEFORE framework-specific config generation (e.g., `create-torchtune-config`).

## Your Task

Read `runs_plan.md` and create a complete directory structure with README files and evaluation subdirectories.

## Steps

### 1. Read runs_plan.md

**IMPORTANT**: This skill assumes `runs_plan.md` already exists in the experiment directory.

If it doesn't exist:
- Ask user to run the `plan-runs` skill first
- Do NOT proceed without a plan

**Extract from runs_plan.md:**
- Run group name (experiment directory name)
- List of all run names (from "All Runs" table)
- Evaluation tasks (from "Evaluations" section)
- Whether base model runs exist (from "Control Runs" table)

### 2. Verify Experiment Directory

Check if the experiment directory exists:

```bash
# From runs_plan.md, get run_group name (e.g., cap_cross_eval_5_9_13L_2025-10-20)
experiment_dir="/scratch/gpfs/MSALGANIK/mjs3/{run_group_name}"

# Check if already exists
if [ -d "$experiment_dir" ]; then
  echo "✓ Experiment directory exists: $experiment_dir"
  cd "$experiment_dir"
else
  echo "❌ ERROR: Experiment directory not found"
  echo "Expected: $experiment_dir"
  exit 1
fi
```

**If directory doesn't exist:**
- This is unexpected (plan-runs should have created it)
- Create it now: `mkdir -p $experiment_dir`
- Move runs_plan.md and runs_status.yaml into it if they're elsewhere

### 3. Create Run Subdirectories

For each run in the plan (both fine-tuned and base models):

```bash
# Create run directory
mkdir -p "{run_name}"

# Create evaluations subdirectory
mkdir -p "{run_name}/evaluations"
```

**Example for current experiment:**
```bash
cd /scratch/gpfs/MSALGANIK/mjs3/cap_cross_eval_5_9_13L_2025-10-20

# Fine-tuned runs (12 total)
mkdir -p Llama-3.2-1B-Instruct_5L_rank4/evaluations
mkdir -p Llama-3.2-1B-Instruct_5L_rank64/evaluations
mkdir -p Llama-3.2-1B-Instruct_9L_rank4/evaluations
mkdir -p Llama-3.2-1B-Instruct_9L_rank64/evaluations
mkdir -p Llama-3.2-1B-Instruct_13L_rank4/evaluations
mkdir -p Llama-3.2-1B-Instruct_13L_rank64/evaluations
mkdir -p Llama-3.2-3B-Instruct_5L_rank4/evaluations
mkdir -p Llama-3.2-3B-Instruct_5L_rank64/evaluations
mkdir -p Llama-3.2-3B-Instruct_9L_rank4/evaluations
mkdir -p Llama-3.2-3B-Instruct_9L_rank64/evaluations
mkdir -p Llama-3.2-3B-Instruct_13L_rank4/evaluations
mkdir -p Llama-3.2-3B-Instruct_13L_rank64/evaluations

# Base model controls (2 total)
mkdir -p Llama-3.2-1B-Instruct_base/evaluations
mkdir -p Llama-3.2-3B-Instruct_base/evaluations
```

### 4. Write README.md (Experiment Level)

Create a top-level README in the experiment directory that summarizes the experiment:

```markdown
# {run_group_name}

**Created**: {date}
**Type**: {experiment_type from runs_plan.md}
**Total Runs**: {count}

## Overview

{Brief description from runs_plan.md}

## Variables

{Copy from runs_plan.md}

## Directory Structure

```
{run_group_name}/
  Llama-3.2-1B-Instruct_5L_rank4/
    evaluations/
      cap_5L_epoch_0/
      cap_5L_epoch_1/
      cap_9L_epoch_0/
      ...
  Llama-3.2-1B-Instruct_5L_rank64/
    evaluations/
      ...
  ...
  runs_plan.md
  runs_status.yaml
  README.md (this file)
```

## Quick Start

```bash
# View plan
cat runs_plan.md

# Check status
cat runs_status.yaml

# Generate configs (if using torchtune)
# [instructions from runs_plan.md]

# Submit jobs
# [instructions from runs_plan.md]
```

## Files

- `runs_plan.md`: Complete experiment plan with resource estimates
- `runs_status.yaml`: Job status tracking (updated by launch-runs skill)
- `README.md`: This file

## Notes

{Any additional context from runs_plan.md}
```

**Write this file:**
```bash
cat > README.md << 'EOF'
[content above]
EOF
```

### 5. Verify Directory Creation

After creating all directories, verify the structure:

```bash
# Count run directories (should match runs_plan.md)
run_count=$(find . -maxdepth 1 -type d -name "Llama*" | wc -l)
echo "Created $run_count run directories"

# Count evaluation subdirectories (should equal run_count)
eval_count=$(find . -maxdepth 2 -type d -name "evaluations" | wc -l)
echo "Created $eval_count evaluation subdirectories"

# List all directories
tree -L 2 -d
# Or if tree not available:
ls -R
```

**Expected output for current experiment:**
```
Created 14 run directories
Created 14 evaluation subdirectories

cap_cross_eval_5_9_13L_2025-10-20/
├── Llama-3.2-1B-Instruct_5L_rank4/
│   └── evaluations/
├── Llama-3.2-1B-Instruct_5L_rank64/
│   └── evaluations/
...
├── Llama-3.2-1B-Instruct_base/
│   └── evaluations/
├── Llama-3.2-3B-Instruct_base/
│   └── evaluations/
├── runs_plan.md
├── runs_status.yaml
└── README.md
```

## Directory Structure Best Practices

### Evaluation Storage

**Store evaluations in each run's subdirectory**, not in a centralized folder.

**Why this approach?**
- Self-contained runs: Everything about a run lives in one place
- Easy archiving: Tar one directory to backup/share a complete run
- Simple cleanup: Delete a run directory, everything goes with it
- Matches tooling: `setup_inspect.py` naturally works in run directories
- inspect view can still compare across runs even when stored separately

**After training completes, evaluation results will be stored like:**
```
[run_name]/
├── finetune.yaml          # (created by create-torchtune-config)
├── finetune.slurm         # (created by generate-slurm-script)
├── epoch_0/               # (created by training job)
├── epoch_1/               # (created by training job)
└── evaluations/           # (created by this skill)
    ├── cap_5L_epoch_0/    # (created by evaluation jobs)
    ├── cap_5L_epoch_1/
    ├── cap_9L_epoch_0/
    └── cap_9L_epoch_1/
```

## Summary Report

After completing all steps, show a summary:

```
=== Directory Setup Complete ===

Experiment: cap_cross_eval_5_9_13L_2025-10-20
Location: /scratch/gpfs/MSALGANIK/mjs3/cap_cross_eval_5_9_13L_2025-10-20

Created:
  ✓ 12 fine-tuned run directories
  ✓ 2 base model directories
  ✓ 14 evaluation subdirectories
  ✓ README.md

Status: Ready for configuration generation

Next steps:
  1. Generate torchtune configs: Use create-torchtune-config skill
  2. Generate SLURM scripts: Use generate-slurm-script skill (or included in step 1)
  3. Submit jobs: Use launch-runs skill

Files ready:
  - runs_plan.md ✓
  - runs_status.yaml ✓
  - README.md ✓
```

## Next Steps

**Suggest to the user:**

"I've created the directory structure with 14 run directories and evaluation subdirectories.

**If using torchtune for fine-tuning:**
Would you like me to generate the torchtune configuration files using the `create-torchtune-config` skill?

**If using a different framework:**
You can now manually create your configuration files in each run directory, or use framework-specific tooling.

**Current structure:**
- All run directories created ✓
- Evaluation subdirectories ready ✓
- Next: Generate framework-specific configs"

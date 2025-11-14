# Model Optimizers - Meta-Organization Pattern (Execution)

This directory contains tool-specific modules for executing model optimization jobs (e.g., fine-tuning, prompt engineering).

## Standard Workflow Stages

Each optimizer tool should follow this four-stage pattern for job execution:

### Stage 1: INPUT PROCESSING
**File:** `parsing.md`
**Purpose:** Extract structured data from experiment_summary.md and scan for job scripts
**Outputs:** List of runs to execute, job metadata

### Stage 2: PLANNING / SELECTION
**Files:** `*_selection.md`
**Purpose:** Decide which jobs to submit (skip completed, handle resumability)
**Examples:**
- `run_selection.md` - Choose which runs to submit based on completion status
- `dependency_checking.md` - Verify prerequisites are met before submission

### Stage 3: EXECUTION
**Files:** `job_submission.md`, `monitoring.md`
**Purpose:** Submit SLURM jobs and monitor their progress
**Key distinction:**
- `job_submission.md` - Submit jobs via sbatch, capture job IDs, handle staggering
- `monitoring.md` - Poll squeue, track status updates, wait for completion

### Stage 4: VERIFICATION
**File:** `validation.md`
**Purpose:** Verify execution completed successfully
**Checks:** All jobs reached terminal states, outputs exist, no failures

## Tool Structure Template

When adding a new optimizer execution tool, use this structure:

```
optimizers/
└── {tool-name}/
    ├── main.md                      # Tool overview and entry point
    ├── parsing.md                   # Stage 1: Parse experiment and scan for jobs
    ├── run_selection.md             # Stage 2: Select which runs to submit
    ├── job_submission.md            # Stage 3a: Submit SLURM jobs
    ├── monitoring.md                # Stage 3b: Monitor job progress
    └── validation.md                # Stage 4: Verify completion
```

## Current Tools

### torchtune
Execute fine-tuning jobs using torchtune framework.

**Workflow:**
1. `parsing.md` - Parse experiment_summary.md, find finetune.slurm scripts
2. `run_selection.md` - Decide which runs to submit (skip completed)
3. `job_submission.md` - Submit jobs with sbatch, capture IDs, stagger submissions
4. `monitoring.md` - Poll squeue every minute, update status
5. `validation.md` - Verify all jobs completed, model checkpoints exist

## Design Principles

**Job Execution Focus:** Unlike scaffold-experiment which generates files, run-experiment submits and monitors SLURM jobs

**Resumability:** Selection logic ensures already-completed jobs aren't resubmitted

**Status Tracking:** Monitoring continuously updates experiment_summary.md with job status

**Progressive Disclosure:** SKILL.md links to main.md, which links to specific stage files

**Consistency:** All tools follow the same 4-stage pattern for predictable organization

## Future Tools

When adding new optimizers (DSPy, custom trainers, etc.), follow this same pattern:
- Keep the 4-stage structure
- Use `run_selection.md` for deciding which jobs to submit
- Use `job_submission.md` for SLURM job submission
- Use `monitoring.md` for progress tracking
- Document tool-specific details in `main.md`

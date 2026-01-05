# Model Evaluators - Meta-Organization Pattern (Execution)

This directory contains tool-specific modules for executing model evaluation jobs (e.g., inspect-ai, custom evaluation frameworks).

## Standard Workflow Stages

Each evaluator tool should follow this four-stage pattern for job execution:

### Stage 1: INPUT PROCESSING
**File:** `parsing.md`
**Purpose:** Extract structured data from experiment_summary.yaml and scan for evaluation scripts
**Outputs:** List of evaluations to execute, job metadata

### Stage 2: PLANNING / SELECTION
**Files:** `*_selection.md`, `*_checking.md`
**Purpose:** Decide which evaluations to submit and verify dependencies
**Examples:**
- `dependency_checking.md` - **CRITICAL:** Verify fine-tuning complete, model checkpoints exist
- `evaluation_selection.md` - Choose which evaluations to submit (skip completed)

**Important:** Evaluations cannot run without fine-tuned model checkpoints. Dependency checking is mandatory.

### Stage 3: EXECUTION
**Files:** `job_submission.md`, `monitoring.md`
**Purpose:** Submit SLURM jobs and monitor their progress
**Key distinction:**
- `job_submission.md` - Submit jobs via sbatch, capture job IDs
- `monitoring.md` - Poll squeue, track status updates, wait for completion

### Stage 4: VERIFICATION
**File:** `validation.md`
**Purpose:** Verify execution completed successfully
**Checks:** All jobs reached terminal states, evaluation logs exist, no failures

## Tool Structure Template

When adding a new evaluator execution tool, use this structure:

```
evaluators/
└── {tool-name}/
    ├── main.md                      # Tool overview and entry point
    ├── parsing.md                   # Stage 1: Parse experiment and scan for eval jobs
    ├── dependency_checking.md       # Stage 2a: Verify prerequisites met
    ├── evaluation_selection.md      # Stage 2b: Select which evals to submit
    ├── job_submission.md            # Stage 3a: Submit SLURM jobs
    ├── monitoring.md                # Stage 3b: Monitor job progress
    └── validation.md                # Stage 4: Verify completion
```

## Current Tools

### inspect
Execute evaluation jobs using inspect-ai framework.

**Workflow:**
1. `parsing.md` - Parse experiment_summary.md, find eval/*.slurm scripts
2. `dependency_checking.md` - **CRITICAL:** Verify fine-tuning complete, checkpoints exist
3. `evaluation_selection.md` - Decide which evaluations to submit (skip completed, skip runs with missing checkpoints)
4. `job_submission.md` - Submit jobs with sbatch, capture IDs
5. `monitoring.md` - Poll squeue every minute, update status
6. `validation.md` - Verify all jobs completed, evaluation logs exist

## Design Principles

**Job Execution Focus:** Unlike scaffold-experiment which generates files, run-experiment submits and monitors SLURM jobs

**Dependency Awareness:** Evaluations MUST NOT run before fine-tuning completes. Always check dependencies first.

**Resumability:** Selection logic ensures already-completed evaluations aren't resubmitted

**Progressive Disclosure:** SKILL.md links to main.md, which links to specific stage files

**Consistency:** All tools follow the same 4-stage pattern for predictable organization

## Future Tools

When adding new evaluators (custom frameworks, benchmark suites, etc.), follow this same pattern:
- Keep the 4-stage structure
- **Always include dependency_checking.md** to verify prerequisites
- Use `evaluation_selection.md` for deciding which evaluations to submit
- Use `job_submission.md` for SLURM job submission
- Use `monitoring.md` for progress tracking
- Document tool-specific details in `main.md`

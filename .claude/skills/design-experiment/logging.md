# Logging - design-experiment

**See [shared/logging_spec.md](../../shared/logging_spec.md) for complete format specification and general logging guidelines.**

This document covers design-experiment-specific logging practices.

---

## Log File Location

`{experiment_dir}/design-experiment.log`

Created during the planning workflow to record all verification steps, calculations, and decisions.

---

## Action Types

Design-experiment uses these action types:

| Action Type | Purpose |
|-------------|---------|
| `VERIFY_MODEL` | Check that a model exists |
| `VERIFY_DATASET` | Check that a dataset exists |
| `VERIFY_EVAL_TASK` | Check that an evaluation script exists |
| `CHECK_DISK` | Verify available disk space |
| `SEARCH_PRIOR_RUNS` | Find similar experiments for estimation |
| `EXTRACT_SPEED` | Extract iteration speed from prior run logs |
| `CALCULATE_TIME` | Compute time estimates |
| `CALCULATE_DISK` | Compute disk space requirements |
| `DECIDE_NAMING` | Document naming decisions |
| `DECIDE_CONFIG` | Document configuration choices |
| `CREATE_SUMMARY` | Create experiment_summary.md |
| `COMPLETE` | Mark planning phase complete |

---

## What to Log

**DO log:**
- ✓ Resource verification commands (ls, du, df) and results
- ✓ Prior run searches and data extraction (find, grep)
- ✓ All calculations (time, disk space, batch sizes, GPU hours)
- ✓ Decisions made (naming, recipe selection, configuration)
- ✓ File creation (experiment_summary.md)

**DON'T log:**
- ✗ Job status checks (squeue, sacct) - those are for run-experiment
- ✗ Simple file reads that don't affect the plan

---

## When to Log

### During param_selection.md
- **Step 1:** Location detection and confirmation
- **Step 6:** Naming decisions
- **Step 7:** All resource verification commands and results
- **Step 8:** All prior run searches, data extraction, calculations

### During experiment_generation.md
- experiment_summary.md creation
- Final status and next steps

---

## Example Log Entries

```
[2025-11-11 14:23:15] VERIFY_MODEL: Checking Llama-3.2-1B-Instruct
Command: ls /models/Llama-3.2-1B-Instruct
Result: Directory exists with 15 files
Explanation: Verifying base model exists before creating experiment plan

[2025-11-11 14:24:01] SEARCH_PRIOR_RUNS: Looking for similar experiments
Command: find /scratch/ck-outputs -name "slurm-*.out" | head -5
Result: Found 3 similar runs
Explanation: Searching for prior SLURM logs to extract training speed data

[2025-11-11 14:24:30] CALCULATE_TIME: Training time estimate
Input: 8000 samples, batch_size=4, speed=4.34 it/s, epochs=2
Calculation: steps_per_epoch = 8000/4 = 2000, time = 2000/4.34 ≈ 8min/epoch
Result: Estimated 16 minutes total (8 min × 2 epochs)

[2025-11-11 14:26:15] COMPLETE: Experiment design finished
Status: Plan approved by user, files written
Next Steps: User can now run scaffold-experiment to generate configs
```

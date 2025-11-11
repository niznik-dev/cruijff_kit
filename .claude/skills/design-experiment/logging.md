# Logging

Create a detailed log file at `{experiment_dir}/design-experiment.log` that records all verification steps, calculations, and decisions made during planning.

## Purpose

The log enables:
1. **Debugging:** If estimates are wrong, check what commands were run and what data was used
2. **Reproducibility:** Another person (or Claude) can understand exactly what was done
3. **Improvement:** Review logs to identify better approaches or missing steps
4. **Auditing:** Verify that all resources were properly checked before committing to the experiment

---

## What to Log

### DO Log
- ✓ Resource verification commands (ls, du, df)
- ✓ Prior run searches and data extraction (find, grep)
- ✓ Calculations (time estimates, batch sizes, disk space)
- ✓ Decisions made (naming choices, recipe selection, configuration)
- ✓ File creation (experiment_summary.md, directories)

### DON'T Log
- ✗ Job status checks (squeue, sacct)
- ✗ Simple read operations that don't affect the plan

---

## Log Format

```
[{timestamp}] {ACTION_TYPE}: {Brief description}
Command: {command_run}
Result: {result_summary}
Explanation: {why_this_matters}
```

**Timestamp:** `YYYY-MM-DD HH:MM:SS` format

**Action types:** VERIFY_MODEL, VERIFY_DATASET, SEARCH_PRIOR_RUNS, EXTRACT_SPEED, CALCULATE_TIME, CHECK_DISK, DECIDE_NAMING, CREATE_SUMMARY, COMPLETE

---

## Example Log Entries

### Resource Verification

```
[2025-10-22 14:23:15] VERIFY_MODEL: Checking Llama-3.2-1B-Instruct
Command: ls {models_dir}/Llama-3.2-1B-Instruct
Result: Directory exists with 15 files (config.json, model.safetensors, etc.)
Explanation: Verifying base model exists before creating experiment plan

[2025-10-22 14:23:42] VERIFY_DATASET: Checking capitalization dataset
Command: ls -lh {repo_dir}/data/green/capitalization/words_8L_80P_10000.json
Result: File exists, 655KB
Explanation: Verifying training dataset exists and checking size
```

### Prior Run Analysis

```
[2025-10-22 14:24:01] SEARCH_PRIOR_RUNS: Looking for similar experiments
Command: find {scratch_dir} -name "slurm-*.out" -path "*/ck-out-*" -size +100k | head -5
Result: Found 3 similar runs: ck-out-happy-narwhal, ck-out-bright-horizon, ck-out-calm-dolphin
Explanation: Searching for prior SLURM logs to extract training speed data for estimates

[2025-10-22 14:24:15] EXTRACT_SPEED: Analyzing prior run for training speed
Command: grep -E "[0-9.]+it/s" {scratch_dir}/ck-out-happy-narwhal/slurm-12345.out | tail -20
Result: Average speed after warmup: 4.34 it/s
Explanation: Extracting iteration speed from similar prior run with same model and batch size
```

### Calculations

```
[2025-10-22 14:24:30] CALCULATE_TIME: Training time estimate
Input: 8000 samples, batch_size=4, speed=4.34 it/s, epochs=2
Calculation: steps_per_epoch = 8000/4 = 2000, time_per_epoch = 2000/4.34 ≈ 461s ≈ 8min
Result: Estimated 16 minutes total (8 min × 2 epochs)
Explanation: Calculated training time based on actual iteration speed from prior run

[2025-10-22 14:25:00] CHECK_DISK: Verifying available disk space
Command: df -h {scratch_dir}
Result: 2.1T available
Explanation: Ensuring sufficient space for ~40 GiB of checkpoints
```

### Decisions

```
[2025-10-22 14:25:30] DECIDE_NAMING: Experiment name chosen
Decision: cap_8L_lora_comparison_2025-10-22
Reasoning: Capitalization task (cap), 8-letter words (8L), comparing LoRA ranks, dated
Explanation: User confirmed this naming follows convention and is descriptive
```

### File Creation

```
[2025-10-22 14:26:00] CREATE_SUMMARY: Writing experiment plan
Command: Created {scratch_dir}/cap_8L_lora_comparison_2025-10-22/experiment_summary.md
Result: File created with 4 runs (2 fine-tuned × 2 ranks + 2 controls)
Explanation: Comprehensive experiment plan with all configurations documented

[2025-10-22 14:26:15] COMPLETE: Experiment design finished
Status: Plan approved by user, files written
Next Steps: User can now run scaffold-experiment to generate configs
Explanation: Planning phase complete, documented in summary and this log
```

---

## When to Log

### During param_selection.md
Log throughout the interactive workflow:
- **Step 1:** Location detection and confirmation
- **Step 6:** Naming decisions
- **Step 7:** All resource verification commands and results
- **Step 8:** All prior run searches, data extraction, calculations

### During experiment_generation.md
Log file creation:
- experiment_summary.md creation
- design-experiment.log creation
- Final status and next steps

---

## Log Location

Always create the log at: `{experiment_dir}/design-experiment.log`

This keeps it with the experiment for easy reference during scaffolding, execution, and analysis phases.

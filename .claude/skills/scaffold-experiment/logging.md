# Logging

Create detailed log files that record all scaffolding actions. Logging is essential for debugging, reproducibility, and tracking what configs were generated.

## Purpose

Logs enable:
1. **Debugging:** Track what files were created and what errors occurred
2. **Reproducibility:** Another person (or Claude) can see exactly what was scaffolded
3. **Auditing:** Verify correct configs were generated before running jobs

## Log Files

### For torchtune scaffolding
**Location:** `{experiment_dir}/scaffold-torchtune.log`

**Created by:** optimizers/torchtune/ modules

### For inspect-ai scaffolding
**Location:** `{experiment_dir}/scaffold-inspect.log`

**Created by:** evaluators/inspect/ modules

## Log Format

```
[{timestamp}] {ACTION_TYPE}
Details: {what_happened}
Result: {outcome}
```

**Timestamp:** `YYYY-MM-DD HH:MM:SS` format

## Torchtune Log Example

```
[2025-11-11 10:15:30] SCAFFOLD_TORCHTUNE_START
Details: Generating fine-tuning configs for 4 runs
Result: Analyzing experiment_summary.md

[2025-11-11 10:15:31] ANALYZE_PARAMETERS
Details: Varying parameters: lora_rank, learning_rate
Result: Directory naming pattern: {model}_r{rank}_lr{lr}

[2025-11-11 10:15:32] CREATE_RUN_DIRS
Details: Llama-3.2-1B_r8_lr1e-5, Llama-3.2-1B_r8_lr5e-5, Llama-3.2-1B_r16_lr1e-5, Llama-3.2-1B_r16_lr5e-5
Result: All directories created successfully

[2025-11-11 10:15:33] GENERATE_YAMLS
Details: Using template /path/to/template_setup_finetune.yaml
Result: 4 setup_finetune.yaml files created

[2025-11-11 10:15:34] RUN_SETUP_SCRIPTS
Details: Batch execution with conda environment activated
Result: 4 runs processed, 4 successful, 0 failures

[2025-11-11 10:15:45] VALIDATE_PARAMETERS
Details: Comparing finetune.yaml parameters against directory names
Result: All runs validated successfully

[2025-11-11 10:15:45] SCAFFOLD_TORCHTUNE_COMPLETE
Details: Generated configs for 4 fine-tuning runs
Duration: 15 seconds
```

## Inspect-ai Log Example

```
[2025-11-11 10:20:15] SCAFFOLD_INSPECT_START
Details: Generating evaluation configs for 4 runs × 2 tasks × 2 epochs
Result: Analyzing experiment_summary.md

[2025-11-11 10:20:16] ANALYZE_EVALUATION_PLAN
Details: Tasks: capitalization, reasoning; Epochs: 0, 1
Result: 16 total evaluations to configure

[2025-11-11 10:20:17] CREATE_EVAL_DIRS
Details: Creating eval/ and logs/ in 4 run directories
Result: All eval directories created successfully

[2025-11-11 10:20:18] VERIFY_TASK_SCRIPTS
Details: Checking capitalization.py, reasoning.py
Result: All task scripts exist

[2025-11-11 10:20:19] GENERATE_SLURM_SCRIPTS
Details: Using inspect task prefix pattern
Result: 16 SLURM scripts created (4 runs × 2 tasks × 2 epochs)

[2025-11-11 10:20:20] VALIDATE_SCRIPTS
Details: Checking all scripts are executable and reference correct checkpoints
Result: All scripts validated successfully

[2025-11-11 10:20:20] SCAFFOLD_INSPECT_COMPLETE
Details: Generated evaluation configs for 16 evaluations
Duration: 5 seconds
```

## Action Types

### Torchtune Actions
- `SCAFFOLD_TORCHTUNE_START` - Begin scaffolding
- `ANALYZE_PARAMETERS` - Parse experiment_summary.md
- `CREATE_RUN_DIRS` - Create run directories
- `GENERATE_YAMLS` - Create setup_finetune.yaml files
- `RUN_SETUP_SCRIPTS` - Execute setup_finetune.py
- `VALIDATE_PARAMETERS` - Verify configs match runs
- `SCAFFOLD_TORCHTUNE_COMPLETE` - Finish scaffolding

### Inspect-ai Actions
- `SCAFFOLD_INSPECT_START` - Begin scaffolding
- `ANALYZE_EVALUATION_PLAN` - Parse experiment_summary.md
- `CREATE_EVAL_DIRS` - Create eval/ subdirectories
- `VERIFY_TASK_SCRIPTS` - Check task scripts exist
- `GENERATE_SLURM_SCRIPTS` - Create SLURM batch scripts
- `VALIDATE_SCRIPTS` - Verify scripts correctness
- `SCAFFOLD_INSPECT_COMPLETE` - Finish scaffolding

## When to Log

### During Each Stage

**Config Parsing:**
- Log which file is being parsed
- Log extracted parameters

**Selection:**
- Log which runs/evaluations are being configured
- Log naming patterns

**Generation:**
- Log each file created
- Log template used
- Log any errors during generation

**Validation:**
- Log validation checks performed
- Log validation results (pass/fail)

## Module Responsibilities

Each module in optimizers/torchtune/ and evaluators/inspect/ should log its actions:

- **config_parsing.md:** Log file parsing and parameter extraction
- **run_selection.md / evaluation_selection.md:** Log which runs/evals selected
- **config_generation.md:** Log each config file created
- **script_generation.md:** Log each script created
- **script_execution.md:** Log setup script execution
- **validation.md:** Log all validation checks

**See each module for specific logging requirements.**

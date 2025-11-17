# Logging - scaffold-experiment

**See [shared/logging_spec.md](../../shared/logging_spec.md) for complete format specification and general logging guidelines.**

This document covers scaffold-experiment-specific logging practices.

---

## Log File Locations

Scaffold-experiment creates tool-specific logs:

- **Torchtune scaffolding:** `{experiment_dir}/scaffold-torchtune.log`
- **Inspect-ai scaffolding:** `{experiment_dir}/scaffold-inspect.log`

Both created during config generation to record what files were created and validation results.

---

## Action Types

### Torchtune Actions

| Action Type | Purpose |
|-------------|---------|
| `SCAFFOLD_TORCHTUNE_START` | Begin fine-tuning config generation |
| `ANALYZE_PARAMETERS` | Parse experiment_summary.md for varying parameters |
| `CREATE_RUN_DIRS` | Create run directories |
| `GENERATE_YAMLS` | Create setup_finetune.yaml files |
| `RUN_SETUP_SCRIPTS` | Execute setup_finetune.py to generate configs |
| `VALIDATE_PARAMETERS` | Verify generated configs match directory names |
| `SCAFFOLD_TORCHTUNE_COMPLETE` | Finish fine-tuning scaffolding |

### Inspect-ai Actions

| Action Type | Purpose |
|-------------|---------|
| `SCAFFOLD_INSPECT_START` | Begin evaluation config generation |
| `ANALYZE_EVALUATION_PLAN` | Parse experiment_summary.md for evaluation plan |
| `CREATE_EVAL_DIRS` | Create eval/ subdirectories |
| `VERIFY_TASK_SCRIPTS` | Check that task scripts exist |
| `GENERATE_SLURM_SCRIPTS` | Create SLURM batch scripts for evaluations |
| `VALIDATE_SCRIPTS` | Verify scripts reference correct checkpoints |
| `SCAFFOLD_INSPECT_COMPLETE` | Finish evaluation scaffolding |

---

## When to Log

### During Config Parsing
- Which file is being parsed (experiment_summary.md, claude.local.md)
- Extracted parameters and paths

### During Selection/Generation
- Which runs/evaluations are being configured
- Naming patterns being used
- Each file created (with template source)
- Any errors during generation

### During Validation
- Validation checks performed
- Validation results (pass/fail)
- Any mismatches found

---

## Module Responsibilities

Each module in optimizers/ and evaluators/ should log its actions:

- **parsing.md:** File parsing and parameter extraction
- **directory_generation.md / scenario_selection.md:** Which runs/evals selected
- **yaml_generation.md / slurm_generation.md:** Each config/script file created
- **script_execution.md:** Setup script execution
- **validation.md:** All validation checks

---

## Example Log Entries

### Torchtune Scaffolding

```
[2025-11-11 10:15:30] SCAFFOLD_TORCHTUNE_START
Details: Generating fine-tuning configs for 4 runs
Result: Analyzing experiment_summary.md

[2025-11-11 10:15:31] ANALYZE_PARAMETERS
Details: Varying parameters: lora_rank, learning_rate
Result: Directory naming pattern: r{rank}_lr{lr}

[2025-11-11 10:15:34] RUN_SETUP_SCRIPTS
Details: Batch execution with conda environment activated
Result: 4 runs processed, 4 successful, 0 failures

[2025-11-11 10:15:45] VALIDATE_PARAMETERS
Details: Comparing finetune.yaml parameters against directory names
Result: All runs validated successfully
```

### Inspect-ai Scaffolding

```
[2025-11-11 10:20:15] SCAFFOLD_INSPECT_START
Details: Generating evaluation configs for 4 runs × 2 tasks × 2 epochs
Result: 16 total evaluations to configure

[2025-11-11 10:20:18] VERIFY_TASK_SCRIPTS
Details: Checking capitalization.py, reasoning.py
Result: All task scripts exist

[2025-11-11 10:20:19] GENERATE_SLURM_SCRIPTS
Details: Using inspect task prefix pattern
Result: 16 SLURM scripts created

[2025-11-11 10:20:20] SCAFFOLD_INSPECT_COMPLETE
Details: Generated evaluation configs for 16 evaluations
Duration: 5 seconds
```

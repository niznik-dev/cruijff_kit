# Logging - convert-tabular-to-text

**See [shared/logging_spec.md](../../shared/logging_spec.md) for complete format specification and general logging guidelines.**

This document covers convert-tabular-to-text-specific logging practices.

---

## Log File Location

```
{experiment_dir}/logs/convert-tabular-to-text.log
```

Created during the dataset generation workflow to record all steps, decisions, and validation results.

**If no experiment directory exists** (standalone usage), write the log to:
```
{scratch_dir}/ck-data/generated/logs/convert-tabular-to-text.log
```

---

## Action Types

| Action Type | Purpose |
|-------------|---------|
| `START_CONVERSION` | Mark beginning of dataset generation |
| `CHECK_EXISTING` | Check for existing generated datasets |
| `LOAD_SCHEMA` | Load and verify schema file |
| `LOAD_EXPERIMENT` | Load experiment_summary.yaml for guided mode |
| `DETERMINE_TEMPLATE` | Identify template style per condition |
| `GENERATE_TEMPLATE` | Generate custom Jinja2 template (narrative only) |
| `DETERMINE_CONDITIONS` | Load or define conditions |
| `PLAN_OUTPUT_FILES` | Determine which (condition x split) files to generate |
| `GENERATE_DATASET` | Generate a single output file via convert.py |
| `VALIDATE_FILE` | Per-file validation result |
| `VALIDATE_CROSS_FILE` | Cross-file consistency check |
| `REPORT_PATHS` | Final paths reported to user |
| `COMPLETE_CONVERSION` | Mark end of dataset generation |

---

## When to Log

### During Setup (Steps 1-5)
- Existing dataset checks (paths checked, files found)
- Schema loading (path, column count, validation result)
- Experiment config loading (conditions found, shared settings)
- Template decisions (type per condition, template file paths)
- Custom template generation (agent invocation, output path, user approval)
- Condition details (features, perturbations, template type per condition)
- Output file plan (file list, user confirmation)

### During Generation (Step 6)
- Each `convert.py` invocation (condition, split, output path, row count, duration)
- Errors or warnings from convert.py

### During Validation (Step 7)
- Per-file validation results (checks passed/failed/warned)
- Cross-file consistency results
- Overall validation summary

### On Completion (Step 8)
- Total files generated
- Total duration
- Final paths

---

## Example Log Entries

### START_CONVERSION

```
[2026-04-05 14:30:00] START_CONVERSION
Details: Beginning tabular-to-text dataset generation
Experiment: acs_dict_to_narr_2026-04-05
Mode: guided (experiment_summary.yaml)
Directory: /scratch/gpfs/MSALGANIK/sarahep/ck-projects/{project}/acs_dict_to_narr_2026-04-05
Result: success
```

### CHECK_EXISTING

```
[2026-04-05 14:30:01] CHECK_EXISTING
Details: Checking for existing generated datasets
Path: /scratch/gpfs/MSALGANIK/sarahep/ck-data/generated/
Files found: 0
Result: success
```

### LOAD_SCHEMA

```
[2026-04-05 14:30:02] LOAD_SCHEMA
Details: Loading schema file
Path: /scratch/gpfs/MSALGANIK/sarahep/ck-data/schemas/acs_pums_oh_2018.yaml
Columns: 9 (AGEP, COW, SCHL, MAR, OCCP, SEX, RAC1P, PINCP, ST)
Result: success
```

### LOAD_EXPERIMENT

```
[2026-04-05 14:30:03] LOAD_EXPERIMENT
Details: Loading experiment configuration
Path: /scratch/gpfs/MSALGANIK/sarahep/ck-projects/{project}/acs_dict_to_narr_2026-04-05/experiment_summary.yaml
Conditions: 2 (dict_full, narr_full)
Target: PINCP > 50000
Seed: 42
Split ratio: 0.8
Subsampling: 0.1
Result: success
```

### DETERMINE_TEMPLATE

```
[2026-04-05 14:30:04] DETERMINE_TEMPLATE
Details: Determining template style for condition
Condition: dict_full
Template: dictionary
Template file: none
Result: success
```

```
[2026-04-05 14:30:04] DETERMINE_TEMPLATE
Details: Determining template style for condition
Condition: narr_full
Template: narrative
Template file: not yet generated
Style guidance: "Natural, readable prose suitable for evaluation..."
Action: will generate custom Jinja2 template
Result: success
```

### GENERATE_TEMPLATE

```
[2026-04-05 14:30:30] GENERATE_TEMPLATE
Details: Generated custom Jinja2 narrative template
Condition: narr_full
Agent: generate-jinja-template
Output: /scratch/gpfs/MSALGANIK/sarahep/ck-data/templates/acs_pums_oh_narr.j2
User approved: yes
Result: success
```

### DETERMINE_CONDITIONS

```
[2026-04-05 14:30:35] DETERMINE_CONDITIONS
Details: Loaded conditions from experiment_summary.yaml
Conditions:
  dict_full: features=[AGEP, COW, SCHL, MAR, OCCP, SEX, RAC1P, ST], template=dictionary, perturbations=[]
  narr_full: features=[AGEP, COW, SCHL, MAR, OCCP, SEX, RAC1P, ST], template=narrative, perturbations=[]
Shared settings: target=PINCP>50000, seed=42, split_ratio=0.8, subsampling=0.1, missing=skip
Result: success
```

### PLAN_OUTPUT_FILES

```
[2026-04-05 14:30:36] PLAN_OUTPUT_FILES
Details: Determined files to generate
Files:
  1. dict_full_train_a1b2c3d4.json (train split)
  2. dict_full_test_a1b2c3d4.json (test split)
  3. narr_full_test_a1b2c3d4.json (test split)
User confirmed: yes
Result: success
```

### GENERATE_DATASET

```
[2026-04-05 14:31:00] GENERATE_DATASET
Details: Generated dataset file
Condition: dict_full
Split: train
Output: /scratch/gpfs/MSALGANIK/sarahep/ck-data/generated/dict_full_train_a1b2c3d4.json
Rows: 9526
Size: 3830 KB
Duration: 12.3 seconds
Result: success
```

Failure example:

```
[2026-04-05 14:31:00] GENERATE_DATASET
Details: Failed to generate dataset file
Condition: narr_full
Split: test
Output: /scratch/gpfs/MSALGANIK/sarahep/ck-data/generated/narr_full_test_a1b2c3d4.json
Error: FileNotFoundError - Template file not found at /path/to/template.j2
Result: failure
```

### VALIDATE_FILE

```
[2026-04-05 14:32:00] VALIDATE_FILE
Details: Validating generated dataset
File: dict_full_train_a1b2c3d4.json
Checks:
  file_existence: pass
  entry_schema: pass
  metadata_consistency: pass
  label_distribution: pass (0=7700, 1=1826)
  text_quality: pass
Status: PASS
Result: success
```

Warning example:

```
[2026-04-05 14:32:05] VALIDATE_FILE
Details: Validating generated dataset
File: dict_full_test_a1b2c3d4.json
Checks:
  file_existence: pass
  entry_schema: pass
  metadata_consistency: pass
  label_distribution: warn (label "1" is 4.2% of data)
  text_quality: pass
Status: WARN
Warnings: label "1" is <5% of data (100/2382)
Result: warning
```

### VALIDATE_CROSS_FILE

```
[2026-04-05 14:32:10] VALIDATE_CROSS_FILE
Details: Cross-file consistency check
Files checked: 3
Seed consistency: pass (all seed=42)
Source consistency: pass (all source_rows_total=119086)
Schema consistency: pass (all same schema)
Split arithmetic: pass (dict_full train=9526 + test=2382 = 11908, ratio=0.800)
Status: PASS
Result: success
```

### COMPLETE_CONVERSION

```
[2026-04-05 14:33:00] COMPLETE_CONVERSION
Details: Dataset generation finished
Experiment: acs_dict_to_narr_2026-04-05
Files generated: 3
Total rows: 21434 (9526 train + 2x2382 test)
Duration: 180 seconds
Log: /scratch/gpfs/MSALGANIK/sarahep/ck-projects/{project}/acs_dict_to_narr_2026-04-05/logs/convert-tabular-to-text.log
Result: success
```

---

## Error Handling

When `Result: failure` or `Result: warning`, include an `Error:` or `Warnings:` line:

```
[2026-04-05 14:31:00] GENERATE_DATASET
Details: Failed to generate dataset file
Condition: narr_full
Split: test
Error: TemplateError - Undefined variable 'occupation' in template
Result: failure
```

---

## Writing Logs

### Write Entries Incrementally

Write log entries as actions complete. Do not wait until the end to write all logs at once. This ensures the log is useful for debugging if the process is interrupted.

### Timestamp Format

Use `YYYY-MM-DD HH:MM:SS` format:

```
[2026-04-05 14:30:01] ACTION_TYPE
```

### Entry Structure

Each entry follows this pattern:

```
[TIMESTAMP] ACTION_TYPE
Details: {what happened}
{action-specific key: value lines}
Result: success|failure|warning
```

Separate entries with a blank line for readability.

# Logging Specification

**Shared specification for all cruijff_kit skills**

This document defines the common logging format and practices used across design-experiment, scaffold-experiment, and run-experiment skills.

---

## Purpose

Skill logs enable:
1. **Debugging:** Track what actions were taken, what commands were run, and what errors occurred
2. **Reproducibility:** Another person (or Claude) can understand exactly what was done
3. **Improvement:** Review logs to identify better approaches or missing steps
4. **Auditing:** Verify that operations completed correctly before proceeding to next stage
5. **Progress tracking:** Monitor experiment progress over time (for execution logs)

---

## Log Format

All skills use a consistent structured format:

```
[{timestamp}] {ACTION_TYPE}: {brief_description}
Details: {what_happened}
Result: {outcome}
```

**Optional fields** (use when relevant):
- `Command: {shell_command}` - For actions that ran shell commands
- `Calculation: {formula}` - For computational actions
- `Input: {parameters}` - For actions with specific inputs
- `Explanation: {why_this_matters}` - For decisions or complex actions
- `Duration: {elapsed_time}` - For completed stages

### Timestamp Format

**Required:** `YYYY-MM-DD HH:MM:SS`

Example: `2025-11-11 14:23:15`

### Action Type Naming

Action types should be:
- **UPPERCASE_WITH_UNDERSCORES**
- **Descriptive verbs** or **noun_verb** pairs
- **Consistent within a skill** (same action type for same kind of operation)

Examples:
- `VERIFY_MODEL`, `VERIFY_DATASET` (resource checks)
- `SUBMIT_JOB`, `STATE_CHANGE` (execution actions)
- `CREATE_SUMMARY`, `GENERATE_YAMLS` (file creation)
- `CALCULATE_TIME`, `EXTRACT_SPEED` (computation)
- `DECIDE_NAMING`, `SELECT_RUNS` (decision points)

---

## General Logging Guidelines

### What to Log

**DO log:**
- ✓ Commands executed (shell, Python scripts)
- ✓ File operations (creation, modification)
- ✓ Resource verification (checking paths, disk space)
- ✓ Calculations and their inputs
- ✓ Decisions made and their reasoning
- ✓ State changes (job status, workflow stages)
- ✓ Errors and warnings
- ✓ Start and completion of major stages

**DON'T log:**
- ✗ Routine status checks that return no changes
- ✗ Internal implementation details unrelated to user's experiment
- ✗ Redundant information already captured elsewhere

### When to Log

Log at these key points:
1. **Start of major stages** - Log what's about to happen
2. **After significant actions** - Log what was done and the result
3. **On state changes** - Log transitions (PENDING→RUNNING, etc.)
4. **On errors** - Log failures with sufficient detail for debugging
5. **End of stages** - Log summary and duration

---

## Example Log Entry

```
[2025-11-11 14:23:15] VERIFY_MODEL: Checking Llama-3.2-1B-Instruct
Command: ls /scratch/gpfs/models/Llama-3.2-1B-Instruct
Result: Directory exists with 15 files (config.json, model.safetensors, etc.)
Explanation: Verifying base model exists before creating experiment plan
```

---

## Skill-Specific Guidance

Each skill documents in its own `logging.md`:
- **Log file location** - Where the log is created
- **Action types** - Complete list of action types used by that skill
- **Skill-specific examples** - Representative log entries for that workflow
- **When to log** - During which stages/modules to create log entries
- **What to log** - Skill-specific things to include or exclude

See:
- `design-experiment/logging.md` - Planning-specific logging
- `scaffold-experiment/logging.md` - Config generation logging
- `run-experiment/logging.md` - Execution logging
- `analyze-experiment/logging.md` - Visualization logging

---

## Log File Locations

Logs are always created within the experiment directory's `logs/` subdirectory:

```
{experiment_dir}/
└── logs/
    ├── design-experiment.log            # Planning decisions
    ├── scaffold-experiment.log          # Scaffolding orchestration
    ├── scaffold-torchtune.log           # Fine-tuning config generation
    ├── scaffold-inspect.log             # Evaluation config generation
    ├── run-torchtune.log                # Fine-tuning execution
    ├── run-inspect.log                  # Evaluation execution
    ├── analyze-experiment.log           # Visualization generation
    └── summarize-experiment.log         # Results summary
```


---

## Integration Notes

### For Skill Authors

When creating logging guidance for a new skill:
1. Reference this shared spec in your skill's `logging.md`
2. Document only the skill-specific aspects (location, action types, examples, when to log)
3. Don't duplicate the format specification or general guidelines
4. Link back to this file: `See [shared/logging_spec.md](../../shared/logging_spec.md)`

### For Claude Code

When executing a skill:
1. Read this shared spec for format requirements
2. Read the skill-specific `logging.md` for action types and workflow integration
3. Create log entries throughout the workflow at appropriate points
4. Ensure timestamps, action types, and format match the specification

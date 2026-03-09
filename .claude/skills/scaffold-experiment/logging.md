# Logging - scaffold-experiment

This document covers scaffold-experiment-specific logging practices for the orchestration process.

---

## Log File Location

Scaffold-experiment creates an orchestration log:

- `{experiment_dir}/logs/scaffold-experiment.log`

This log records the high-level orchestration process. Individual subagents create their own detailed logs:
- `{experiment_dir}/logs/scaffold-torchtune.log` (created by scaffold-torchtune subagent)
- `{experiment_dir}/logs/scaffold-inspect.log` (created by scaffold-inspect subagent)

**Note on log formats across skills:**
- `.jsonl` (JSON Lines) - Used by design-experiment for structured audit trails (machine-readable, reproducibility)
- `.log` (text) - Used by all other skills for human-readable progress tracking

---

## Log Format

```
[YYYY-MM-DD HH:MM:SS] ACTION: Description
Details: {specifics}
Result: {outcome}

```

---

## What to Log

### During Experiment Discovery
- Experiment directory location
- Presence of experiment_summary.yaml
- Number of runs and evaluations identified

### During Prerequisite Verification
- Which files were checked (experiment_summary.yaml, claude.local.md)
- Whether all prerequisites satisfied
- Any missing files or invalid YAML

### During Tool Specification Parsing
- Preparation and evaluation tools specified
- Which subagents will be launched
- Any unsupported tools encountered

### During Subagent Launch
- Which subagent was launched (scaffold-torchtune, scaffold-inspect)
- What role it serves (preparation/evaluation)
- Timestamp of launch
- Whether parallel execution is being used

### During Subagent Completion
- Which subagent completed
- Success or failure status
- Summary from subagent report (runs configured, evaluations created, etc.)
- Duration of subagent execution
- Path to detailed subagent log for more information

### At Completion
- Total duration of orchestration
- Which subagents were launched
- Final status (all success, partial success, failure)
- Paths to all created log files
- Next steps recommendation

---

## What NOT to Log

**Orchestration log focuses on high-level flow. Details go in subagent logs:**

- Detailed file contents → subagent logs
- Individual file creation events → subagent logs
- Parameter validation details → subagent logs
- Bash command outputs → subagent logs
- Config generation specifics → subagent logs

**Principle:** The orchestration log tracks coordination and timing. Subagent logs contain implementation details.

---

## Action Types

### Orchestration Actions

| Action Type | Purpose | When to Log |
|-------------|---------|-------------|
| `DISCOVER_EXPERIMENT` | Locate experiment directory | After finding experiment_summary.yaml |
| `VERIFY_PREREQUISITES` | Check required files exist | After locating experiment |
| `READ_TOOL_SPECS` | Parse tool specifications from YAML | After reading experiment_summary.yaml |
| `LAUNCH_SUBAGENTS` | Launch worker subagents in parallel | When invoking Task tool for subagents |
| `SCAFFOLD_TORCHTUNE_COMPLETE` | Preparation subagent finished | When scaffold-torchtune reports back |
| `SCAFFOLD_INSPECT_COMPLETE` | Evaluation subagent finished | When scaffold-inspect reports back |
| `COMPLETE` | Finish scaffolding process | At skill completion |

---

## Example Log Entries

### Experiment Discovery

```
[2025-10-24 17:30:00] DISCOVER_EXPERIMENT: Found experiment
Details: /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_comparison/experiment_summary.yaml
Result: Experiment plan ready for scaffolding (3 runs, 1 evaluation task)
```

### Prerequisite Verification

```
[2025-10-24 17:30:05] VERIFY_PREREQUISITES: Checking required files
Details: experiment_summary.yaml exists, claude.local.md found
Result: All prerequisites satisfied
```

### Tool Specification Parsing

```
[2025-10-24 17:30:08] READ_TOOL_SPECS: Parsing tool specifications
Details: Reading Tools section from experiment_summary.yaml
Result: Preparation tool = torchtune, Evaluation tool = inspect-ai
Explanation: Will launch scaffold-torchtune and scaffold-inspect subagents
```

### Subagent Launch

```
[2025-10-24 17:30:10] LAUNCH_SUBAGENTS: Starting parallel scaffolding
Details: Launching scaffold-torchtune and scaffold-inspect in parallel
Result: Both subagents launched at 2025-10-24 17:30:10
```

### Preparation Subagent Completion

```
[2025-10-24 17:31:30] SCAFFOLD_TORCHTUNE_COMPLETE: Fine-tuning configs generated
Details: 2 fine-tuned runs scaffolded successfully, 1 control run (directory only)
Duration: 1m 20s
Result: See scaffold-torchtune.log for detailed process
Outputs: Llama-3.2-1B-Instruct_rank4/, Llama-3.2-1B-Instruct_rank8/, Llama-3.2-1B-Instruct_base/
```

### Evaluation Subagent Completion

```
[2025-10-24 17:31:35] SCAFFOLD_INSPECT_COMPLETE: Evaluation configs generated
Details: 3 evaluation scripts created successfully (0 failures)
Duration: 1m 25s
Result: See scaffold-inspect.log for detailed process
Outputs: {run_dir}/eval/ directories with SLURM scripts
```

### Overall Completion

```
[2025-10-24 17:31:40] COMPLETE: Experiment scaffolding finished
Summary: All configs generated successfully
- Fine-tuning: 2 fine-tuned runs + 1 control run ready
- Evaluation: 3 evaluation scripts ready
Next: User can proceed with run-experiment skill to execute workflow
```

---

## Error Logging Examples

### Missing Prerequisite

```
[2025-10-24 17:30:05] VERIFY_PREREQUISITES: Missing required file
Details: experiment_summary.yaml not found in /path/to/experiment
Result: FAILURE - Cannot proceed without experiment plan
Action: Suggest user run design-experiment skill first
```

### Subagent Failure

```
[2025-10-24 17:31:30] SCAFFOLD_TORCHTUNE_COMPLETE: Fine-tuning scaffolding failed
Details: scaffold-torchtune subagent reported errors
Duration: 1m 20s
Result: FAILURE - See scaffold-torchtune.log for error details
Recommendation: Check experiment_summary.yaml for completeness, verify claude.local.md settings
```

### Partial Success

```
[2025-10-24 17:31:40] COMPLETE: Experiment scaffolding partially successful
Summary: Fine-tuning configs generated, evaluation scaffolding failed
- Fine-tuning: 2 runs ready (scaffold-torchtune succeeded)
- Evaluation: Failed (scaffold-inspect encountered errors)
Next: User can proceed with fine-tuning only, or fix evaluation issues and re-run scaffold-inspect
```
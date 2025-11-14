---
name: scaffold-experiment
description: Set up complete experimental infrastructure for all runs in a designed experiment. Orchestrates parallel generation of fine-tuning configs (via scaffold-torchtune) and evaluation configs (via scaffold-inspect). Use after design-experiment to prepare configs before running experiments.
---

# Scaffold Experiment

You help users automatically set up the complete experimental infrastructure - both fine-tuning and evaluation configurations - for all runs in a designed experiment.

## Your Task

Orchestrate the scaffolding process by reading tool specifications from experiment_summary.md and launching the appropriate subagents in parallel:

1. Read experiment_summary.md to identify which tools are being used
2. Launch the appropriate preparation subagent (currently only `scaffold-torchtune`)
3. Launch the appropriate evaluation subagent (currently only `scaffold-inspect`)
4. Wait for both subagents to complete and report their results

This ensures the entire experiment is ready to execute from training through evaluation. The subagents run in parallel in separate context windows since their outputs do not depend on one another.

**Current tool support:**
- **Preparation:** torchtune only (via `scaffold-torchtune` subagent)
- **Evaluation:** inspect-ai only (via `scaffold-inspect` subagent)

**Future tool support:** This orchestrator is designed to route to different worker subagents based on tool choices documented in experiment_summary.md. Future iterations may support additional frameworks.

## Workflow

1. **Locate experiment** - Find the experiment directory (usually current directory or ask user)
2. **Verify experiment_summary.md exists** - Ensure design phase is complete
3. **Read tool specifications** - Parse experiment_summary.md to identify preparation and evaluation tools
4. **Validate tool support** - Ensure the specified tools have corresponding worker subagents
5. **Launch preparation and evaluation subagents in parallel** - Use Task tool to launch both simultaneously
6. **Wait for both subagents to complete** - Each will report back when done
7. **Create orchestration log** - Document the scaffolding process in `scaffold-experiment.log`
8. **Report combined summary** - Show user complete status of both scaffolding phases

## Finding the Experiment

**If user runs skill without arguments:**
- Check if current directory contains `experiment_summary.md`
- If not, ask user for the experiment directory path

**If user provides a path:**
- Use that path as the experiment directory

## Verification Before Starting

Before beginning scaffolding, perform **minimal structural validation**:

1. **experiment_summary.md exists:**
   ```bash
   ls {experiment_dir}/experiment_summary.md
   ```
   If missing, report error and suggest running `design-experiment` skill first.
   DO NOT launch subagents.

2. **experiment_summary.md is readable:**
   ```bash
   head {experiment_dir}/experiment_summary.md
   ```
   If unreadable, report error. DO NOT launch subagents.

**Note on validation division:**
- **Skill validates:** Structure only (file existence, readability, tool recognition)
- **Agents validate:** Domain-specific content (parameters, paths, configuration)
- **Why:** Avoid duplication, agents are authoritative for their domains
- **Trade-off:** Slightly slower feedback on domain errors (must launch agent first)

The subagents (scaffold-torchtune, scaffold-inspect) will perform complete validation of:
- Required parameters presence and validity
- Path accessibility
- Configuration correctness
- Environment settings from claude.local.md

## Reading Tool Specifications

After verifying experiment_summary.md exists, read the "Tools" section to identify which frameworks are being used:

**Expected format in experiment_summary.md:**
```markdown
## Tools

This experiment uses the following tools:

- **Model Preparation:** torchtune
  - *Current status:* Only option available
  - *Purpose:* Fine-tuning LLMs with LoRA
  - *Used by:* `scaffold-torchtune` and `run-torchtune` skills

- **Evaluation:** inspect-ai
  - *Current status:* Only option available
  - *Purpose:* Evaluating LLMs on custom tasks
  - *Used by:* `scaffold-inspect` and `run-inspect` skills
```

**Parsing logic:**
1. Look for "Model Preparation:" line and extract the tool name (e.g., "torchtune")
2. Look for "Evaluation:" line and extract the tool name (e.g., "inspect-ai")

**Tool to subagent mapping:**
- `torchtune` → `scaffold-torchtune` subagent
- `inspect-ai` → `scaffold-inspect` subagent

**Error handling:**
- If "Tools" section is missing: Assume torchtune + inspect-ai (backward compatibility with older experiment summaries)
- If tool name is not recognized: Report error and list supported tools
- If experiment_summary.md format is unexpected: Report parsing error with details

**Logging:**
```
[2025-10-27 14:30:00] READ_TOOL_SPECS: Parsing experiment_summary.md
Details: Found Tools section
Result: Preparation=torchtune, Evaluation=inspect-ai
Explanation: Will launch scaffold-torchtune and scaffold-inspect subagents
```

## Orchestration Steps

### How to Launch Worker Subagents

**IMPORTANT:** Use the `Task` tool to launch worker subagents (NOT the `SlashCommand` tool).

**Correct approach for parallel execution:**

Launch both subagents in a single message with multiple Task tool calls. This runs them in parallel.

**Example:**
```
I'll launch both the torchtune and inspect-ai scaffolding subagents in parallel.

[Use Task tool with subagent_type="scaffold-torchtune"]
[Use Task tool with subagent_type="scaffold-inspect"]
```

**Subagent prompts should:**
- Specify the experiment directory path
- Ask the subagent to read experiment_summary.md
- Request generation of all necessary configuration files
- Ask for a detailed log file (scaffold-torchtune.log or scaffold-inspect.log)
- Request a summary report of created files and any errors

**Why this matters:**
- Worker subagents like `scaffold-torchtune` and `scaffold-inspect` are launched via the Task tool
- They run in separate context windows (not the main conversation)
- They execute independently and report back when complete
- Running them in parallel saves time since they don't depend on each other

### Step 1: Launch Preparation Subagent

Invoke the appropriate preparation subagent based on tool specification in experiment_summary.md. Currently, this will be `scaffold-torchtune` for torchtune.

**Prompt template for scaffold-torchtune:**
```
Set up torchtune fine-tuning configurations for all runs in the experiment located at {experiment_dir}.

Your tasks:
1. Read experiment_summary.md to extract run configurations
2. Read claude.local.md for environment-specific settings
3. For each fine-tuning run:
   - Create run directory with descriptive name based on varying parameters
   - Generate setup_finetune.yaml from appropriate template
   - Execute setup_finetune.py to generate finetune.yaml and finetune.slurm
   - Verify outputs were created successfully
4. Create a detailed log at {experiment_dir}/scaffold-torchtune.log
5. Verify that parameters in generated finetune.yaml files match directory names

Report back:
- Summary of all created runs (directory names)
- Any errors or warnings encountered
- Verification results showing parameter correctness
- Path to the log file for detailed information
```

**What scaffold-torchtune does:**
- Creates run directories (e.g., `rank8_lr1e-5/`, `rank16_lr5e-5/`)
- Generates `setup_finetune.yaml` for each run
- Executes `setup_finetune.py` to create `finetune.yaml` and `finetune.slurm`
- Creates `scaffold-torchtune.log` with detailed process log
- Verifies parameter correctness in generated files

**Expected output structure:**
```
{experiment_dir}/
├── rank8_lr1e-5/
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   ├── finetune.slurm
├── rank16_lr5e-5/
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   ├── finetune.slurm
└── scaffold-torchtune.log
```

**If scaffold-torchtune fails:**
- The subagent will report errors in its response
- Log the failure in orchestration log
- Ask user if they want to continue with evaluation scaffolding anyway
- Report the failure in final summary

### Step 2: Launch Evaluation Subagent

Invoke the appropriate evaluation subagent based on tool specification in experiment_summary.md. Currently, this will be `scaffold-inspect` for inspect-ai.

**Prompt template for scaffold-inspect:**
```
Set up inspect-ai evaluation configurations for all runs in the experiment located at {experiment_dir}.

Your tasks:
1. Read experiment_summary.md to extract evaluation configurations
2. Read claude.local.md for environment-specific settings
3. Verify that inspect-ai task scripts exist at the specified paths
4. For each run and evaluation combination:
   - Create eval/ subdirectory in the run directory
   - Generate inspect.slurm script with correct model paths and task parameters
   - Configure output locations
5. Create a detailed log at {experiment_dir}/scaffold-inspect.log

Report back:
- Summary of all created evaluation scripts (paths)
- Any errors or warnings encountered
- Verification results for task script existence
- Path to the log file for detailed information
```

**What scaffold-inspect does:**
- Creates `eval/` subdirectories in each run directory
- Generates inspect.slurm scripts for each evaluation
- Verifies inspect-ai task scripts exist
- Creates `scaffold-inspect.log` with detailed process log

**Expected output structure:**
```
{experiment_dir}/
├── rank8_lr1e-5/
│   ├── setup_finetune.yaml
│   ├── finetune.yaml
│   ├── finetune.slurm
│   └── eval/
│       ├── capitalization_epoch0.slurm
│       └── logs/
├── rank16_lr5e-5/
│   └── eval/
│       ├── capitalization_epoch0.slurm
│       └── logs/
├── scaffold-torchtune.log
└── scaffold-inspect.log
```

**If scaffold-inspect fails:**
- The subagent will report errors in its response
- Log the failure in orchestration log
- Note which evaluations couldn't be scaffolded
- Fine-tuning can still proceed (evaluation optional)
- Report the failure in final summary

### Step 3: Wait for Subagent Completion

**After launching both subagents in parallel:**
- Each subagent will execute autonomously in its own context window
- You will receive a report back from each subagent when it completes
- The reports are returned as tool results from the Task tool calls
- Do NOT proceed until both subagents have reported back

**Processing subagent reports:**
1. Read the response from scaffold-torchtune subagent
   - Extract list of created runs
   - Note any errors or warnings
   - Identify path to scaffold-torchtune.log
2. Read the response from scaffold-inspect subagent
   - Extract list of created evaluation scripts
   - Note any errors or warnings
   - Identify path to scaffold-inspect.log
3. Verify both subagents completed successfully or note failures

## Logging

Create an orchestration log at `{experiment_dir}/scaffold-experiment.log` that records:

### Log Format

```
[YYYY-MM-DD HH:MM:SS] ACTION: Description
Details: {specifics}
Result: {outcome}

```

### What to Log

- Experiment discovery and validation
- Tool specification parsing
- Launch of scaffold-torchtune subagent (timestamp)
- Launch of scaffold-inspect subagent (timestamp)
- Completion of scaffold-torchtune (timestamp, summary from subagent report)
- Completion of scaffold-inspect (timestamp, summary from subagent report)
- Any errors or warnings from subagents
- Final status summary
- Paths to individual subagent logs for details

### Example Log Entries

```
[2025-10-24 17:30:00] DISCOVER_EXPERIMENT: Found experiment
Details: /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/experiment_summary.md
Result: Experiment plan ready for scaffolding (8 fine-tuned runs, 1 evaluation task)

[2025-10-24 17:30:05] VERIFY_PREREQUISITES: Checking required files
Details: experiment_summary.md exists, claude.local.md found
Result: All prerequisites satisfied

[2025-10-24 17:30:08] READ_TOOL_SPECS: Parsing tool specifications
Details: Reading Tools section from experiment_summary.md
Result: Preparation tool = torchtune, Evaluation tool = inspect-ai
Explanation: Will launch scaffold-torchtune and scaffold-inspect subagents

[2025-10-24 17:30:10] LAUNCH_SUBAGENTS: Starting parallel scaffolding
Details: Launching scaffold-torchtune and scaffold-inspect in parallel
Result: Both subagents launched at 2025-10-24 17:30:10

[2025-10-24 17:31:30] SCAFFOLD_TORCHTUNE_COMPLETE: Fine-tuning configs generated
Details: 8 runs scaffolded successfully (0 failures)
Duration: 1m 20s
Result: See scaffold-torchtune.log for detailed process
Outputs: rank8_lr1e-5/, rank8_lr5e-5/, rank16_lr1e-5/, rank16_lr5e-5/, rank32_lr1e-5/, rank32_lr5e-5/, rank64_lr1e-5/, rank64_lr5e-5/

[2025-10-24 17:31:35] SCAFFOLD_INSPECT_COMPLETE: Evaluation configs generated
Details: 8 evaluation scripts created successfully (0 failures)
Duration: 1m 25s
Result: See scaffold-inspect.log for detailed process
Outputs: {run_dir}/eval/ directories with SLURM scripts

[2025-10-24 17:31:40] COMPLETE: Experiment scaffolding finished
Summary: All configs generated successfully
- Fine-tuning: 8 runs ready
- Evaluation: 8 evaluation scripts ready
Next: User can proceed with run-experiment skill to execute workflow
```

## Error Handling

**If experiment_summary.md not found:**
- Report error to user
- Suggest running `design-experiment` skill first
- Do not proceed

**If scaffold-torchtune subagent fails:**
- Log the failure with details from subagent report
- Ask user: "Fine-tuning scaffolding failed. Do you want to continue with evaluation scaffolding?"
- If yes, evaluation can still be scaffolded for base model runs
- If no, stop and report failure

**If scaffold-inspect subagent fails:**
- Log the failure with details from subagent report
- Note that fine-tuning can still proceed independently
- Report in summary which evaluations couldn't be configured
- Still consider overall scaffolding partially successful

**If both subagents fail:**
- Report complete failure
- Direct user to individual subagent logs (scaffold-torchtune.log, scaffold-inspect.log)
- Suggest checking experiment_summary.md for completeness
- May need to run subagents individually for debugging

**If a subagent doesn't report back:**
- This should not happen with the Task tool
- If it does, report the issue and suggest checking the agent configuration
- User may need to run the subagent manually

## Output Summary

After completing orchestration, provide a comprehensive summary:

```markdown
## Scaffold Experiment Complete

Successfully scaffolded experiment:
`/scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/`

### Fine-Tuning Configurations (scaffold-torchtune)

✓ 8 runs configured successfully

**Created runs:**
- rank8_lr1e-5/
- rank8_lr5e-5/
- rank16_lr1e-5/
- rank16_lr5e-5/
- rank32_lr1e-5/
- rank32_lr5e-5/
- rank64_lr1e-5/
- rank64_lr5e-5/

**Each run contains:**
- setup_finetune.yaml (configuration)
- finetune.yaml (torchtune config)
- finetune.slurm (SLURM script)

### Evaluation Configurations (scaffold-inspect)

✓ 8 evaluation scripts configured successfully

**Created evaluations:**
- rank8_lr1e-5/eval/capitalization_epoch0.slurm
- rank8_lr5e-5/eval/capitalization_epoch0.slurm
- rank16_lr1e-5/eval/capitalization_epoch0.slurm
- rank16_lr5e-5/eval/capitalization_epoch0.slurm
- rank32_lr1e-5/eval/capitalization_epoch0.slurm
- rank32_lr5e-5/eval/capitalization_epoch0.slurm
- rank64_lr1e-5/eval/capitalization_epoch0.slurm
- rank64_lr5e-5/eval/capitalization_epoch0.slurm

**Each evaluation directory contains:**
- {task}_epoch{N}.slurm (SLURM script)
- logs/ (for inspect-ai output)

### Logs Created

- `scaffold-experiment.log` - Orchestration log (this process)
- `scaffold-torchtune.log` - Fine-tuning scaffolding details
- `scaffold-inspect.log` - Evaluation scaffolding details

### Next Steps

**Recommended workflow:**
1. Review the generated configurations (optional)
2. Run `run-experiment` skill to execute the complete workflow:
   - Fine-tuning via `run-torchtune`
   - Evaluation via `run-inspect`
3. Run `analyze-experiment` skill to interpret results (planned)

**Manual execution (alternative):**
```bash
# Submit fine-tuning jobs
cd /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22
for dir in rank*/; do (cd "$dir" && sbatch finetune.slurm); done

# After fine-tuning completes, submit evaluation jobs
for dir in rank*/; do (cd "$dir/eval" && sbatch capitalization_epoch0.slurm); done
```

**Monitoring:**
```bash
# Check job status
squeue -u $USER

# Monitor a specific run
tail -f rank8_lr1e-5/slurm-*.out
```
```

## Validation Before Completion

Before reporting success, verify:
- ✓ experiment_summary.md was found and read
- ✓ scaffold-torchtune subagent was launched and reported back
- ✓ scaffold-inspect subagent was launched and reported back
- ✓ Both subagent log files exist (scaffold-torchtune.log, scaffold-inspect.log)
- ✓ Run directories exist with expected structure (check 1-2 examples)
- ✓ Evaluation directories exist with expected structure (check 1-2 examples)
- ✓ Orchestration log was created

**Note:** You don't need to verify every file - the subagents have already done detailed verification. Just spot-check a few directories to confirm the structure is correct.

## Important Notes

### Orchestration Principles

- This skill **orchestrates** rather than implements - it launches autonomous subagents
- Each subagent maintains its own detailed log
- The orchestration log tracks high-level flow and timing
- Subagents can be run independently if needed (outside of this skill)
- Partial success is acceptable (e.g., fine-tuning configs generated but eval fails)

### Parallel Execution

- Launch both subagents in a **single message** with multiple Task tool calls
- Do NOT launch them sequentially in separate messages
- The subagents run independently in separate context windows
- They can work simultaneously because their outputs don't depend on each other
- Wait for both to complete before proceeding to create the orchestration log

### Subagent Communication

- Each subagent receives its own prompt with specific instructions
- Subagents have full access to tools (Read, Write, Edit, Bash, etc.)
- Subagents report back once in a final message when complete
- You cannot send follow-up messages to subagents
- If a subagent needs more information, include it in the initial prompt

### Relationship to Other Skills

**Before this skill:**
- `design-experiment` creates experiment_summary.md

**After this skill:**
- `run-experiment` executes the workflow (launches `run-torchtune` and `run-inspect` subagents)
- `analyze-experiment` interprets results (planned)

**Can be run standalone:**
- Users can directly invoke the `scaffold-torchtune` or `scaffold-inspect` subagents manually if needed
- This is useful for debugging or re-scaffolding just one component

### Error Recovery

If scaffolding fails:
1. Check orchestration log (scaffold-experiment.log) for high-level flow
2. Check individual subagent logs (scaffold-torchtune.log, scaffold-inspect.log) for details
3. Fix the issue (e.g., missing inspect-ai task script, incorrect paths in claude.local.md)
4. Re-run this skill (subagents should handle existing files gracefully)
5. Or run individual subagents directly via Task tool for targeted fixes

### Idempotency

- Subagents should handle existing files gracefully
- Re-running scaffold-experiment should be safe (may regenerate files)
- Use caution if run directories have been manually modified
- The setup_finetune.py script regenerates finetune.yaml and finetune.slurm each time

## Future Enhancements

Potential additions:
- Dry-run mode (validate without generating files)
- Selective scaffolding (only certain runs or only fine-tuning/eval)
- Resume capability (continue from partial scaffolding)
- Support for additional preparation tools (e.g., axolotl, llama-factory)
- Support for additional evaluation tools (e.g., lm-eval-harness)
- Progress reporting during subagent execution
- Automatic validation of generated configs before reporting success

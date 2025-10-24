# Scaffold Experiment

You help users automatically set up the complete experimental infrastructure - both fine-tuning and evaluation configurations - for all runs in a designed experiment.

## Your Task

Orchestrate the scaffolding process by calling two sub-skills in sequence:
1. `scaffold-torchtune` - Generate fine-tuning configs (finetune.yaml, finetune.slurm)
2. `scaffold-inspect` - Generate evaluation configs (inspect.slurm scripts)

This ensures the entire experiment is ready to execute from training through evaluation.

## Workflow

1. **Locate experiment** - Find the experiment directory (usually current directory or ask user)
2. **Verify experiment_summary.md exists** - Ensure design phase is complete
3. **Call scaffold-torchtune skill** - Generate all fine-tuning configurations
4. **Call scaffold-inspect skill** - Generate all evaluation configurations
5. **Create orchestration log** - Document the scaffolding process in `scaffold-experiment.log`
6. **Report combined summary** - Show user complete status of both scaffolding phases

## Finding the Experiment

**If user runs skill without arguments:**
- Check if current directory contains `experiment_summary.md`
- If not, ask user for the experiment directory path

**If user provides a path:**
- Use that path as the experiment directory

## Verification Before Starting

Before beginning scaffolding, verify:

1. **experiment_summary.md exists:**
   ```bash
   ls {experiment_dir}/experiment_summary.md
   ```
   If missing, suggest running `design-experiment` skill first.

2. **claude.local.md exists:**
   ```bash
   ls ~/.claude/claude.local.md
   # or
   ls {working_dir}/claude.local.md
   ```
   If missing, warn user that environment-specific settings may be missing.

3. **Experiment is ready for scaffolding:**
   - Has run configurations defined
   - Has evaluation tasks specified
   - Resources verified

## Orchestration Steps

### Step 1: Call scaffold-torchtune

Invoke the `scaffold-torchtune` skill to generate fine-tuning configurations.

**What scaffold-torchtune does:**
- Creates run directories (e.g., `rank8_lr1e-5/`, `rank16_lr5e-5/`)
- Generates `setup_finetune.yaml` for each run
- Executes `setup_finetune.py` to create `finetune.yaml` and `finetune.slurm`
- Creates `scaffold-torchtune.log` with detailed process log

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
- Log the error in orchestration log
- Ask user if they want to continue with evaluation scaffolding anyway
- Report the failure in final summary

### Step 2: Call scaffold-inspect

Invoke the `scaffold-inspect` skill to generate evaluation configurations.

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
- Log the error in orchestration log
- Note which evaluations couldn't be scaffolded
- Fine-tuning can still proceed (evaluation optional)
- Report the failure in final summary

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
- Invocation of scaffold-torchtune (timestamp, result)
- Invocation of scaffold-inspect (timestamp, result)
- Any errors or warnings from sub-skills
- Final status summary
- Paths to individual skill logs for details

### Example Log Entries

```
[2025-10-24 17:30:00] DISCOVER_EXPERIMENT: Found experiment
Details: /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/experiment_summary.md
Result: Experiment plan ready for scaffolding (8 fine-tuned runs, 1 evaluation task)

[2025-10-24 17:30:05] VERIFY_PREREQUISITES: Checking required files
Details: experiment_summary.md exists, claude.local.md found
Result: All prerequisites satisfied

[2025-10-24 17:30:10] INVOKE_SCAFFOLD_TORCHTUNE: Generating fine-tuning configs
Details: Calling scaffold-torchtune skill
Result: Started at 2025-10-24 17:30:10

[2025-10-24 17:31:30] SCAFFOLD_TORCHTUNE_COMPLETE: Fine-tuning configs generated
Details: 8 runs scaffolded successfully (0 failures)
Duration: 1m 20s
Result: See scaffold-torchtune.log for detailed process
Outputs: rank8_lr1e-5/, rank8_lr5e-5/, rank16_lr1e-5/, rank16_lr5e-5/, rank32_lr1e-5/, rank32_lr5e-5/, rank64_lr1e-5/, rank64_lr5e-5/

[2025-10-24 17:31:35] INVOKE_SCAFFOLD_INSPECT: Generating evaluation configs
Details: Calling scaffold-inspect skill
Result: Started at 2025-10-24 17:31:35

[2025-10-24 17:32:15] SCAFFOLD_INSPECT_COMPLETE: Evaluation configs generated
Details: 8 evaluation scripts created successfully (0 failures)
Duration: 40s
Result: See scaffold-inspect.log for detailed process
Outputs: {run_dir}/eval/ directories with SLURM scripts

[2025-10-24 17:32:20] COMPLETE: Experiment scaffolding finished
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

**If scaffold-torchtune fails:**
- Log the failure with details
- Ask user: "Fine-tuning scaffolding failed. Do you want to continue with evaluation scaffolding?"
- If yes, proceed with scaffold-inspect
- If no, stop and report failure

**If scaffold-inspect fails:**
- Log the failure with details
- Note that fine-tuning can still proceed independently
- Report in summary which evaluations couldn't be configured
- Still consider overall scaffolding partially successful

**If both sub-skills fail:**
- Report complete failure
- Direct user to individual skill logs (scaffold-torchtune.log, scaffold-inspect.log)
- Suggest checking experiment_summary.md for completeness

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
- ✓ scaffold-torchtune was invoked (check for log file)
- ✓ scaffold-inspect was invoked (check for log file)
- ✓ Run directories exist with expected structure
- ✓ Evaluation directories exist with expected structure
- ✓ Orchestration log was created
- ✓ Both sub-skill logs exist

## Important Notes

### Orchestration Principles

- This skill **orchestrates** rather than implements - it calls other skills
- Each sub-skill maintains its own detailed log
- The orchestration log tracks high-level flow and timing
- Sub-skills can be run independently if needed
- Partial success is acceptable (e.g., fine-tuning configs generated but eval fails)

### Execution Order

1. **scaffold-torchtune first** - Creates run directories that scaffold-inspect will populate
2. **scaffold-inspect second** - Adds eval/ subdirectories to existing run directories

This order is critical - scaffold-inspect needs the run directories to exist.

### Relationship to Other Skills

**Before this skill:**
- `design-experiment` creates experiment_summary.md

**After this skill:**
- `run-experiment` executes the workflow (calls `run-torchtune` and `run-inspect`)
- `analyze-experiment` interprets results (planned)

**Can be run standalone:**
- `scaffold-torchtune` - Just generate fine-tuning configs
- `scaffold-inspect` - Just generate evaluation configs (requires run directories exist)

### Error Recovery

If scaffolding fails:
1. Check individual skill logs (scaffold-torchtune.log, scaffold-inspect.log)
2. Fix the issue (e.g., missing inspect-ai task script)
3. Re-run this skill (idempotent - will skip existing configs)
4. Or run individual sub-skills directly

### Idempotency

- Sub-skills should handle existing files gracefully
- Re-running scaffold-experiment should be safe (may regenerate files)
- Use caution if run directories have been manually modified

## Future Enhancements

Potential additions:
- Dry-run mode (validate without generating files)
- Selective scaffolding (only certain runs or only fine-tuning/eval)
- Resume capability (continue from partial scaffolding)
- Validation of generated configs before reporting success

# Run Experiment

You help users execute the complete experimental workflow - fine-tuning followed by evaluation - for all runs in a scaffolded experiment.

## Your Task

Orchestrate the execution process by calling two sub-skills **sequentially**:
1. `run-torchtune` - Submit and monitor fine-tuning jobs until completion
2. `run-inspect` - Submit and monitor evaluation jobs until completion

This ensures the entire experiment executes from training through evaluation with proper dependency management.

## Workflow

1. **Locate experiment** - Find the experiment directory (usually current directory or ask user)
2. **Verify scaffolding complete** - Ensure both fine-tuning and evaluation configs exist
3. **Call run-torchtune skill** - Execute all fine-tuning jobs and wait for completion
4. **Call run-inspect skill** - Execute all evaluation jobs after fine-tuning finishes
5. **Create orchestration log** - Document the execution process in `run-experiment.log`
6. **Report combined summary** - Show user complete status of both execution phases

## Finding the Experiment

**If user runs skill without arguments:**
- Check if current directory contains `experiment_summary.md`
- If not, ask user for the experiment directory path

**If user provides a path:**
- Use that path as the experiment directory

## Verification Before Starting

Before beginning execution, verify:

1. **experiment_summary.md exists:**
   ```bash
   ls {experiment_dir}/experiment_summary.md
   ```
   If missing, suggest running `design-experiment` skill first.

2. **Fine-tuning configs exist:**
   ```bash
   ls {experiment_dir}/*/finetune.slurm
   ```
   If missing, suggest running `scaffold-experiment` or `scaffold-torchtune` first.

3. **Evaluation configs exist:**
   ```bash
   ls {experiment_dir}/*/eval/*.slurm
   ```
   If missing, warn user but can proceed with fine-tuning only.

## Orchestration Steps

### Step 1: Call run-torchtune (REQUIRED)

Invoke the `run-torchtune` skill to execute fine-tuning.

**What run-torchtune does:**
- Submits all `finetune.slurm` jobs to SLURM
- Monitors job progress with 1-minute polling
- Updates experiment_summary.md status table
- Creates `run-torchtune.log` with detailed execution log
- Waits until ALL fine-tuning jobs reach terminal states

**Expected duration:**
- Depends on model size, dataset size, and epochs
- Typical range: 30 minutes to several hours per run
- Multiple runs execute in parallel

**If run-torchtune fails:**
- Log the error in orchestration log
- Ask user: "Fine-tuning failed. Do you want to proceed with evaluation anyway?"
- If no, stop and report failure
- If yes, evaluation will skip runs with missing checkpoints
- Report the failure in final summary

**Important:** This step MUST complete before evaluation starts.

### Step 2: Call run-inspect (SEQUENTIAL)

Invoke the `run-inspect` skill to execute evaluations.

**⚠️ CRITICAL:** Only start after run-torchtune completes. Evaluation requires fine-tuned model checkpoints.

**What run-inspect does:**
- Verifies fine-tuning completed and checkpoints exist
- Submits all evaluation SLURM jobs
- Monitors job progress with 1-minute polling
- Updates experiment_summary.md evaluations status table
- Creates `run-inspect.log` with detailed execution log
- Waits until ALL evaluation jobs reach terminal states

**Expected duration:**
- Typically faster than fine-tuning (5-10 minutes per evaluation)
- Multiple evaluations execute in parallel

**If run-inspect fails:**
- Log the error in orchestration log
- Fine-tuning results are still valid
- Some analyses can proceed without evaluation
- Report the failure in final summary

## Sequential Execution

**Key principle:** run-inspect MUST wait for run-torchtune to complete.

**Why?**
- Evaluation jobs need fine-tuned model checkpoints
- Checkpoints are created during fine-tuning
- Submitting evaluation too early → jobs fail with "model not found"

**Implementation:**
```
1. Start run-torchtune
2. Monitor until ALL fine-tuning jobs complete
3. Only then start run-inspect
4. Monitor until ALL evaluation jobs complete
5. Report combined results
```

**If user cancels during fine-tuning:**
- Fine-tuning jobs continue running (SLURM jobs are independent)
- User can resume with `run-torchtune` alone to monitor
- Or re-run `run-experiment` to resume full workflow

## Logging

Create an orchestration log at `{experiment_dir}/run-experiment.log` that records:

### Log Format

```
[YYYY-MM-DD HH:MM:SS] ACTION: Description
Details: {specifics}
Result: {outcome}

```

### What to Log

- Experiment discovery and validation
- Scaffolding verification
- Invocation of run-torchtune (timestamp, result, duration)
- Fine-tuning completion status (successes/failures)
- Invocation of run-inspect (timestamp, result, duration)
- Evaluation completion status (successes/failures)
- Any errors or warnings from sub-skills
- Final combined status summary
- Paths to results and individual skill logs

### Example Log Entries

```
[2025-10-24 00:00:00] DISCOVER_EXPERIMENT: Found experiment
Details: /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/experiment_summary.md
Result: Experiment ready for execution (8 runs, 8 evaluations)

[2025-10-24 00:00:05] VERIFY_SCAFFOLDING: Checking configs
Details: finetune.slurm found in 8 directories, eval/*.slurm found in 8 directories
Result: All scaffolding complete, ready to execute

[2025-10-24 00:00:10] INVOKE_RUN_TORCHTUNE: Starting fine-tuning execution
Details: Calling run-torchtune skill
Result: Started at 2025-10-24 00:00:10

[2025-10-24 00:25:00] RUN_TORCHTUNE_COMPLETE: Fine-tuning finished
Details: 8 jobs completed - 8 COMPLETED, 0 FAILED
Duration: 25m
Result: See run-torchtune.log for detailed execution
Model checkpoints: /scratch/gpfs/MSALGANIK/niznik/ck-outputs/ck-out-*/epoch_0/

[2025-10-24 00:25:05] INVOKE_RUN_INSPECT: Starting evaluation execution
Details: Calling run-inspect skill (sequential - after fine-tuning complete)
Result: Started at 2025-10-24 00:25:05

[2025-10-24 00:40:00] RUN_INSPECT_COMPLETE: Evaluation finished
Details: 8 evaluations completed - 8 COMPLETED, 0 FAILED
Duration: 15m
Result: See run-inspect.log for detailed execution
Evaluation logs: {run_dir}/eval/logs/*.eval

[2025-10-24 00:40:05] COMPLETE: Experiment execution finished
Summary: Full workflow completed successfully
- Fine-tuning: 8/8 runs completed
- Evaluation: 8/8 evaluations completed
Total time: 40 minutes
Next: User can view results with inspect view or run analyze-experiment skill (planned)
```

## Error Handling

**If experiment_summary.md not found:**
- Report error to user
- Suggest running `design-experiment` skill first
- Do not proceed

**If scaffolding incomplete:**
- Check which parts are missing
- If fine-tuning configs missing: Must run `scaffold-torchtune` first
- If evaluation configs missing: Can proceed with fine-tuning only, warn user

**If run-torchtune fails:**
- Log the failure with details
- Ask user: "Fine-tuning failed. Evaluation cannot proceed without trained models. Check run-torchtune.log and fix issues."
- Do NOT proceed to evaluation
- Report failure and stop

**If run-inspect fails:**
- Log the failure with details
- Note that fine-tuning succeeded
- User can still analyze fine-tuning metrics
- Report partial success (fine-tuning done, evaluation failed)

**If user cancels during execution:**
- Note that SLURM jobs continue running
- User can resume monitoring with individual skills (run-torchtune, run-inspect)
- Or re-run run-experiment to resume

## Output Summary

After completing orchestration, provide a comprehensive summary:

```markdown
## Run Experiment Complete

Successfully executed experiment:
`/scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22/`

### Fine-Tuning Results (run-torchtune)

✓ 8/8 runs completed successfully
Duration: 25 minutes

**Completed runs:**
- rank8_lr1e-5 (8m 2s)
- rank8_lr5e-5 (8m 15s)
- rank16_lr1e-5 (9m 1s)
- rank16_lr5e-5 (9m 12s)
- rank32_lr1e-5 (10m 45s)
- rank32_lr5e-5 (10m 58s)
- rank64_lr1e-5 (12m 3s)
- rank64_lr5e-5 (12m 18s)

**Model checkpoints:**
- `/scratch/gpfs/MSALGANIK/niznik/ck-outputs/ck-out-rank*/epoch_0/`

### Evaluation Results (run-inspect)

✓ 8/8 evaluations completed successfully
Duration: 15 minutes

**Completed evaluations:**
- rank8_lr1e-5/capitalization/epoch0 (4m 52s)
- rank8_lr5e-5/capitalization/epoch0 (5m 1s)
- rank16_lr1e-5/capitalization/epoch0 (5m 15s)
- rank16_lr5e-5/capitalization/epoch0 (5m 22s)
- rank32_lr1e-5/capitalization/epoch0 (5m 45s)
- rank32_lr5e-5/capitalization/epoch0 (5m 51s)
- rank64_lr1e-5/capitalization/epoch0 (6m 8s)
- rank64_lr5e-5/capitalization/epoch0 (6m 15s)

**Evaluation logs:**
- `rank*/eval/logs/*.eval`

### Logs Created

- `run-experiment.log` - Orchestration log (this process)
- `run-torchtune.log` - Fine-tuning execution details
- `run-inspect.log` - Evaluation execution details

### Total Time

**Complete workflow:** 40 minutes
- Fine-tuning: 25 minutes
- Evaluation: 15 minutes

### Viewing Results

**Interactive viewer (recommended):**
```bash
cd /scratch/gpfs/MSALGANIK/niznik/cap_4L_lora_lr_sweep_2025-10-22
inspect view --port=$(get_free_port)
```
Then open the provided URL in your browser.

**Training metrics:**
- Check SLURM logs: `rank*/slurm-*.out`
- Check WandB (if configured): https://wandb.ai/

### Next Steps

1. **Analyze results:**
   Run `analyze-experiment` skill to generate comparison tables and plots (planned)

2. **Export evaluation data:**
   ```bash
   for dir in rank*/eval/logs; do
     inspect log export $dir/*.eval --format csv >> all_results.csv
   done
   ```

3. **Share findings:**
   - Results are in experiment directory
   - experiment_summary.md tracks all job IDs and status
   - Logs provide full audit trail

**Congratulations!** Your experiment workflow is complete.
```

## Validation Before Completion

Before reporting success, verify:
- ✓ experiment_summary.md was found and read
- ✓ Scaffolding was verified before starting
- ✓ run-torchtune was invoked (check for log file)
- ✓ run-torchtune completed (all jobs reached terminal states)
- ✓ run-inspect was invoked (check for log file)
- ✓ run-inspect completed (all jobs reached terminal states)
- ✓ Model checkpoints exist
- ✓ Evaluation logs exist
- ✓ Orchestration log was created
- ✓ Both sub-skill logs exist

## Important Notes

### Orchestration Principles

- This skill **orchestrates** rather than implements - it calls other skills
- Each sub-skill maintains its own detailed log
- The orchestration log tracks high-level flow and timing
- **Sequential execution is critical** - evaluation requires fine-tuning to complete first
- Sub-skills can be run independently if needed (e.g., re-run just evaluation)
- Partial success is acceptable (e.g., fine-tuning succeeds but evaluation fails)

### Execution Order

1. **run-torchtune first** - Trains models and creates checkpoints
2. **Wait for completion** - ALL fine-tuning jobs must finish
3. **run-inspect second** - Evaluates trained models
4. **Wait for completion** - ALL evaluation jobs must finish

This order is **mandatory** - evaluation cannot proceed without trained models.

### Relationship to Other Skills

**Before this skill:**
- `design-experiment` creates experiment_summary.md
- `scaffold-experiment` generates configs (calls `scaffold-torchtune` and `scaffold-inspect`)

**After this skill:**
- `analyze-experiment` interprets results (planned)

**Can be run standalone:**
- `run-torchtune` - Just execute fine-tuning
- `run-inspect` - Just execute evaluation (requires fine-tuning complete)

### Time Management

**Typical timeline:**
- Design: 10-30 minutes (interactive with user)
- Scaffold: 2-5 minutes (automated)
- Execute (this skill): 30 minutes to several hours
  - Fine-tuning: Most of the time (model and data dependent)
  - Evaluation: Typically 5-15 minutes
- Analysis: 5-10 minutes (planned)

**Long-running experiments:**
- If experiments take hours, user doesn't need to stay connected
- SLURM jobs run independently
- Can resume monitoring later
- experiment_summary.md tracks all job IDs

### Error Recovery

If execution fails:
1. Check individual skill logs (run-torchtune.log, run-inspect.log)
2. Check SLURM logs in run directories
3. Fix the issue (e.g., out of disk space, model path incorrect)
4. Re-run this skill (will skip completed jobs)
5. Or run individual sub-skills directly (run-torchtune, run-inspect)

### Resumability

- Re-running run-experiment is safe
- Sub-skills check for already-completed jobs
- Won't re-submit jobs that finished successfully
- Useful if monitoring was interrupted

## Future Enhancements

Potential additions:
- Dry-run mode (validate without submitting jobs)
- Selective execution (only certain runs)
- Conditional evaluation (only evaluate successful fine-tuning runs)
- Email notifications on completion
- Integration with analyze-experiment (auto-run analysis after completion)

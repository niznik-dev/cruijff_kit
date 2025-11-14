# Validation - Verification

Verify fine-tuning execution completed successfully.

## Validation Checks

### 1. All Jobs Submitted

Verify:
- All selected runs had jobs submitted
- Job IDs were captured for all submissions
- No submission errors occurred

### 2. All Jobs Reached Terminal States

Check that no jobs are still:
- PENDING (queued)
- RUNNING (executing)

All should be:
- COMPLETED (success)
- FAILED (error)
- CANCELLED (user action)
- TIMEOUT (time limit)

### 3. Model Checkpoints Exist

For each COMPLETED job, verify checkpoint created:

```bash
ls {output_dir_base}/ck-out-{run_name}/epoch_{N}/
```

Expected files:
- `adapter_model.bin` or similar weight files
- `adapter_config.json`
- Other model artifacts

**If checkpoint missing:**
- Log warning
- Check SLURM output for errors
- Mark as validation failure

### 4. experiment_summary.md Updated

Verify status table contains:
- Job IDs for all runs
- Current status for all jobs
- Timestamps (Started, Completed)
- Elapsed times for completed jobs

### 5. Log File Created

Verify `run-torchtune.log` exists with:
- All job submissions logged
- Status checks recorded
- State changes documented
- Final summary present

## Success Criteria

**All of:**
- ✓ All jobs submitted successfully
- ✓ All jobs reached terminal states
- ✓ Model checkpoints exist for COMPLETED jobs
- ✓ experiment_summary.md fully updated
- ✓ Log file complete

## Failure Scenarios

**If any job FAILED:**
- Note in validation output
- Recommend checking SLURM log: `{run_dir}/slurm-{job_id}.out`
- Partial success is acceptable (some runs succeeded)

**If checkpoints missing:**
- Note in validation output
- Recommend checking disk space and output directory permissions
- Job may have completed but failed to write outputs

## Validation Output

```markdown
### Validation Results

✓ All jobs submitted (8/8)
✓ All jobs completed (7 COMPLETED, 1 FAILED)
✓ Model checkpoints verified (7/7 successful runs)
✓ experiment_summary.md updated
✓ Log file created: run-torchtune.log

**Warnings:**
- r32_lr5e-5: Job FAILED (Job ID 12345683)
  - Check: r32_lr5e-5/slurm-12345683.out
```

## Next Steps

**If validation passes:**
- Fine-tuning complete
- Ready for evaluation (if configured)

**If validation fails:**
- Review errors
- Fix issues
- Re-run failed jobs if needed

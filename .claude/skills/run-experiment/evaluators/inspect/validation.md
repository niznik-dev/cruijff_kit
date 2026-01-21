# Validation - Verification

Verify evaluation execution completed successfully.

## Validation Checks

### 1. Fine-Tuning Completion Verified

Verify dependency checking was performed:
- Fine-tuning jobs status was checked
- Model checkpoints were verified before submission

### 2. All Selected Evaluations Submitted

Verify:
- All selected evaluations had jobs submitted
- Job IDs were captured for all submissions
- No submission errors occurred

### 3. All Jobs Reached Terminal States

Check that no jobs are still:
- PENDING (queued)
- RUNNING (executing)

All should be:
- COMPLETED (success)
- FAILED (error)
- CANCELLED (user action)
- TIMEOUT (time limit)

### 4. Evaluation Logs Exist

For each COMPLETED evaluation, verify inspect-ai log created:

```bash
ls {run_dir}/eval/logs/*.eval
```

**Expected files:**
- `{task_name}_*.eval` - inspect-ai evaluation log(s)

**If log file missing:**
- Log warning
- Check SLURM output for errors
- Mark as validation failure

### 5. Log File Created

Verify `run-inspect.log` exists with:
- Dependency checks logged
- All job submissions logged
- Status checks recorded
- State changes documented
- Final summary present

## Success Criteria

**All of:**
- ✓ Fine-tuning completion verified
- ✓ Model checkpoints verified
- ✓ All evaluations submitted successfully
- ✓ All jobs reached terminal states
- ✓ Evaluation logs exist for COMPLETED jobs
- ✓ experiment_summary.yaml fully updated
- ✓ Log file complete

## Failure Scenarios

**If any evaluation FAILED:**
- Note in validation output
- Recommend checking SLURM log: `{run_dir}/eval/slurm-{job_id}.out`
- Recommend checking inspect-ai logs if they exist
- Partial success is acceptable (some evaluations succeeded)

**If evaluation logs missing:**
- Note in validation output
- Recommend checking disk space and permissions
- Job may have completed but failed to write logs

**If evaluations skipped due to missing checkpoints:**
- Note which runs were skipped
- Recommend checking fine-tuning jobs for those runs
- This is expected if fine-tuning failed for some runs

## Validation Output

```markdown
### Validation Results

✓ Fine-tuning completion verified
✓ Model checkpoints verified (7/8 runs)
✓ All evaluations submitted (7 evaluations)
✓ All jobs completed (6 COMPLETED, 1 FAILED)
✓ Evaluation logs verified (6/6 successful)
✓ Log file created: run-inspect.log

**Skipped:**
- r32_lr5e-5: Model checkpoint missing (fine-tuning may have failed)

**Warnings:**
- r16_lr5e-5/capitalization/epoch0: Job FAILED (Job ID 12345693)
  - Check: r16_lr5e-5/eval/slurm-12345693.out
  - Check: r16_lr5e-5/eval/logs/ for error details
```

## Result Viewing Recommendations

**Interactive viewer:**
```bash
inspect view --port=$(get_free_port)
```

**Export results:**
```bash
inspect log export {run_dir}/eval/logs/*.eval --format csv > results.csv
```

**Command-line summary:**
```bash
for dir in */eval/logs; do
  echo "=== $(dirname $(dirname $dir)) ==="
  inspect log ls $dir/*.eval
done
```

## Next Steps

**If validation passes:**
- Evaluation complete
- Results ready for analysis

**If validation fails:**
- Review errors
- Check SLURM and inspect-ai logs
- Re-run failed evaluations if needed

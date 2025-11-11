# Resource Estimation

Calculate compute requirements for the complete experiment (training + evaluation).

## What to Estimate

- **Training time:** Per-run and total training time
- **Evaluation time:** Total evaluation time across all runs and tasks
- **Disk space:** Checkpoint storage requirements
- **GPU hours:** Sum total GPU time needed

## Time Estimates

### From prior runs (preferred):
1. Find similar runs in `{scratch_dir}/ck-out-*/`
2. Extract iteration speed from SLURM logs: `grep -E "it/s" {log_path}`
3. Calculate: `time = (samples / batch_size / speed) * epochs`

### If no prior runs:
- Use conservative estimates based on model size and GPU type
- Clearly mark as "preliminary - verify with test run"
- Typical ranges:
  - 1B models: 30-60 min/epoch
  - 3B models: 1-2 hours/epoch
  - 7B+ models: 3-5 hours/epoch

### Evaluation time:
- Inference-only: ~2-3x faster than training
- Typically 1-5 minutes per evaluation
- Multiply by (num_runs × num_tasks × num_epochs)

## Disk Space Estimates

### From prior runs:
```bash
du -sh {prior_run_dir}/epoch_0
```

### Typical checkpoint sizes:
- 1B: ~2-3 GiB per epoch
- 3B: ~6-7 GiB per epoch
- 7B: ~14-20 GiB per epoch

**Total:** `num_runs × num_epochs × checkpoint_size + 20% buffer`

## Batch Size Guidance

### From prior runs:
1. Find GPU memory usage: `grep "GPU peak memory" {log_path}`
2. Calculate headroom: `available_memory / peak_memory`
3. Scale conservatively: `max_batch = headroom × 0.7`

### If dataset packing enabled (default):
- Reduces effective batch size by 2-4x
- Start conservative: batch_size=4 (1B), batch_size=2 (3B)

### No prior data:
- 80GB GPU: batch_size=4-8 (1B), 2-4 (3B)
- 40GB GPU: batch_size=2-4 (1B), 2 (3B)
- Start small, monitor first run, adjust

## Log All Calculations

All estimation calculations should be logged in `design-experiment.log`:

```
[2025-10-22 14:24:01] SEARCH_PRIOR_RUNS: Looking for similar experiments
Command: find {scratch_dir} -name "slurm-*.out" -path "*/ck-out-*" -size +100k | head -5
Result: Found 3 similar runs: ck-out-happy-narwhal, ck-out-bright-horizon, ck-out-calm-dolphin
Explanation: Searching for prior SLURM logs to extract training speed data for estimates

[2025-10-22 14:24:15] EXTRACT_SPEED: Analyzing prior run for training speed
Command: grep -E "[0-9.]+it/s" {scratch_dir}/ck-out-happy-narwhal/slurm-12345.out | tail -20
Result: Average speed after warmup: 4.34 it/s
Explanation: Extracting iteration speed from similar prior run with same model and batch size

[2025-10-22 14:24:30] CALCULATE_TIME: Training time estimate
Input: 8000 samples, batch_size=4, speed=4.34 it/s, epochs=2
Calculation: steps_per_epoch = 8000/4 = 2000, time_per_epoch = 2000/4.34 ≈ 461s ≈ 8min
Result: Estimated 16 minutes total (8 min × 2 epochs)
Explanation: Calculated training time based on actual iteration speed from prior run

[2025-10-22 14:25:00] CHECK_DISK: Verifying available disk space
Command: df -h {scratch_dir}
Result: 2.1T available
Explanation: Ensuring sufficient space for ~40 GiB of checkpoints
```

## Document in experiment_summary.md

```markdown
## Compute Estimates

### Training
- **Per-run time:** ~10 minutes/epoch
- **Total runs:** 4 fine-tuned runs
- **Total training time:** ~80 minutes (4 runs × 2 epochs × 10 min)

### Evaluation
- **Per-eval time:** ~2 minutes
- **Total evals:** 16 (4 runs × 2 tasks × 2 epochs)
- **Total eval time:** ~32 minutes

### Disk Space
- **Per-epoch checkpoint:** ~2.5 GiB
- **Total checkpoints:** ~40 GiB (4 runs × 2 epochs × 2.5 GiB + 20% buffer)
- **Available space:** 2.1T

### Total GPU Hours
- **Training:** ~1.3 hours
- **Evaluation:** ~0.5 hours
- **Total:** ~1.8 GPU hours
```

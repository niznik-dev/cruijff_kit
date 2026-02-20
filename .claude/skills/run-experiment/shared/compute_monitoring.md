# Shared Compute Monitoring Procedures

Common monitoring procedures used by both optimizer and evaluator modules. Each module references this file and provides its own **identifier format** (e.g., `{run_name}` for training, `{run_name}/{task_name}/{epoch}` for evaluation).

## Post-Completion Metrics

When a job transitions to COMPLETED or FAILED, capture compute metrics with `seff`. SLURM accounting has a lag — if seff returns no data, wait 30 seconds and retry once.

```bash
# First attempt
seff {job_id}
# If output shows "not available" or empty, wait and retry:
sleep 30
seff {job_id}
```

Log the metrics:

```
[YYYY-MM-DD HH:MM:SS] COMPUTE_METRICS: {identifier}
Job ID: {job_id}
GPU Time: {wall_time} / {time_limit} ({time_eff}%)
Memory: {mem_used} / {mem_requested}
CPU Efficiency: {cpu_eff}%
```

If the wall time exceeds the time limit (job state TIMEOUT), flag it:

```
[YYYY-MM-DD HH:MM:SS] COMPUTE_METRICS: {identifier}
Job ID: {job_id}
GPU Time: {wall_time} / {time_limit} (**EXCEEDED**)
Memory: {mem_used} / {mem_requested}
CPU Efficiency: {cpu_eff}%
⚠️ Time limit exceeded — consider increasing the time allocation
```

## GPU Status (Real-Time Feedback)

Every 5th poll (~5 minutes) for RUNNING jobs, read the latest GPU metrics from the shared filesystem:

```bash
tail -1 {output_dir}/gpu_metrics.csv
```

Log as:

```
[YYYY-MM-DD HH:MM:SS] RESOURCE_STATUS: {identifier}
GPU Utilization: {gpu_util}% | GPU Memory: {gpu_mem_used}/{gpu_mem_total} GB
```

If `gpu_metrics.csv` doesn't exist yet (job just started), skip silently.

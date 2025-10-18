# Monitor SLURM Jobs

You are helping the user monitor their SLURM jobs with detailed progress and resource usage information in an easy-to-read format.

## Your Task

Provide comprehensive, user-friendly monitoring of SLURM jobs including:
- Current job status (running, pending, completed, failed)
- Progress indicators (time elapsed, time remaining)
- Resource usage (CPU, memory, GPU)
- Queue position for pending jobs
- Recent output from running jobs

## Steps

### 1. Identify Jobs to Monitor

First, determine which jobs to monitor:
- Current user's jobs: `squeue -u $USER`
- Specific job IDs if provided by user
- Recent jobs from today: `sacct -u $USER -S today`

### 2. Display Job Status Table

Create a clear, formatted table showing:

**For Running Jobs:**
```
Job ID | Job Name           | Status  | Time Used/Limit | Node      | Mem Usage | GPU
-------|-------------------|---------|-----------------|-----------|-----------|----
123456 | cap7L_1B_r4       | RUNNING | 05:23/00:30:00  | della-g01 | 12.5/32GB | 1
123457 | cap7L_3B_r8       | RUNNING | 12:45/00:45:00  | della-g02 | 28.3/48GB | 1
```

**For Pending Jobs:**
```
Job ID | Job Name      | Status  | Reason                    | Priority | Queue Pos
-------|--------------|---------|---------------------------|----------|----------
123458 | cap7L_3B_r16 | PENDING | QOSMaxJobsPerUserLimit    | 4500     | 2
123459 | cap7L_3B_r32 | PENDING | Resources                 | 4500     | 5
```

**For Recently Completed Jobs:**
```
Job ID | Job Name      | Status    | Exit Code | Elapsed   | Max RSS  | Completion
-------|--------------|-----------|-----------|-----------|----------|------------
123450 | cap7L_1B_r4  | COMPLETED | 0:0       | 00:28:34  | 12.4GB   | 09:15:23
123451 | cap7L_1B_r8  | FAILED    | 1:0       | 00:05:12  | 3.2GB    | 09:10:45
```

### 3. Show Detailed Resource Usage

Use `sstat` for running jobs to show real-time resource usage:

```bash
sstat -j <JOB_ID> --format=JobID,MaxRSS,AveCPU,AveVMSize,MaxDiskRead,MaxDiskWrite
```

Display in human-readable format:
```
Resource Usage for Job 123456 (cap7L_1B_r4):
  Memory (RSS):        12.5 GB / 32 GB (39%)
  Average CPU:         85%
  Virtual Memory:      14.2 GB
  Disk Read:           2.3 GB
  Disk Write:          1.8 GB
  GPU Utilization:     95% (if available)
```

### 4. Show Recent Output/Progress

For running jobs, show the last 10-20 lines of output to give progress indication:

```bash
tail -20 /path/to/experiment/slurm-<JOB_ID>.out
```

Look for progress indicators like:
- Training step counts (e.g., "Step 150/223")
- Loss values (e.g., "loss: 0.0573")
- Percentage complete
- "Generating train split" messages
- wandb sync messages

Format this nicely for the user:
```
Progress for cap7L_1B_r4 (Job 123456):
  Training Step: 150/223 (67%)
  Current Loss:  0.0573
  Last Update:   2 minutes ago
  Status:        Training in progress ✓
```

### 5. Provide Summary Statistics

Calculate and display:
- Total jobs: Running / Pending / Completed today
- Average job duration for completed jobs
- Success rate (completed vs failed)
- Estimated time until all pending jobs complete
- Total GPU hours used today

Example:
```
Summary for Today (2025-10-18):
  Total Jobs:           12
  Running:              4
  Pending:              2
  Completed:            5 (100% success)
  Failed:               1

  Avg Job Duration:     28 minutes
  Total GPU Hours:      2.3 hours
  Est. Completion:      ~45 minutes (for all pending)

  Success Rate:         83% (5/6 completed successfully)
```

### 6. Check for Issues

Automatically detect and highlight common issues:
- Jobs with high memory usage (>90% of limit)
- Jobs near time limit (>90% of time used)
- Failed jobs (exit code != 0)
- Jobs stuck in pending for long time
- Out of memory (OOM) errors in logs

Display warnings:
```
⚠️  WARNINGS:
  - Job 123457 using 96% of memory limit (may OOM)
  - Job 123458 pending for 45 minutes (QOSMaxJobsPerUserLimit)
  - Job 123451 FAILED with exit code 1:0
    Last error: "OSError: Cannot find data file"
```

### 7. Provide Quick Actions

Suggest relevant commands based on job status:

**For Running Jobs:**
```
Watch live output:
  tail -f /path/to/slurm-123456.out

Monitor resource usage:
  watch -n 5 'sstat -j 123456 --format=JobID,MaxRSS,AveCPU'

Cancel job if needed:
  scancel 123456
```

**For Failed Jobs:**
```
View full error log:
  cat /path/to/slurm-123451.out

Check exit status:
  sacct -j 123451 --format=JobID,ExitCode,State

Resubmit job:
  cd /path/to/experiment && sbatch finetune.slurm
```

**For Pending Jobs:**
```
Check queue position:
  squeue -j 123458 --start

Cancel pending jobs:
  scancel 123458 123459
```

## Advanced Features

### GPU Monitoring (if available)

If nvidia-smi is available on compute nodes, show GPU usage:
```bash
srun --jobid=<JOB_ID> nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

Display as:
```
GPU Usage for Job 123456:
  GPU Utilization:  95%
  GPU Memory:       6.8 / 8.0 GB (85%)
  GPU Temperature:  72°C
```

### Comparative Analysis

For experiment sets, compare across jobs:
```
LoRA Rank Comparison (1B Model):
  Rank  | Job ID | Status    | Time    | Memory  | Progress
  ------|--------|-----------|---------|---------|----------
  4     | 123456 | COMPLETED | 00:28   | 12.4GB  | 100% ✓
  8     | 123457 | RUNNING   | 15:32   | 13.1GB  | 68%
  16    | 123458 | PENDING   | --      | --      | Queued
  32    | 123459 | PENDING   | --      | --      | Queued
  64    | 123460 | COMPLETED | 00:29   | 14.8GB  | 100% ✓
```

### Historical Trends

Show trends for the user's jobs over time:
```
Jobs Over Last 7 Days:
  Date       | Total | Success | Failed | Avg Duration
  -----------|-------|---------|--------|-------------
  2025-10-18 | 12    | 5       | 1      | 28 min
  2025-10-17 | 8     | 7       | 1      | 32 min
  2025-10-16 | 15    | 14      | 1      | 25 min
```

## Output Format

Always use:
- ✓ for completed successfully
- ⚙️ for running
- ⏸ for pending
- ✗ for failed
- ⚠️ for warnings

Use colors (if terminal supports):
- Green for completed
- Blue for running
- Yellow for pending
- Red for failed
- Orange for warnings

Keep tables aligned and easy to scan.

## Implementation Tips

1. **Use Multiple SLURM Commands:**
   - `squeue`: Current queue status
   - `sacct`: Historical job accounting
   - `sstat`: Real-time resource statistics
   - `scontrol show job`: Detailed job information

2. **Format Output with Column:**
   ```bash
   squeue -u $USER --format="%.10i %.30j %.8T %.10M %.10l %.8N" | column -t
   ```

3. **Parse Job State:**
   - RUNNING (R)
   - PENDING (PD)
   - COMPLETED (CD)
   - FAILED (F)
   - CANCELLED (CA)
   - TIMEOUT (TO)
   - OUT_OF_MEMORY (OOM)

4. **Calculate Progress:**
   - Parse step indicators from output logs
   - Compare elapsed time to time limit
   - Look for "epoch N/M" patterns
   - Check for final "Run summary" in wandb logs

5. **Resource Usage Calculations:**
   - Convert memory to human-readable (KB → GB)
   - Calculate percentages for memory and time
   - Show rates (e.g., tokens/second from logs)

## Example Session

```
$ monitor-jobs

Monitoring SLURM jobs for user: mjs3
Current time: 2025-10-18 09:50:15
═══════════════════════════════════════════════════════════════════════

RUNNING JOBS (4)
────────────────────────────────────────────────────────────────────
Job ID  │ Name          │ Time Used  │ Memory     │ Progress │ Node
────────────────────────────────────────────────────────────────────
1469558 │ cap7L_1B_r8   │ 11:23/30:00│ 13.2/32 GB │ 51% ⚙️   │ della-l07g4
1469581 │ cap7L_3B_r4   │ 08:15/45:00│ 28.5/48 GB │ 36% ⚙️   │ della-l07g4
1469582 │ cap7L_3B_r8   │ 08:12/45:00│ 31.2/48 GB │ 36% ⚙️   │ della-l07g4
1469583 │ cap7L_3B_r16  │ 08:10/45:00│ 35.8/48 GB │ 36% ⚙️   │ della-l07g4

PENDING JOBS (2)
────────────────────────────────────────────────────────────────────
Job ID  │ Name          │ Reason                    │ Priority │ ETA
────────────────────────────────────────────────────────────────────
1469584 │ cap7L_3B_r32  │ QOSMaxJobsPerUserLimit    │ 4500     │ ~37 min ⏸
1469585 │ cap7L_3B_r64  │ QOSMaxJobsPerUserLimit    │ 4500     │ ~37 min ⏸

COMPLETED TODAY (4)
────────────────────────────────────────────────────────────────────
Job ID  │ Name          │ Status      │ Duration │ Completed At
────────────────────────────────────────────────────────────────────
1469385 │ cap7L_1B_r4   │ COMPLETED ✓ │ 00:28:34 │ 09:04:15
1469386 │ cap7L_1B_r16  │ COMPLETED ✓ │ 00:29:12 │ 09:08:42
1469387 │ cap7L_1B_r32  │ COMPLETED ✓ │ 00:29:45 │ 09:12:18
1469388 │ cap7L_1B_r64  │ COMPLETED ✓ │ 00:30:01 │ 09:15:55

═══════════════════════════════════════════════════════════════════════
SUMMARY
  Total Active:        6 jobs (4 running, 2 pending)
  Completed Today:     4 jobs (100% success)
  Estimated Finish:    ~45 minutes
  Total GPU Hours:     2.8 hours today

Quick Commands:
  Watch all jobs:    watch -n 5 'squeue -u mjs3'
  Check specific:    tail -f /path/to/slurm-1469558.out
  Cancel all:        scancel -u mjs3
```

## Notes

- Update information every 30-60 seconds if monitoring continuously
- Cache sacct queries to avoid overwhelming the SLURM database
- Respect cluster policies about query frequency
- Provide helpful context, not just raw SLURM output
- Make it actionable - tell users what to do, not just what's happening

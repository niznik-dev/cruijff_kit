# Run Management Tools

Utilities for managing fine-tuning and evaluation runs in cruijff_kit.

## Tools

### update_run_status.py

Atomically update `runs_status.yaml` with job IDs and timestamps. Eliminates the need for multiple manual edits when submitting batch jobs.

**Single Update:**
```bash
python update_run_status.py \
    --status-file /path/to/runs_status.yaml \
    --run-name Llama-3.2-1B-Instruct_5L_rank4 \
    --job-id 1234567 \
    --status submitted
```

**Batch Update:**
```bash
# Create batch file
cat > job_updates.json <<EOF
[
    {
        "run_name": "Llama-3.2-1B-Instruct_5L_rank4",
        "job_id": 1234567,
        "status": "submitted"
    },
    {
        "run_name": "Llama-3.2-1B-Instruct_5L_rank64",
        "job_id": 1234568,
        "status": "submitted"
    }
]
EOF

python update_run_status.py \
    --status-file /path/to/runs_status.yaml \
    --batch job_updates.json
```

**Add Note:**
```bash
python update_run_status.py \
    --status-file /path/to/runs_status.yaml \
    --run-name Llama-3.2-1B-Instruct_9L_rank4 \
    --note "Resubmitted after cache collision"
```

### submit_pending_runs.sh

Batch submit all pending fine-tuning jobs from `runs_status.yaml`. Automatically updates status file with job IDs.

**Parallel Submission (default):**
```bash
./submit_pending_runs.sh /scratch/gpfs/MSALGANIK/mjs3/cap_cross_eval_5_9_13L_2025-10-20
```

**Sequential Submission (avoid cache collisions):**
```bash
./submit_pending_runs.sh --sequential /path/to/experiment
```

**Custom Delay:**
```bash
./submit_pending_runs.sh --sequential --delay 15 /path/to/experiment
```

**Submit Specific Runs Only:**
```bash
./submit_pending_runs.sh \
    --only Llama-3.2-1B-Instruct_5L_rank4 \
    --only Llama-3.2-1B-Instruct_5L_rank64 \
    /path/to/experiment
```

**Dry Run:**
```bash
./submit_pending_runs.sh --dry-run /path/to/experiment
```

## Features

### update_run_status.py

- ✓ Atomic YAML updates (no partial writes)
- ✓ Automatic timestamp generation
- ✓ Validation of status values
- ✓ Auto-generates output paths
- ✓ Supports batch updates
- ✓ Works with both finetune and evaluation stages

### submit_pending_runs.sh

- ✓ Parses YAML to find pending runs
- ✓ Pre-flight verification (checks for required files)
- ✓ Sequential or parallel submission
- ✓ Configurable delays between submissions
- ✓ Automatic status file updates via update_run_status.py
- ✓ Error handling with clear failure reporting
- ✓ Progress indicators during submission
- ✓ Dry-run mode for preview
- ✓ Filter by specific run names

## Integration with Skills

These tools are designed to be used by the launch-runs skill to simplify batch job submission:

**Before (manual edits):**
- Read runs_status.yaml
- Submit job 1 via Bash
- Edit YAML for job 1
- Submit job 2 via Bash
- Edit YAML for job 2
- ... (repeat for N jobs)

**After (automated):**
- Run `./submit_pending_runs.sh --sequential /path/to/experiment`
- Done! (all submissions + status updates handled)

## Cache Collision Prevention

When submitting jobs that share datasets, use `--sequential` mode to prevent HuggingFace cache collisions:

```bash
# Good: Sequential submission avoids cache race conditions
./submit_pending_runs.sh --sequential --delay 10 /path/to/experiment

# Risk: Parallel submission may cause cache collisions
./submit_pending_runs.sh /path/to/experiment
```

**Why this matters:**
- Multiple jobs loading the same dataset simultaneously can corrupt the HuggingFace cache
- Sequential submission with delays ensures the first job populates the cache
- Subsequent jobs reuse the cached data safely

## Error Handling

### submit_pending_runs.sh

**Pre-flight checks:**
- Verifies experiment directory exists
- Verifies runs_status.yaml exists
- Checks that all pending runs have finetune.yaml and finetune.slurm
- Exits with error if any configs are missing

**During submission:**
- Captures sbatch output to extract job IDs
- Only updates status file for successful submissions
- Continues with remaining jobs if one fails
- Reports failures at the end with clear error messages

### update_run_status.py

**Validation:**
- Verifies status file exists
- Verifies run name exists in YAML
- Validates status values against allowed list
- Atomic writes prevent partial updates

## Examples

### Complete Workflow

```bash
# 1. Plan and setup experiment (done via skills)
# 2. Generate configs (done via skills)

# 3. Submit all pending jobs sequentially (avoiding cache collisions)
cd /home/mjs3/cruijff_kit/tools/run_management
./submit_pending_runs.sh --sequential --delay 10 \
    /scratch/gpfs/MSALGANIK/mjs3/cap_cross_eval_5_9_13L_2025-10-20

# Output:
# === Found 12 pending runs ===
#   - Llama-3.2-1B-Instruct_5L_rank4
#   - Llama-3.2-1B-Instruct_5L_rank64
#   ...
#
# === Verifying configurations ===
#   ✓ Llama-3.2-1B-Instruct_5L_rank4
#   ✓ Llama-3.2-1B-Instruct_5L_rank64
#   ...
#
# === Submitting 12 jobs SEQUENTIALLY with 10s delay ===
# Proceed? (yes/no): yes
#
# [1/12] Submitting Llama-3.2-1B-Instruct_5L_rank4...
# [1/12] ✓ Llama-3.2-1B-Instruct_5L_rank4 → Job 1234567
# ✓ Updated Llama-3.2-1B-Instruct_5L_rank4 (finetune): status=submitted, job_id=1234567
#   Waiting 10s before next submission...
# [2/12] Submitting Llama-3.2-1B-Instruct_5L_rank64...
# ...
#
# === Submission Summary ===
# ✓ Successfully submitted: 12/12
# All jobs submitted successfully!
# Updated: /scratch/.../runs_status.yaml
# Monitor jobs: squeue -u mjs3
```

### Resubmit Failed Jobs

```bash
# 1. Identify failed jobs
sacct -u $USER -S today --format=JobID,JobName,State -X | grep FAILED

# 2. Update their status to pending
python update_run_status.py \
    --status-file /path/to/runs_status.yaml \
    --run-name Llama-3.2-1B-Instruct_9L_rank4 \
    --status pending

# 3. Resubmit only those runs
./submit_pending_runs.sh \
    --only Llama-3.2-1B-Instruct_9L_rank4 \
    --only Llama-3.2-1B-Instruct_9L_rank64 \
    /path/to/experiment
```

## Future Enhancements

Potential additions:
- `verify_run_configs.sh`: Standalone pre-flight checker
- `summarize_pending_runs.py`: Generate formatted dry-run previews
- `resubmit_failed_runs.sh`: Automatically detect and resubmit failed jobs
- `update_run_status_from_slurm.py`: Sync status file with actual SLURM state

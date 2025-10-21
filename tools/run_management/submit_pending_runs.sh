#!/bin/bash
#
# Submit pending fine-tuning jobs from runs_status.yaml
#
# Usage:
#   ./submit_pending_runs.sh [OPTIONS] <experiment_dir>
#
# Options:
#   --dry-run          Show what would be submitted without submitting
#   --sequential       Submit jobs with delays to avoid cache collisions
#   --delay SECONDS    Delay between sequential submissions (default: 10)
#   --only RUN_NAME    Only submit specific run (can be repeated)
#
# Example:
#   ./submit_pending_runs.sh /scratch/gpfs/MSALGANIK/mjs3/cap_cross_eval_5_9_13L_2025-10-20
#   ./submit_pending_runs.sh --sequential /path/to/experiment
#   ./submit_pending_runs.sh --only Llama-3.2-1B-Instruct_5L_rank4 /path/to/experiment

set -euo pipefail

# Default options
DRY_RUN=false
SEQUENTIAL=false
DELAY=10
ONLY_RUNS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --sequential)
            SEQUENTIAL=true
            shift
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --only)
            ONLY_RUNS+=("$2")
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            EXPERIMENT_DIR="$1"
            shift
            ;;
    esac
done

# Validate experiment directory
if [[ -z "${EXPERIMENT_DIR:-}" ]]; then
    echo "ERROR: Must provide experiment directory" >&2
    echo "Usage: $0 [OPTIONS] <experiment_dir>" >&2
    exit 1
fi

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "ERROR: Experiment directory not found: $EXPERIMENT_DIR" >&2
    exit 1
fi

STATUS_FILE="$EXPERIMENT_DIR/runs_status.yaml"
if [[ ! -f "$STATUS_FILE" ]]; then
    echo "ERROR: Status file not found: $STATUS_FILE" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATE_SCRIPT="$SCRIPT_DIR/update_run_status.py"

if [[ ! -f "$UPDATE_SCRIPT" ]]; then
    echo "ERROR: update_run_status.py not found at: $UPDATE_SCRIPT" >&2
    exit 1
fi

# Find pending runs using Python to parse YAML
PENDING_RUNS=$(python3 -c "
import yaml
import sys

with open('$STATUS_FILE', 'r') as f:
    data = yaml.safe_load(f)

pending = []
for run_name, run_data in data['runs'].items():
    if run_data.get('finetune', {}).get('status') == 'pending':
        pending.append(run_name)

print(' '.join(pending))
")

if [[ -z "$PENDING_RUNS" ]]; then
    echo "No pending runs found in $STATUS_FILE"
    exit 0
fi

# Filter by --only if specified
if [[ ${#ONLY_RUNS[@]} -gt 0 ]]; then
    FILTERED_RUNS=()
    for run in $PENDING_RUNS; do
        for only_run in "${ONLY_RUNS[@]}"; do
            if [[ "$run" == "$only_run" ]]; then
                FILTERED_RUNS+=("$run")
                break
            fi
        done
    done
    PENDING_RUNS="${FILTERED_RUNS[@]}"

    if [[ -z "$PENDING_RUNS" ]]; then
        echo "None of the specified runs are pending"
        exit 0
    fi
fi

# Convert to array
PENDING_ARRAY=($PENDING_RUNS)
TOTAL_RUNS=${#PENDING_ARRAY[@]}

echo "=== Found $TOTAL_RUNS pending runs ==="
for run in "${PENDING_ARRAY[@]}"; do
    echo "  - $run"
done
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== DRY RUN - No jobs will be submitted ==="
    for run in "${PENDING_ARRAY[@]}"; do
        run_dir="$EXPERIMENT_DIR/$run"
        if [[ -f "$run_dir/finetune.yaml" ]] && [[ -f "$run_dir/finetune.slurm" ]]; then
            echo "✓ Would submit: $run"
        else
            echo "✗ Missing configs: $run"
        fi
    done
    exit 0
fi

# Verify all runs have required files before submitting
echo "=== Verifying configurations ==="
for run in "${PENDING_ARRAY[@]}"; do
    run_dir="$EXPERIMENT_DIR/$run"
    if [[ ! -f "$run_dir/finetune.yaml" ]]; then
        echo "✗ Missing finetune.yaml: $run" >&2
        exit 1
    fi
    if [[ ! -f "$run_dir/finetune.slurm" ]]; then
        echo "✗ Missing finetune.slurm: $run" >&2
        exit 1
    fi
    echo "  ✓ $run"
done
echo ""

# Ask for confirmation
if [[ "$SEQUENTIAL" == "true" ]]; then
    echo "=== Submitting $TOTAL_RUNS jobs SEQUENTIALLY with ${DELAY}s delay ==="
else
    echo "=== Submitting $TOTAL_RUNS jobs in PARALLEL ==="
fi

read -p "Proceed? (yes/no): " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "Cancelled"
    exit 0
fi
echo ""

# Submit jobs
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_RUNS=()

count=0
for run in "${PENDING_ARRAY[@]}"; do
    ((count++))
    run_dir="$EXPERIMENT_DIR/$run"

    echo "[$count/$TOTAL_RUNS] Submitting $run..."

    # Submit job and capture output
    cd "$run_dir"
    output=$(sbatch finetune.slurm 2>&1) || true

    # Check if submission succeeded
    if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        job_id="${BASH_REMATCH[1]}"
        echo "[$count/$TOTAL_RUNS] ✓ $run → Job $job_id"

        # Update status file
        python3 "$UPDATE_SCRIPT" \
            --status-file "$STATUS_FILE" \
            --run-name "$run" \
            --job-id "$job_id" \
            --status submitted

        ((SUCCESS_COUNT++))

        # Add delay if sequential mode
        if [[ "$SEQUENTIAL" == "true" ]] && [[ $count -lt $TOTAL_RUNS ]]; then
            echo "  Waiting ${DELAY}s before next submission..."
            sleep "$DELAY"
        fi
    else
        echo "[$count/$TOTAL_RUNS] ✗ $run → FAILED: $output" >&2
        FAILED_RUNS+=("$run")
        ((FAILED_COUNT++))
    fi
done

echo ""
echo "=== Submission Summary ==="
echo "✓ Successfully submitted: $SUCCESS_COUNT/$TOTAL_RUNS"

if [[ $FAILED_COUNT -gt 0 ]]; then
    echo "✗ Failed: $FAILED_COUNT"
    echo ""
    echo "Failed runs:"
    for run in "${FAILED_RUNS[@]}"; do
        echo "  - $run"
    done
    exit 1
else
    echo ""
    echo "All jobs submitted successfully!"
    echo "Updated: $STATUS_FILE"
    echo ""
    echo "Monitor jobs: squeue -u $USER"
    exit 0
fi

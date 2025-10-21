#!/usr/bin/env python3
"""
Update runs_status.yaml with job submission information.

This utility atomically updates the status YAML file with job IDs and timestamps,
eliminating the need for multiple manual edits when submitting batch jobs.

Usage:
    # Single update
    python update_run_status.py \\
        --status-file /path/to/runs_status.yaml \\
        --run-name Llama-3.2-1B-Instruct_5L_rank4 \\
        --job-id 1234567 \\
        --status submitted

    # Batch update from JSON
    python update_run_status.py \\
        --status-file /path/to/runs_status.yaml \\
        --batch job_updates.json

    # Add note to existing run
    python update_run_status.py \\
        --status-file /path/to/runs_status.yaml \\
        --run-name Llama-3.2-1B-Instruct_5L_rank4 \\
        --note "Resubmitted after cache collision"
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml


def get_timestamp() -> str:
    """Get current timestamp in ISO 8601 format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def validate_status(status: str) -> bool:
    """Validate that status is a valid value."""
    valid_statuses = {"pending", "submitted", "running", "completed", "failed", "skipped"}
    return status in valid_statuses


def update_run_status(
    status_file: Path,
    run_name: str,
    job_id: Optional[int] = None,
    status: Optional[str] = None,
    note: Optional[str] = None,
    stage: str = "finetune",
) -> None:
    """
    Update a single run's status in the YAML file.

    Args:
        status_file: Path to runs_status.yaml
        run_name: Name of the run to update
        job_id: SLURM job ID (optional)
        status: New status value (optional)
        note: Note to add (optional)
        stage: Stage to update (default: "finetune", can also be evaluation task name)
    """
    # Read YAML
    with open(status_file, "r") as f:
        data = yaml.safe_load(f)

    # Validate run exists
    if run_name not in data["runs"]:
        print(f"ERROR: Run '{run_name}' not found in status file", file=sys.stderr)
        sys.exit(1)

    # Validate status if provided
    if status and not validate_status(status):
        print(f"ERROR: Invalid status '{status}'", file=sys.stderr)
        print(f"Valid statuses: pending, submitted, running, completed, failed, skipped", file=sys.stderr)
        sys.exit(1)

    # Get the stage section (finetune or specific evaluation)
    if stage not in data["runs"][run_name]:
        print(f"ERROR: Stage '{stage}' not found in run '{run_name}'", file=sys.stderr)
        sys.exit(1)

    run_section = data["runs"][run_name][stage]

    # Update fields
    timestamp = get_timestamp()

    if status:
        run_section["status"] = status

    if job_id is not None:
        run_section["job_id"] = job_id
        # Auto-generate output path
        experiment_dir = status_file.parent
        run_section["output"] = f"{experiment_dir}/{run_name}/slurm-{job_id}.out"

    if note:
        run_section["note"] = note

    run_section["last_updated"] = timestamp

    # Write YAML atomically (write to temp file, then rename)
    temp_file = status_file.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    temp_file.replace(status_file)

    print(f"✓ Updated {run_name} ({stage}): status={status}, job_id={job_id}")


def batch_update(status_file: Path, batch_file: Path) -> None:
    """
    Update multiple runs from a JSON batch file.

    Batch file format:
    [
        {
            "run_name": "Llama-3.2-1B-Instruct_5L_rank4",
            "job_id": 1234567,
            "status": "submitted",
            "stage": "finetune"
        },
        ...
    ]
    """
    with open(batch_file, "r") as f:
        updates = json.load(f)

    for update in updates:
        run_name = update["run_name"]
        job_id = update.get("job_id")
        status = update.get("status")
        note = update.get("note")
        stage = update.get("stage", "finetune")

        update_run_status(
            status_file=status_file,
            run_name=run_name,
            job_id=job_id,
            status=status,
            note=note,
            stage=stage,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Update runs_status.yaml with job information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--status-file",
        type=Path,
        required=True,
        help="Path to runs_status.yaml",
    )

    # Single update mode
    parser.add_argument("--run-name", help="Name of run to update")
    parser.add_argument("--job-id", type=int, help="SLURM job ID")
    parser.add_argument(
        "--status",
        help="New status (pending, submitted, running, completed, failed, skipped)",
    )
    parser.add_argument("--note", help="Note to add to run")
    parser.add_argument(
        "--stage",
        default="finetune",
        help="Stage to update (default: finetune, or evaluation task name)",
    )

    # Batch update mode
    parser.add_argument(
        "--batch",
        type=Path,
        help="JSON file with batch updates",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.status_file.exists():
        print(f"ERROR: Status file not found: {args.status_file}", file=sys.stderr)
        sys.exit(1)

    # Determine mode
    if args.batch:
        if not args.batch.exists():
            print(f"ERROR: Batch file not found: {args.batch}", file=sys.stderr)
            sys.exit(1)
        batch_update(args.status_file, args.batch)
    elif args.run_name:
        if not any([args.job_id, args.status, args.note]):
            print("ERROR: Must provide at least one of --job-id, --status, or --note", file=sys.stderr)
            sys.exit(1)
        update_run_status(
            status_file=args.status_file,
            run_name=args.run_name,
            job_id=args.job_id,
            status=args.status,
            note=args.note,
            stage=args.stage,
        )
    else:
        print("ERROR: Must provide either --run-name or --batch", file=sys.stderr)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Submit-and-monitor entrypoint for torchtune fine-tuning runs.

Drip-feeds `*/finetune.slurm` files for fine-tuned runs declared in
`experiment_summary.yaml`, emits canonical SUBMIT_JOB log blocks, and
persists resume state. See `_submit_common.py` for shared logic and
the canonical log shape.

Usage:
    python -m cruijff_kit.tools.run.submit_torchtune <experiment_dir>

Reads:
    {experiment_dir}/experiment_summary.yaml
    {experiment_dir}/<run>/finetune.slurm

Writes:
    {experiment_dir}/logs/run-torchtune.log
    {experiment_dir}/logs/run-torchtune.state.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from cruijff_kit.tools.run._submit_common import (
    TodoItem,
    resolve_max_submit,
    resolve_user,
    submit_and_monitor,
)

LOG_NAME = "run-torchtune.log"
STATE_NAME = "run-torchtune.state.json"


def _build_todo(experiment_dir: Path) -> list[TodoItem]:
    """Return a TodoItem for each fine-tuned run that has a finetune.slurm."""
    summary_path = experiment_dir / "experiment_summary.yaml"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"experiment_summary.yaml not found at {summary_path}. "
            f"Run design-experiment first."
        )

    summary = yaml.safe_load(summary_path.read_text()) or {}
    runs = summary.get("runs", []) or []

    todo: list[TodoItem] = []
    for run in runs:
        if run.get("type") != "fine-tuned":
            continue
        run_name = run.get("name")
        if not run_name:
            continue
        work_dir = experiment_dir / run_name
        slurm = work_dir / "finetune.slurm"
        if not slurm.exists():
            print(
                f"WARNING: skipping {run_name!r}: {slurm} not found "
                f"(scaffold-experiment may not have run for this run)",
                file=sys.stderr,
            )
            continue
        todo.append(
            TodoItem(work_dir=work_dir, slurm_name="finetune.slurm", name=run_name)
        )
    return todo


def run(
    experiment_dir: Path,
    user: str | None = None,
    max_submit: int | None = None,
) -> dict:
    """Programmatic entrypoint (also used by tests)."""
    experiment_dir = experiment_dir.resolve()
    todo = _build_todo(experiment_dir)
    log_path = experiment_dir / "logs" / LOG_NAME
    state_path = experiment_dir / "logs" / STATE_NAME

    return submit_and_monitor(
        todo=todo,
        log_path=log_path,
        state_path=state_path,
        action_type="SUBMIT_JOB",
        user=resolve_user(user),
        experiment_dir=experiment_dir,
        max_submit=resolve_max_submit(max_submit),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument(
        "--user",
        default=None,
        help="SLURM username for queue-depth checks. Defaults to $USER.",
    )
    parser.add_argument(
        "--max-submit",
        type=int,
        default=None,
        help="Cap on concurrent submissions (default: 25, override via MAX_SUBMIT env).",
    )
    args = parser.parse_args(argv)

    summary = run(args.experiment_dir, user=args.user, max_submit=args.max_submit)
    print(f"Done. Terminal-state breakdown: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

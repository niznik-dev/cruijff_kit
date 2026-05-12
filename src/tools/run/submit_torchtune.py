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
    monitor_only,
    resolve_max_submit,
    resolve_poll_sec,
    resolve_stagger_sec,
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
    poll_sec: float | None = None,
    stagger_sec: float | None = None,
    resume_monitor: bool = False,
) -> dict:
    """Programmatic entrypoint (also used by tests).

    `resume_monitor=True` skips the submission phase entirely and re-attaches
    a watcher to the existing state file. Useful after a clean detach
    (SIGINT / SIGTERM / sentinel) to pick up monitoring without resubmitting.

    Precedence for max_submit / poll_sec / stagger_sec: logs/monitor.json
    > CLI flag > <repo>/.config/config.json > built-in default.
    """
    experiment_dir = experiment_dir.resolve()
    log_path = experiment_dir / "logs" / LOG_NAME
    state_path = experiment_dir / "logs" / STATE_NAME

    if resume_monitor:
        return monitor_only(
            log_path=log_path,
            state_path=state_path,
            action_type="SUBMIT_JOB",
            user=resolve_user(user),
            experiment_dir=experiment_dir,
            max_submit=resolve_max_submit(max_submit),
            poll_sec=resolve_poll_sec(poll_sec),
            stagger_sec=resolve_stagger_sec(stagger_sec),
        )

    todo = _build_todo(experiment_dir)
    return submit_and_monitor(
        todo=todo,
        log_path=log_path,
        state_path=state_path,
        action_type="SUBMIT_JOB",
        user=resolve_user(user),
        experiment_dir=experiment_dir,
        max_submit=resolve_max_submit(max_submit),
        poll_sec=resolve_poll_sec(poll_sec),
        stagger_sec=resolve_stagger_sec(stagger_sec),
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
        help="Cap on concurrent submissions for this run only. Defaults are "
        "set in <repo>/.config/config.json (built-in: 25). Live mid-run "
        "override via <exp_dir>/logs/monitor.json.",
    )
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=None,
        help="Poll interval in seconds for this run only. Defaults are set "
        "in <repo>/.config/config.json (built-in: 60). Live mid-run "
        "override via <exp_dir>/logs/monitor.json.",
    )
    parser.add_argument(
        "--stagger-sec",
        type=float,
        default=None,
        help="Stagger between submissions in seconds for this run only. "
        "Defaults are set in <repo>/.config/config.json (built-in: 5). "
        "Live mid-run override via <exp_dir>/logs/monitor.json.",
    )
    parser.add_argument(
        "--resume-monitor",
        action="store_true",
        help="Skip submission; re-attach a watcher to the existing state file. "
        "Use after a clean detach to resume monitoring without resubmitting.",
    )
    args = parser.parse_args(argv)

    summary = run(
        args.experiment_dir,
        user=args.user,
        max_submit=args.max_submit,
        poll_sec=args.poll_sec,
        stagger_sec=args.stagger_sec,
        resume_monitor=args.resume_monitor,
    )
    print(f"Done. Terminal-state breakdown: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

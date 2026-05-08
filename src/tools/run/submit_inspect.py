"""Submit-and-monitor entrypoint for inspect-ai evaluation runs.

Drip-feeds every `<run>/eval/*.slurm` file under an experiment directory,
emits canonical SUBMIT_EVAL log blocks, and persists resume state. See
`_submit_common.py` for shared logic.

Drip-feed (5-second stagger, 25-job queue cap) applies on this side too —
the HF datasets cache race that hits fine-tunes also hits evals when many
jobs land on `datasets.load_dataset` at once.

Usage:
    python -m cruijff_kit.tools.run.submit_inspect <experiment_dir>

Reads:
    {experiment_dir}/<run>/eval/*.slurm

Writes:
    {experiment_dir}/logs/run-inspect.log
    {experiment_dir}/logs/run-inspect.state.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cruijff_kit.tools.run._submit_common import (
    TodoItem,
    resolve_max_submit,
    resolve_user,
    submit_and_monitor,
)

LOG_NAME = "run-inspect.log"
STATE_NAME = "run-inspect.state.json"


def _eval_name(run_dir_name: str, slurm_name: str) -> str:
    """Build a SUBMIT_EVAL identifier matching the analyze-experiment regex `[\\w./-]+`.

    Examples:
      ("Llama-3.2-1B-Instruct_base",  "capitalization.slurm")        -> "Llama-3.2-1B-Instruct_base/capitalization"
      ("Llama-3.2-1B-Instruct_rank4", "capitalization_epoch0.slurm") -> "Llama-3.2-1B-Instruct_rank4/capitalization/epoch0"
    """
    base = slurm_name
    if base.endswith(".slurm"):
        base = base[: -len(".slurm")]
    if "_epoch" in base:
        task, epoch = base.split("_epoch", 1)
        return f"{run_dir_name}/{task}/epoch{epoch}"
    return f"{run_dir_name}/{base}"


def _build_todo(experiment_dir: Path) -> list[TodoItem]:
    """Return a TodoItem for each *.slurm under any */eval/ subdirectory."""
    todo: list[TodoItem] = []
    for slurm_path in sorted(experiment_dir.glob("*/eval/*.slurm")):
        eval_dir = slurm_path.parent
        run_dir = eval_dir.parent
        todo.append(
            TodoItem(
                work_dir=eval_dir,
                slurm_name=slurm_path.name,
                name=_eval_name(run_dir.name, slurm_path.name),
            )
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
        action_type="SUBMIT_EVAL",
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

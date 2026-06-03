"""Submit-and-monitor entrypoint for inspect-ai evaluation runs.

Drip-feeds every `<run>/eval/<cell>/cell.slurm` file under an experiment
directory, emits canonical SUBMIT_EVAL log blocks, and persists resume
state. See `_submit_common.py` for shared logic.

Drip-feed (5-second stagger, 25-job queue cap) applies on this side too —
the HF datasets cache race that hits fine-tunes also hits evals when many
jobs land on `datasets.load_dataset` at once.

Usage:
    python -m cruijff_kit.tools.run.submit_inspect <experiment_dir>

Reads:
    {experiment_dir}/<run>/eval/<cell>/cell.slurm
    {experiment_dir}/<run>/eval/<cell>/eval_config.yaml

Writes:
    {experiment_dir}/logs/run-inspect.log
    {experiment_dir}/logs/run-inspect.state.json
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
from cruijff_kit.tools.torchtune.adapter_utils import (
    check_adapter_base_path,
)

LOG_NAME = "run-inspect.log"
STATE_NAME = "run-inspect.state.json"


def _eval_name(run_dir_name: str, cell_dir_name: str) -> str:
    """Build a SUBMIT_EVAL identifier matching the explore regex `[\\w./-]+`.

    The cell-dir name encodes task (+ epoch for fine-tuned runs):

    Examples:
      ("Llama-3.2-1B-Instruct_base",  "capitalization")          -> "Llama-3.2-1B-Instruct_base/capitalization"
      ("Llama-3.2-1B-Instruct_rank4", "capitalization_epoch0")   -> "Llama-3.2-1B-Instruct_rank4/capitalization/epoch0"
    """
    if "_epoch" in cell_dir_name:
        task, epoch = cell_dir_name.split("_epoch", 1)
        return f"{run_dir_name}/{task}/epoch{epoch}"
    return f"{run_dir_name}/{cell_dir_name}"


def _build_todo(experiment_dir: Path) -> list[TodoItem]:
    """Return a TodoItem for each cell.slurm under any */eval/<cell>/ subdir."""
    todo: list[TodoItem] = []
    for slurm_path in sorted(experiment_dir.glob("*/eval/*/cell.slurm")):
        cell_dir = slurm_path.parent
        run_dir = cell_dir.parent.parent
        todo.append(
            TodoItem(
                work_dir=cell_dir,
                slurm_name=slurm_path.name,
                name=_eval_name(run_dir.name, cell_dir.name),
            )
        )
    return todo


def _preflight_adapter_base_paths(experiment_dir: Path) -> None:
    """Refuse to submit if any eval points at an adapter dir whose baked-in
    base_model_name_or_path no longer resolves on disk. Catches the case
    where pretrained-llms/ has moved between fine-tune and eval.
    """
    problems: list[str] = []
    for cfg_path in sorted(experiment_dir.glob("*/eval/*/eval_config.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        model_path = cfg.get("model_path")
        if not model_path:
            continue
        problem = check_adapter_base_path(Path(model_path))
        if problem:
            problems.append(f"  - via {cfg_path}:\n      {problem}")
    if problems:
        raise SystemExit(
            "Refusing to submit eval jobs — one or more adapters have stale "
            "base-model paths:\n" + "\n".join(problems)
        )


def run(
    experiment_dir: Path,
    user: str | None = None,
    max_submit: int | None = None,
    poll_sec: float | None = None,
    stagger_sec: float | None = None,
    resume_monitor: bool = False,
    no_retry: bool = False,
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
            action_type="SUBMIT_EVAL",
            user=resolve_user(user),
            experiment_dir=experiment_dir,
            max_submit=resolve_max_submit(max_submit),
            poll_sec=resolve_poll_sec(poll_sec),
            stagger_sec=resolve_stagger_sec(stagger_sec),
            no_retry=no_retry,
        )

    _preflight_adapter_base_paths(experiment_dir)
    todo = _build_todo(experiment_dir)
    return submit_and_monitor(
        todo=todo,
        log_path=log_path,
        state_path=state_path,
        action_type="SUBMIT_EVAL",
        user=resolve_user(user),
        experiment_dir=experiment_dir,
        max_submit=resolve_max_submit(max_submit),
        poll_sec=resolve_poll_sec(poll_sec),
        stagger_sec=resolve_stagger_sec(stagger_sec),
        no_retry=no_retry,
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
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable automatic retries on transient failures. Currently "
        "the only retry strategy applies to fine-tuning OOMs (handled by "
        "submit_torchtune); future eval-side retry strategies will share "
        "this gate.",
    )
    args = parser.parse_args(argv)

    summary = run(
        args.experiment_dir,
        user=args.user,
        max_submit=args.max_submit,
        poll_sec=args.poll_sec,
        stagger_sec=args.stagger_sec,
        resume_monitor=args.resume_monitor,
        no_retry=args.no_retry,
    )
    print(f"Done. Terminal-state breakdown: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

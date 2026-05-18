"""One-shot consolidated status snapshot across both run-experiment submitters.

Reads `logs/run-torchtune.state.json` and `logs/run-inspect.state.json` if
present, refreshes in-flight entries from `squeue`/`sacct`, prints a
consolidated table, and exits. Side effects are bounded to:

- writing back refreshed state files (idempotent)
- emitting `STATE_CHANGE` log blocks when transitions are observed
- emitting `ALL_COMPLETE` once per tool when refresh observes all-terminal
  (idempotent via the guard inside `log_all_complete`)

Does NOT submit anything and does not write `MONITOR_DETACHED` or
`SUBMIT_*` blocks. Safe to call from `/loop`, cron, or interactively at
any cadence — and the refresh keeps the run logs alive even when no
monitor is currently attached.

Note: this snapshot is one-shot and doesn't poll, so it doesn't read
`logs/monitor.json`. That file affects the live watchers
(`submit_torchtune` / `submit_inspect`); see the run-experiment skill
for details.

Usage:
    python -m cruijff_kit.tools.run.status <experiment_dir>
    python -m cruijff_kit.tools.run.status <experiment_dir> --json

Reads:
    {experiment_dir}/logs/run-torchtune.state.json
    {experiment_dir}/logs/run-inspect.state.json

Writes (refreshed, idempotent):
    same paths, plus STATE_CHANGE entries appended to the matching .log
    when SLURM has moved a job since the last refresh, plus a single
    ALL_COMPLETE block when refresh first observes all-terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from cruijff_kit.tools.run._submit_common import (
    TERMINAL_STATES,
    _refresh_in_flight_state,
    log_all_complete,
    read_state,
    write_state_atomic,
)

# (tool name, log filename, state filename) for every submitter we know about.
# When a new submitter lands (DSPy etc.), add it here.
TOOLS = [
    ("torchtune", "run-torchtune.log", "run-torchtune.state.json"),
    ("inspect", "run-inspect.log", "run-inspect.state.json"),
]


def snapshot(experiment_dir: Path) -> dict[str, dict]:
    """Refresh state for every known tool. Returns {tool_name: state_dict}.

    Skips tools whose state file is missing (i.e. that submitter was never
    invoked for this experiment). Empty state dicts are kept so callers can
    distinguish "tool ran but submitted nothing" from "tool never ran."
    """
    out: dict[str, dict] = {}
    for tool_name, log_name, state_name in TOOLS:
        state_path = experiment_dir / "logs" / state_name
        log_path = experiment_dir / "logs" / log_name
        if not state_path.exists():
            continue
        state = read_state(state_path)
        state = _refresh_in_flight_state(state, log_path)
        write_state_atomic(state_path, state)
        # If status is the one that happens to observe all-terminal,
        # emit ALL_COMPLETE so the pipeline log closes cleanly without
        # needing a separate --resume-monitor invocation just for log hygiene.
        # log_all_complete is idempotent — at most one block per log.
        if state and all(e.get("state") in TERMINAL_STATES for e in state.values()):
            summary = Counter(e.get("state", "UNKNOWN") for e in state.values())
            log_all_complete(log_path, dict(summary))
        out[tool_name] = state
    return out


def _format_age(submitted_at_str: str) -> str:
    try:
        submitted = datetime.strptime(submitted_at_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return "?"
    total = int((datetime.now() - submitted).total_seconds())
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    return f"{total // 3600}h{(total % 3600) // 60}m"


def format_table(snapshots: dict[str, dict]) -> str:
    if not snapshots:
        return "No state files found under logs/. Has run-experiment been invoked yet?"

    # Width the key column to the widest entry across both tools so eval
    # paths like `<run>/eval/<task>_epoch<N>.slurm` don't get truncated.
    keys = [k for state in snapshots.values() for k in state]
    maxw = max((len(k) for k in keys), default=20)

    lines: list[str] = []
    grand: Counter[str] = Counter()
    for tool_name, state in snapshots.items():
        lines.append(f"\n[{tool_name}]")
        if not state:
            lines.append("  (no submissions yet)")
            continue
        counter: Counter[str] = Counter()
        for key, entry in sorted(state.items()):
            jid = entry.get("job_id") or "-"
            s = entry.get("state", "?")
            submitted = entry.get("submitted_at", "")
            age = _format_age(submitted) if submitted else "?"
            counter[s] += 1
            lines.append(f"  {key:<{maxw}}  jid {str(jid):>10}  {s:14}  ({age} ago)")
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))
        lines.append(f"  -- {sum(counter.values())} jobs: {breakdown}")
        grand.update(counter)

    if grand:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(grand.items()))
        lines.append(
            f"\nTotal: {sum(grand.values())} jobs across "
            f"{len(snapshots)} tool(s): {breakdown}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the table.",
    )
    args = parser.parse_args(argv)

    experiment_dir = args.experiment_dir.resolve()
    snapshots = snapshot(experiment_dir)

    if args.json:
        print(json.dumps(snapshots, indent=2, sort_keys=True))
    else:
        print(format_table(snapshots))
    return 0


if __name__ == "__main__":
    sys.exit(main())

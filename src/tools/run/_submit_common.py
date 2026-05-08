"""Shared helpers for the run-experiment submitters.

`submit_torchtune` and `submit_inspect` both:
- discover slurm scripts under an experiment directory,
- drip-feed sbatch them under the SLURM gpu-test QoS cap (default 25),
- emit the canonical 4-line log block that analyze-experiment's regex consumes,
- persist resume state as JSON keyed by relative path,
- monitor the submitted jobs to terminal state.

The shared logic lives here. The two submitters are thin wrappers that
build the right todo list and hand it to `submit_and_monitor()`.

Canonical log block shape (from .claude/skills/run-experiment/logging.md):

    [YYYY-MM-DD HH:MM:SS] SUBMIT_JOB: <name>
    Details: sbatch <slurm_name>
    Job ID: <jid>
    Result: success

The regex in .claude/skills/analyze-experiment/generation.md is
`SUBMIT_JOB: ([\\w.-]+)\\n.*?\\nJob ID: (\\d+)` (and SUBMIT_EVAL for evals).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

DEFAULT_MAX_SUBMIT = 25  # Della gpu-test QoS cap on MaxSubmitJobsPerUser
DEFAULT_STAGGER_SEC = 5  # avoids HF datasets cache race when jobs land at once
DEFAULT_POLL_SEC = 60

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# State-file helpers
# ---------------------------------------------------------------------------


def state_key(work_dir: Path, slurm_name: str, experiment_dir: Path) -> str:
    """Stable per-job key for the resume state file.

    Using `work_dir.name` collapses all eval entries onto a single key
    because every run's eval workdir is `<run_dir>/eval` (issue #451 comment).
    Including the relative path keeps each entry unique.
    """
    rel = work_dir.relative_to(experiment_dir)
    return f"{rel}/{slurm_name}"


def read_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except json.JSONDecodeError:
        # Corrupt mid-write — treat as empty so resume re-checks SLURM truth.
        return {}


def write_state_atomic(state_path: Path, state: dict) -> None:
    """Write JSON atomically — write to .tmp, then rename.

    Prevents a half-written file from breaking resume after an interruption.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, state_path)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def append_log(log_path: Path, block: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(block.rstrip("\n") + "\n\n")


def log_submit(
    log_path: Path,
    action_type: str,
    name: str,
    slurm_name: str,
    job_id: str | int,
) -> None:
    """Emit the canonical 4-line submission block.

    `action_type` is "SUBMIT_JOB" (torchtune) or "SUBMIT_EVAL" (inspect).
    `name` is whatever identifies the submission to a future log reader —
    a run name for finetunes, a `run/task/epoch` triple for evals.
    """
    block = (
        f"[{_now()}] {action_type}: {name}\n"
        f"Details: sbatch {slurm_name}\n"
        f"Job ID: {job_id}\n"
        f"Result: success"
    )
    append_log(log_path, block)


def log_submit_failure(
    log_path: Path,
    action_type: str,
    name: str,
    slurm_name: str,
    error: str,
) -> None:
    block = (
        f"[{_now()}] {action_type}: {name}\n"
        f"Details: sbatch {slurm_name}\n"
        f"Result: failure: {error}"
    )
    append_log(log_path, block)


def log_state_change(
    log_path: Path,
    name: str,
    job_id: str | int,
    old_state: str,
    new_state: str,
) -> None:
    block = (
        f"[{_now()}] STATE_CHANGE: {name}\n"
        f"Details: job {job_id}: {old_state} -> {new_state}\n"
        f"Result: tracked"
    )
    append_log(log_path, block)


def log_all_complete(log_path: Path, summary: dict) -> None:
    parts = ", ".join(f"{k}={v}" for k, v in sorted(summary.items()))
    block = (
        f"[{_now()}] ALL_COMPLETE\n"
        f"Details: terminal-state breakdown: {parts}\n"
        f"Result: done"
    )
    append_log(log_path, block)


# ---------------------------------------------------------------------------
# SLURM interaction (mockable via subprocess.run)
# ---------------------------------------------------------------------------


def queue_depth(user: str) -> int:
    """Number of jobs the user has in the SLURM queue."""
    r = subprocess.run(
        ["squeue", "-u", user, "-h"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sum(1 for line in r.stdout.splitlines() if line.strip())


def sbatch_submit(work_dir: Path, slurm_name: str) -> str:
    """Submit one slurm script, return the job ID as a string.

    Raises CalledProcessError on submission failure so the caller can log it.
    """
    r = subprocess.run(
        ["sbatch", "--parsable", slurm_name],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.strip().split(";")[0]


def squeue_state(jid: str) -> str | None:
    """Return current state for an in-flight job, or None if not in squeue."""
    r = subprocess.run(
        ["squeue", "-j", jid, "-h", "-o", "%T"],
        capture_output=True,
        text=True,
    )
    line = r.stdout.strip()
    return line or None


def sacct_state(jid: str) -> str | None:
    """Fall back to sacct for terminal state when squeue no longer lists the job."""
    r = subprocess.run(
        ["sacct", "-j", jid, "-X", "-n", "--format=State", "--parsable2"],
        capture_output=True,
        text=True,
    )
    line = r.stdout.strip().splitlines()
    return line[0].strip() if line and line[0].strip() else None


def current_state(jid: str) -> str | None:
    """squeue first, sacct fallback. Returns None if neither knows the job."""
    return squeue_state(jid) or sacct_state(jid)


# ---------------------------------------------------------------------------
# Drip-feed submit-and-monitor
# ---------------------------------------------------------------------------


@dataclass
class TodoItem:
    work_dir: Path
    slurm_name: str
    name: str  # SUBMIT_JOB / SUBMIT_EVAL identifier (e.g. run name)


@dataclass
class _SubmitConfig:
    log_path: Path
    state_path: Path
    action_type: str
    user: str
    experiment_dir: Path
    max_submit: int = DEFAULT_MAX_SUBMIT
    stagger_sec: float = DEFAULT_STAGGER_SEC
    poll_sec: float = DEFAULT_POLL_SEC


def submit_and_monitor(
    todo: Iterable[TodoItem],
    log_path: Path,
    state_path: Path,
    action_type: str,
    user: str,
    experiment_dir: Path,
    max_submit: int = DEFAULT_MAX_SUBMIT,
    stagger_sec: float = DEFAULT_STAGGER_SEC,
    poll_sec: float = DEFAULT_POLL_SEC,
) -> dict:
    """Drip-feed-submit `todo` and poll until all jobs reach terminal state.

    Returns a dict summarizing terminal-state counts (e.g. {"COMPLETED": 2}).
    Resumable: existing entries in the state file with terminal state are skipped.

    Honors:
    - `max_submit` queue-depth cap (gpu-test default = 25; override via env or arg).
    - `stagger_sec` between submissions (HF datasets cache race).
    - `poll_sec` when waiting for queue room or for in-flight jobs.
    """
    cfg = _SubmitConfig(
        log_path=log_path,
        state_path=state_path,
        action_type=action_type,
        user=user,
        experiment_dir=experiment_dir,
        max_submit=max_submit,
        stagger_sec=stagger_sec,
        poll_sec=poll_sec,
    )

    state = read_state(state_path)
    state = _refresh_in_flight_state(state, log_path)
    write_state_atomic(state_path, state)

    pending = _filter_pending(todo, state, experiment_dir)
    state = _drip_feed_submit(pending, state, cfg)
    state = _monitor_to_terminal(state, cfg)

    summary = _summarize(state)
    log_all_complete(log_path, summary)
    return summary


def _filter_pending(
    todo: Iterable[TodoItem],
    state: dict,
    experiment_dir: Path,
) -> list[TodoItem]:
    """Drop items already submitted (by state-key match)."""
    submitted = set(state.keys())
    pending: list[TodoItem] = []
    for item in todo:
        key = state_key(item.work_dir, item.slurm_name, experiment_dir)
        if key not in submitted:
            pending.append(item)
    return pending


def _refresh_in_flight_state(state: dict, log_path: Path) -> dict:
    """For non-terminal entries, re-check SLURM and update."""
    for key, entry in state.items():
        if entry.get("state") in TERMINAL_STATES:
            continue
        jid = entry.get("job_id")
        if not jid:
            continue
        new_state = current_state(str(jid))
        if new_state and new_state != entry.get("state"):
            log_state_change(
                log_path,
                name=key,
                job_id=jid,
                old_state=entry.get("state", "?"),
                new_state=new_state,
            )
            entry["state"] = new_state
    return state


def _drip_feed_submit(
    pending: list[TodoItem],
    state: dict,
    cfg: _SubmitConfig,
) -> dict:
    todo = list(pending)
    while todo:
        depth = queue_depth(cfg.user)
        room = cfg.max_submit - depth
        if room <= 0:
            time.sleep(cfg.poll_sec)
            continue

        batch = min(room, len(todo))
        for i in range(batch):
            item = todo.pop(0)
            key = state_key(item.work_dir, item.slurm_name, cfg.experiment_dir)
            try:
                jid = sbatch_submit(item.work_dir, item.slurm_name)
            except subprocess.CalledProcessError as e:
                err = (e.stderr or "").strip() or f"exit {e.returncode}"
                log_submit_failure(
                    cfg.log_path, cfg.action_type, item.name, item.slurm_name, err
                )
                state[key] = {
                    "job_id": None,
                    "submitted_at": _now(),
                    "state": "SUBMIT_FAILED",
                    "error": err,
                }
                write_state_atomic(cfg.state_path, state)
                continue

            log_submit(cfg.log_path, cfg.action_type, item.name, item.slurm_name, jid)
            state[key] = {
                "job_id": jid,
                "submitted_at": _now(),
                "state": "PENDING",
            }
            write_state_atomic(cfg.state_path, state)

            if todo or i < batch - 1:
                time.sleep(cfg.stagger_sec)
    return state


def _monitor_to_terminal(state: dict, cfg: _SubmitConfig) -> dict:
    while not _all_terminal(state):
        time.sleep(cfg.poll_sec)
        state = _refresh_in_flight_state(state, cfg.log_path)
        write_state_atomic(cfg.state_path, state)
    return state


def _all_terminal(state: dict) -> bool:
    return all(
        entry.get("state") in TERMINAL_STATES or entry.get("state") == "SUBMIT_FAILED"
        for entry in state.values()
    )


def _summarize(state: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for entry in state.values():
        s = entry.get("state", "UNKNOWN")
        out[s] = out.get(s, 0) + 1
    return out


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def resolve_user(cli_user: str | None) -> str:
    return cli_user or os.environ.get("USER") or ""


def resolve_max_submit(cli_max: int | None) -> int:
    if cli_max is not None:
        return cli_max
    env = os.environ.get("MAX_SUBMIT")
    if env:
        try:
            return int(env)
        except ValueError:
            print(
                f"WARNING: ignoring non-integer MAX_SUBMIT env var: {env!r}",
                file=sys.stderr,
            )
    return DEFAULT_MAX_SUBMIT

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
import signal
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

MONITOR_CONFIG_NAME = "monitor.json"
_MONITOR_KNOBS = ("poll_sec", "stagger_sec", "max_submit")

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
# Detach (signal + sentinel)
# ---------------------------------------------------------------------------
#
# Watching is detachable. Three trigger paths converge on the same exit:
# flush state -> log MONITOR_DETACHED -> exit. Jobs keep running either way.
#
# - SIGINT  (Ctrl+C, human at a terminal)
# - SIGTERM (parent process killing the subprocess)
# - touch <exp_dir>/logs/.detach  (anyone, including agents that no longer
#   hold the process handle; the sentinel persists across invocations so
#   re-attach via --resume-monitor will also exit until removed)

SENTINEL_NAME = ".detach"

_detach_signal_received: str | None = None  # set by handler; "SIGINT" / "SIGTERM"


def _on_detach_signal(signum, frame):  # noqa: ARG001 — signal handler signature
    global _detach_signal_received
    _detach_signal_received = "SIGINT" if signum == signal.SIGINT else "SIGTERM"


def install_detach_handlers() -> None:
    """Install SIGINT/SIGTERM handlers that flip the module-level flag."""
    signal.signal(signal.SIGINT, _on_detach_signal)
    signal.signal(signal.SIGTERM, _on_detach_signal)


def _reset_detach_state() -> None:
    """Reset the in-memory signal flag (does NOT touch the sentinel file)."""
    global _detach_signal_received
    _detach_signal_received = None


def sentinel_path(log_path: Path) -> Path:
    """The detach sentinel lives next to the per-tool log files."""
    return log_path.parent / SENTINEL_NAME


def detach_requested(log_path: Path) -> tuple[bool, str | None]:
    """Return (True, reason) if anything has asked us to stop watching.

    reason is one of: "SIGINT", "SIGTERM", "sentinel", or None.
    """
    if _detach_signal_received:
        return True, _detach_signal_received
    if sentinel_path(log_path).exists():
        return True, "sentinel"
    return False, None


def _interruptible_sleep(
    seconds: float, log_path: Path, check_interval: float = 1.0
) -> bool:
    """Sleep up to `seconds`, returning True if detach was requested mid-sleep.

    Polls `detach_requested` every `check_interval` seconds. The 1s default
    bounds the worst-case "how long after touch .detach does the monitor
    notice" at 1 second, which is plenty responsive without spinning.
    """
    elapsed = 0.0
    while elapsed < seconds:
        chunk = min(check_interval, seconds - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        detached, _ = detach_requested(log_path)
        if detached:
            return True
    return False


def _reattach_hint(experiment_dir: Path, action_type: str, reason: str) -> str:
    """Loud banner shown after detach so the agent re-reading later sees how to resume."""
    submitter = "submit_torchtune" if action_type == "SUBMIT_JOB" else "submit_inspect"
    sentinel_note = ""
    if reason == "sentinel":
        sentinel_note = (
            f"\n   NOTE: sentinel is sticky — remove it before re-attaching:\n"
            f"         rm {experiment_dir}/logs/{SENTINEL_NAME}"
        )
    return (
        f"\n⏸  Monitor detached (reason={reason}). Jobs continue running.{sentinel_note}\n"
        f"   Check status:  python -m cruijff_kit.tools.run.status {experiment_dir}\n"
        f"   Re-attach:     python -m cruijff_kit.tools.run.{submitter} {experiment_dir} --resume-monitor\n"
    )


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
    """Write the ALL_COMPLETE closing block. Idempotent — skips if one is
    already present in this log.

    This lets `status` emit ALL_COMPLETE when it observes all-terminal without
    risking duplicates from a later `--resume-monitor` invocation, and also
    closes a latent issue where running `--resume-monitor` twice on a finished
    experiment would have written two ALL_COMPLETE blocks.
    """
    if log_path.exists() and "ALL_COMPLETE" in log_path.read_text():
        return
    parts = ", ".join(f"{k}={v}" for k, v in sorted(summary.items()))
    block = (
        f"[{_now()}] ALL_COMPLETE\n"
        f"Details: terminal-state breakdown: {parts}\n"
        f"Result: done"
    )
    append_log(log_path, block)


def log_monitor_detached(log_path: Path, reason: str, summary: dict) -> None:
    """Canonical 'monitor stopped watching' block.

    Reason is one of "SIGINT", "SIGTERM", "sentinel". Jobs are unaffected;
    this only records that the watcher exited. analyze-experiment's
    SUBMIT_JOB/SUBMIT_EVAL/Job ID regex won't match this block, so the
    harvest path is undisturbed.
    """
    parts = ", ".join(f"{k}={v}" for k, v in sorted(summary.items())) or "(no state)"
    block = (
        f"[{_now()}] MONITOR_DETACHED\n"
        f"Details: reason={reason}; state breakdown: {parts}\n"
        f"Result: monitoring paused; jobs continue"
    )
    append_log(log_path, block)


def log_monitor_config(
    log_path: Path,
    current: dict,
    changes: dict,
) -> None:
    """Canonical 'live monitor settings applied' block.

    `current` is the post-update view of all three knobs (poll_sec,
    stagger_sec, max_submit). `changes` maps knob -> prior value for the
    ones that actually changed in this refresh. Unchanged knobs are still
    shown so an operator reading the log sees the full active picture.

    Block shape (no `Job ID:` line — safe from the analyze-experiment
    SUBMIT_JOB/SUBMIT_EVAL harvest regex):

        [YYYY-MM-DD HH:MM:SS] MONITOR_CONFIG: applied
        Details: poll_sec=30 (was 60); stagger_sec=5 (unchanged); max_submit=25 (unchanged)
        Result: settings updated from logs/monitor.json
    """
    parts = []
    for knob in _MONITOR_KNOBS:
        val = current[knob]
        if knob in changes:
            parts.append(f"{knob}={val} (was {changes[knob]})")
        else:
            parts.append(f"{knob}={val} (unchanged)")
    block = (
        f"[{_now()}] MONITOR_CONFIG: applied\n"
        f"Details: {'; '.join(parts)}\n"
        f"Result: settings updated from logs/{MONITOR_CONFIG_NAME}"
    )
    append_log(log_path, block)


# ---------------------------------------------------------------------------
# Live monitor settings (logs/monitor.json)
# ---------------------------------------------------------------------------
#
# The watcher re-reads this file on every poll iteration so `poll_sec`,
# `stagger_sec`, and `max_submit` can be tuned mid-run without detach +
# re-attach. Same filesystem-as-control-plane shape as the `.detach`
# sentinel — anyone with FS access (human, agent, /loop, cron) can edit.
#
# Precedence: monitor.json > CLI flag > env var > built-in default.


def monitor_config_path(log_path: Path) -> Path:
    """The live-settings file lives next to the per-tool log files."""
    return log_path.parent / MONITOR_CONFIG_NAME


def _validate_monitor_value(
    key: str, raw: object
) -> tuple[bool, float | int | None, str | None]:
    """Coerce + validate a single monitor.json entry.

    Returns `(ok, coerced_value, error_msg)`. On failure, `coerced_value`
    is None and `error_msg` describes the problem suitable for a warning.
    """
    if key in ("poll_sec", "stagger_sec"):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return False, None, f"{key} must be a number, got {type(raw).__name__}"
        # Preserve int vs float so the log block prints cleanly (10 not 10.0).
        if key == "poll_sec" and raw <= 0:
            return False, None, f"poll_sec must be > 0, got {raw}"
        if key == "stagger_sec" and raw < 0:
            return False, None, f"stagger_sec must be >= 0, got {raw}"
        return True, raw, None
    if key == "max_submit":
        if isinstance(raw, bool) or not isinstance(raw, int):
            return (
                False,
                None,
                f"max_submit must be an integer, got {type(raw).__name__}",
            )
        if raw < 1:
            return False, None, f"max_submit must be >= 1, got {raw}"
        return True, raw, None
    return False, None, f"unknown key {key!r}"


def _load_monitor_config(log_path: Path) -> tuple[dict, list[str]]:
    """Read + validate logs/monitor.json. Never raises.

    Returns `(overrides, warnings)`. `overrides` contains only valid,
    recognized keys with coerced values; `warnings` lists problems
    suitable for stderr. Missing file -> ({}, []).
    """
    path = monitor_config_path(log_path)
    if not path.exists():
        return {}, []
    try:
        raw = path.read_text()
    except OSError as e:
        return {}, [f"could not read {path}: {e}"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, [f"{path} is not valid JSON ({e.msg}); keeping previous values"]
    if not isinstance(data, dict):
        return {}, [f"{path} must be a JSON object; keeping previous values"]

    overrides: dict = {}
    warnings: list[str] = []
    for key, raw_val in data.items():
        if key not in _MONITOR_KNOBS:
            warnings.append(f"{path}: ignoring unknown key {key!r}")
            continue
        ok, val, err = _validate_monitor_value(key, raw_val)
        if not ok:
            warnings.append(f"{path}: {err}; keeping previous value")
            continue
        overrides[key] = val
    return overrides, warnings


def _refresh_monitor_config(cfg: _SubmitConfig) -> None:
    """Re-read monitor.json and apply overrides to `cfg` in place.

    - Silent on missing file.
    - Warns to stderr (not the run log) on malformed file / bad values
      and keeps previously-applied values.
    - Emits a MONITOR_CONFIG block to the run log only when at least one
      knob actually changes value.
    """
    overrides, warnings = _load_monitor_config(cfg.log_path)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    changes: dict = {}
    for knob in _MONITOR_KNOBS:
        if knob not in overrides:
            continue
        new_val = overrides[knob]
        old_val = getattr(cfg, knob)
        if new_val != old_val:
            changes[knob] = old_val
            setattr(cfg, knob, new_val)

    if changes:
        current = {knob: getattr(cfg, knob) for knob in _MONITOR_KNOBS}
        log_monitor_config(cfg.log_path, current, changes)


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
    """Default flow: submit drip-fed, then monitor to terminal.

    Both phases honor detach (SIGINT/SIGTERM/sentinel). On detach, flushes
    state, emits MONITOR_DETACHED, prints a re-attach hint to stderr, and
    returns the partial summary. Jobs continue running.

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
    _reset_detach_state()
    install_detach_handlers()

    state = read_state(state_path)
    state = _refresh_in_flight_state(state, log_path)
    write_state_atomic(state_path, state)

    pending = _filter_pending(todo, state, experiment_dir)
    state = _drip_feed_submit(pending, state, cfg)
    if _finalize_if_detached(state, cfg) is not None:
        return _summarize(state)

    state = _monitor_to_terminal(state, cfg)
    if _finalize_if_detached(state, cfg) is not None:
        return _summarize(state)

    summary = _summarize(state)
    log_all_complete(log_path, summary)
    return summary


def submit_only(
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
    """Submit phase only — drip-feed-submits then returns without monitoring.

    Honors detach: if requested mid-drip-feed, flushes state, emits
    MONITOR_DETACHED, and returns early. Currently not wired to a CLI flag
    (the default `submit_and_monitor` covers the dispatch+watch case); kept
    available for tests and future composition.
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
    _reset_detach_state()
    install_detach_handlers()

    state = read_state(state_path)
    state = _refresh_in_flight_state(state, log_path)
    write_state_atomic(state_path, state)

    pending = _filter_pending(todo, state, experiment_dir)
    state = _drip_feed_submit(pending, state, cfg)
    _finalize_if_detached(state, cfg)
    return _summarize(state)


def monitor_only(
    log_path: Path,
    state_path: Path,
    action_type: str,
    user: str,
    experiment_dir: Path,
    max_submit: int = DEFAULT_MAX_SUBMIT,
    stagger_sec: float = DEFAULT_STAGGER_SEC,
    poll_sec: float = DEFAULT_POLL_SEC,
) -> dict:
    """Re-attach to an existing state file and watch to terminal — no new submissions.

    Powers `--resume-monitor`. If the state file is missing or empty, warns
    loudly to stderr and returns an empty summary rather than silently no-op.

    Honors detach: emits MONITOR_DETACHED on signal/sentinel and returns early.
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
    _reset_detach_state()
    install_detach_handlers()

    state = read_state(state_path)
    if not state:
        print(
            f"WARNING: no state at {state_path}; nothing to monitor. "
            f"Run the submitter without --resume-monitor first to dispatch jobs.",
            file=sys.stderr,
        )
        return {}

    state = _refresh_in_flight_state(state, log_path)
    write_state_atomic(state_path, state)

    state = _monitor_to_terminal(state, cfg)
    if _finalize_if_detached(state, cfg) is not None:
        return _summarize(state)

    summary = _summarize(state)
    log_all_complete(log_path, summary)
    return summary


def _finalize_if_detached(state: dict, cfg: _SubmitConfig) -> str | None:
    """If a detach was requested, log MONITOR_DETACHED + print hint; else no-op.

    Returns the detach reason (truthy) so the caller can decide whether to
    short-circuit, or None if no detach.
    """
    detached, reason = detach_requested(cfg.log_path)
    if not detached:
        return None
    summary = _summarize(state)
    log_monitor_detached(cfg.log_path, reason, summary)
    sys.stderr.write(_reattach_hint(cfg.experiment_dir, cfg.action_type, reason))
    return reason


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
        # Pick up any live monitor.json edits before this iteration's
        # back-pressure check / batch sizing.
        _refresh_monitor_config(cfg)

        # Detach check before any further submission work.
        detached, _ = detach_requested(cfg.log_path)
        if detached:
            return state

        depth = queue_depth(cfg.user)
        room = cfg.max_submit - depth
        if room <= 0:
            if _interruptible_sleep(cfg.poll_sec, cfg.log_path):
                return state
            continue

        batch = min(room, len(todo))
        for i in range(batch):
            # Detach check between submissions inside a batch.
            detached, _ = detach_requested(cfg.log_path)
            if detached:
                return state

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
                # Refresh inside the batch too so a long batch picks up
                # stagger / max_submit edits without waiting for the next
                # outer iteration.
                _refresh_monitor_config(cfg)
                if _interruptible_sleep(cfg.stagger_sec, cfg.log_path):
                    return state
    return state


def _monitor_to_terminal(state: dict, cfg: _SubmitConfig) -> dict:
    while not _all_terminal(state):
        _refresh_monitor_config(cfg)
        if _interruptible_sleep(cfg.poll_sec, cfg.log_path):
            return state
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


def resolve_poll_sec(cli_poll: float | None) -> float:
    if cli_poll is not None:
        return cli_poll
    env = os.environ.get("POLL_SEC")
    if env:
        try:
            return float(env)
        except ValueError:
            print(
                f"WARNING: ignoring non-numeric POLL_SEC env var: {env!r}",
                file=sys.stderr,
            )
    return DEFAULT_POLL_SEC


def resolve_stagger_sec(cli_stagger: float | None) -> float:
    if cli_stagger is not None:
        return cli_stagger
    env = os.environ.get("STAGGER_SEC")
    if env:
        try:
            return float(env)
        except ValueError:
            print(
                f"WARNING: ignoring non-numeric STAGGER_SEC env var: {env!r}",
                file=sys.stderr,
            )
    return DEFAULT_STAGGER_SEC

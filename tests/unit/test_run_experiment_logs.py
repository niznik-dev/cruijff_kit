"""Tests for the run-experiment submitters (#451).

Three responsibilities are exercised:

1. Canonical log shape — every submission writes a 4-line block whose
   regex round-trips through the analyze-experiment regex anchors at
   `.claude/skills/analyze-experiment/generation.md:387-388`.
2. State-file shape and resume — the state key includes the relative path,
   so eval entries don't collapse onto a single key (the issue #451 comment
   #2 bug). Atomic write + resume skips already-submitted jobs.
3. Drip-feed back-pressure — when squeue returns >= MAX_SUBMIT, no
   submissions until depth drops; 5-second stagger between submissions.
"""

from __future__ import annotations

import json
import re
import signal
import subprocess
from pathlib import Path

import pytest
import yaml

from cruijff_kit.tools.run import _submit_common as common
from cruijff_kit.tools.run import status as run_status
from cruijff_kit.tools.run import submit_inspect, submit_torchtune


# ---------------------------------------------------------------------------
# Regexes (must match analyze-experiment/generation.md exactly)
# ---------------------------------------------------------------------------

SUBMIT_JOB_RE = re.compile(r"SUBMIT_JOB: ([\w.-]+)\n.*?\nJob ID: (\d+)")
SUBMIT_EVAL_RE = re.compile(r"SUBMIT_EVAL: ([\w./-]+)\n.*?\nJob ID: (\d+)")


# ---------------------------------------------------------------------------
# Mock SLURM
# ---------------------------------------------------------------------------


class FakeSlurm:
    """Mock-friendly SLURM. sbatch hands out monotonic JIDs; squeue tracks
    'queued' jobs that the test can advance to terminal state."""

    def __init__(self, queue_depth: int = 0, sbatch_failures: list[str] | None = None):
        self.next_jid = 1000
        self.queue_depth = queue_depth
        self.sbatch_failures = list(
            sbatch_failures or []
        )  # slurm names that should fail
        self.submissions: list[tuple[str, str]] = []  # (cwd_str, slurm_name)
        self.in_queue: dict[str, str] = {}  # jid -> state ("PENDING"/"RUNNING")
        self.finished: dict[str, str] = {}  # jid -> terminal state

    def __call__(self, cmd, *args, **kwargs):
        argv = list(cmd)
        head = argv[0]
        if head == "sbatch":
            return self._sbatch(argv, kwargs)
        if head == "squeue":
            return self._squeue(argv)
        if head == "sacct":
            return self._sacct(argv)
        raise AssertionError(f"unexpected subprocess command: {argv}")

    def _result(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        return subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr=stderr
        )

    def _sbatch(self, argv, kwargs):
        slurm_name = argv[-1]
        cwd = kwargs.get("cwd", "")
        if slurm_name in self.sbatch_failures:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=argv,
                output="",
                stderr="Job violates accounting/QOS policy, job not submitted",
            )
        jid = str(self.next_jid)
        self.next_jid += 1
        self.submissions.append((str(cwd), slurm_name))
        self.in_queue[jid] = "PENDING"
        return self._result(stdout=f"{jid}\n")

    def _squeue(self, argv):
        # Two flavors: `squeue -u <user> -h` (queue depth) or `squeue -j <jid> -h -o %T`
        if "-u" in argv:
            return self._result(stdout="x\n" * self.queue_depth)
        if "-j" in argv:
            j_idx = argv.index("-j") + 1
            jid = argv[j_idx]
            state = self.in_queue.get(jid)
            return self._result(stdout=(state + "\n") if state else "")
        return self._result()

    def _sacct(self, argv):
        j_idx = argv.index("-j") + 1
        jid = argv[j_idx]
        state = self.finished.get(jid)
        return self._result(stdout=(state + "\n") if state else "")

    # Test helpers --------------------------------------------------------
    def finish_all(self, terminal_state: str = "COMPLETED"):
        for jid in list(self.in_queue.keys()):
            self.finished[jid] = terminal_state
            del self.in_queue[jid]


@pytest.fixture
def fake_slurm(monkeypatch):
    fs = FakeSlurm(queue_depth=0)
    monkeypatch.setattr(subprocess, "run", fs)
    return fs


@pytest.fixture
def fast_sleep(monkeypatch):
    """Replace time.sleep + the submitters' sleep with a no-op recorder."""
    calls = []
    monkeypatch.setattr(common.time, "sleep", lambda s: calls.append(s))
    return calls


# ---------------------------------------------------------------------------
# Experiment fixture builders
# ---------------------------------------------------------------------------


def _write_finetune_experiment(
    tmp_path: Path,
    run_names: list[str],
    include_control: bool = True,
) -> Path:
    summary_runs = []
    for name in run_names:
        summary_runs.append({"name": name, "type": "fine-tuned", "model": "x"})
        run_dir = tmp_path / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "finetune.slurm").write_text("#!/bin/bash\n")
    if include_control:
        ctrl_name = run_names[0] + "_base"
        summary_runs.append({"name": ctrl_name, "type": "control", "model": "x"})
        (tmp_path / ctrl_name).mkdir(parents=True, exist_ok=True)
        # No finetune.slurm — control runs don't train.

    (tmp_path / "experiment_summary.yaml").write_text(yaml.dump({"runs": summary_runs}))
    return tmp_path


def _write_eval_experiment(
    tmp_path: Path,
    runs_with_evals: dict[str, list[str]],
) -> Path:
    """`runs_with_evals` maps run_dir name -> list of cell names (e.g.
    'capitalization', 'capitalization_epoch0'). Per-cell layout (#498):
    each cell gets its own directory containing cell.slurm.
    """
    for run_name, cell_names in runs_with_evals.items():
        for cell in cell_names:
            cell_dir = tmp_path / run_name / "eval" / cell
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "cell.slurm").write_text("#!/bin/bash\n")
    return tmp_path


# ---------------------------------------------------------------------------
# (1) Canonical log shape — regex round-trip
# ---------------------------------------------------------------------------


class TestCanonicalLogShape:
    def test_finetune_log_lines_match_analyze_experiment_regex(
        self, tmp_path, fake_slurm, fast_sleep
    ):
        _write_finetune_experiment(tmp_path, ["rank4", "rank8"])

        # Auto-finish jobs as they're submitted so monitor loop exits.
        original_sbatch = fake_slurm._sbatch

        def sbatch_then_finish(argv, kwargs):
            r = original_sbatch(argv, kwargs)
            fake_slurm.finish_all("COMPLETED")
            return r

        fake_slurm._sbatch = sbatch_then_finish

        submit_torchtune.run(tmp_path, user="testuser", max_submit=10)

        log = (tmp_path / "logs" / "run-torchtune.log").read_text()
        matches = SUBMIT_JOB_RE.findall(log)
        names = {m[0] for m in matches}
        jids = [m[1] for m in matches]
        assert names == {"rank4", "rank8"}
        assert all(j.isdigit() for j in jids)
        # ALL_COMPLETE is emitted at end
        assert "ALL_COMPLETE" in log

    def test_eval_log_lines_match_analyze_experiment_regex(
        self, tmp_path, fake_slurm, fast_sleep
    ):
        _write_eval_experiment(
            tmp_path,
            {
                "Llama_base": ["capitalization"],
                "Llama_rank4": ["capitalization_epoch0"],
            },
        )

        original_sbatch = fake_slurm._sbatch

        def sbatch_then_finish(argv, kwargs):
            r = original_sbatch(argv, kwargs)
            fake_slurm.finish_all("COMPLETED")
            return r

        fake_slurm._sbatch = sbatch_then_finish

        submit_inspect.run(tmp_path, user="testuser", max_submit=10)

        log = (tmp_path / "logs" / "run-inspect.log").read_text()
        matches = SUBMIT_EVAL_RE.findall(log)
        names = {m[0] for m in matches}
        # _eval_name() formatting:
        assert names == {
            "Llama_base/capitalization",
            "Llama_rank4/capitalization/epoch0",
        }


# ---------------------------------------------------------------------------
# (2) State-file shape and resume
# ---------------------------------------------------------------------------


class TestStateFileShape:
    def test_state_key_distinguishes_eval_collisions(self, tmp_path):
        """The bug from issue #451 comment #2: keying by work_dir.name
        collapsed all eval entries onto 'eval'."""
        run_a = tmp_path / "run_a" / "eval"
        run_b = tmp_path / "run_b" / "eval"
        run_a.mkdir(parents=True)
        run_b.mkdir(parents=True)

        key_a = common.state_key(run_a, "capitalization.slurm", tmp_path)
        key_b = common.state_key(run_b, "capitalization.slurm", tmp_path)
        assert key_a != key_b
        assert key_a == "run_a/eval/capitalization.slurm"
        assert key_b == "run_b/eval/capitalization.slurm"

    def test_atomic_write_round_trips(self, tmp_path):
        path = tmp_path / "state.json"
        common.write_state_atomic(path, {"k": {"job_id": "1", "state": "RUNNING"}})
        assert common.read_state(path) == {"k": {"job_id": "1", "state": "RUNNING"}}

    def test_corrupt_state_file_resumes_as_empty(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{not valid json")
        # Treat as empty so resume re-checks SLURM truth instead of crashing.
        assert common.read_state(path) == {}

    def test_resume_skips_already_submitted_jobs(
        self, tmp_path, fake_slurm, fast_sleep
    ):
        """If state file already has an entry, the submitter must not re-sbatch."""
        _write_finetune_experiment(tmp_path, ["rank4", "rank8"], include_control=False)

        # Pre-populate state as if rank4 was submitted in a prior run.
        state_path = tmp_path / "logs" / "run-torchtune.state.json"
        common.write_state_atomic(
            state_path,
            {
                "rank4/finetune.slurm": {
                    "job_id": "999",
                    "submitted_at": "2026-05-08 10:00:00",
                    "state": "COMPLETED",
                }
            },
        )

        original_sbatch = fake_slurm._sbatch

        def sbatch_then_finish(argv, kwargs):
            r = original_sbatch(argv, kwargs)
            fake_slurm.finish_all("COMPLETED")
            return r

        fake_slurm._sbatch = sbatch_then_finish

        submit_torchtune.run(tmp_path, user="testuser", max_submit=10)

        # Only rank8 should have been submitted; rank4 was already terminal.
        submitted_slurms = [s[1] for s in fake_slurm.submissions]
        submitted_dirs = [Path(s[0]).name for s in fake_slurm.submissions]
        assert submitted_slurms == ["finetune.slurm"]
        assert submitted_dirs == ["rank8"]


# ---------------------------------------------------------------------------
# (3) Drip-feed back-pressure
# ---------------------------------------------------------------------------


class TestDripFeed:
    def test_back_pressure_blocks_when_queue_full(
        self, tmp_path, fake_slurm, fast_sleep
    ):
        """If queue_depth >= max_submit, sbatch must not run until queue drains."""
        _write_finetune_experiment(tmp_path, ["rank4", "rank8"], include_control=False)

        # Start with the queue full (depth == max_submit). Drain on the second
        # check so the loop can proceed.
        fake_slurm.queue_depth = 2
        check_count = {"n": 0}
        original_squeue = fake_slurm._squeue

        def squeue_drain(argv):
            if "-u" in argv:
                check_count["n"] += 1
                # First poll: full. Subsequent polls: empty.
                fake_slurm.queue_depth = 0 if check_count["n"] >= 2 else 2
            return original_squeue(argv)

        fake_slurm._squeue = squeue_drain

        original_sbatch = fake_slurm._sbatch

        def sbatch_then_finish(argv, kwargs):
            r = original_sbatch(argv, kwargs)
            fake_slurm.finish_all("COMPLETED")
            return r

        fake_slurm._sbatch = sbatch_then_finish

        submit_torchtune.run(tmp_path, user="testuser", max_submit=2)

        # Both jobs eventually submitted, but at least one full back-pressure
        # poll cycle (DEFAULT_POLL_SEC total) elapsed before the second sbatch.
        # We assert on the SUM rather than an exact-value match because
        # interruptible sleeps chunk into 1s slices for sentinel polling.
        assert len(fake_slurm.submissions) == 2
        assert sum(fast_sleep) >= common.DEFAULT_POLL_SEC, (
            f"expected at least {common.DEFAULT_POLL_SEC}s of back-pressure sleep, "
            f"got sleeps {fast_sleep}"
        )

    def test_stagger_between_submissions(self, tmp_path, fake_slurm, fast_sleep):
        """5-second stagger after each submission (HF cache race)."""
        _write_finetune_experiment(tmp_path, ["rank4", "rank8"], include_control=False)

        original_sbatch = fake_slurm._sbatch

        def sbatch_then_finish(argv, kwargs):
            r = original_sbatch(argv, kwargs)
            fake_slurm.finish_all("COMPLETED")
            return r

        fake_slurm._sbatch = sbatch_then_finish

        submit_torchtune.run(tmp_path, user="testuser", max_submit=10)

        # At least one stagger (DEFAULT_STAGGER_SEC total) happened between
        # the two submissions. Sum-based rather than exact-match because
        # interruptible sleeps chunk into 1s slices for sentinel polling.
        assert sum(fast_sleep) >= common.DEFAULT_STAGGER_SEC, (
            f"expected at least {common.DEFAULT_STAGGER_SEC}s of stagger sleep, "
            f"got {fast_sleep}"
        )

    def test_submit_failure_recorded_in_state(self, tmp_path, fake_slurm, fast_sleep):
        """A QoS-style sbatch failure is captured as SUBMIT_FAILED in state and log."""
        _write_finetune_experiment(tmp_path, ["rank4"], include_control=False)
        fake_slurm.sbatch_failures = ["finetune.slurm"]

        submit_torchtune.run(tmp_path, user="testuser", max_submit=10)

        state = json.loads((tmp_path / "logs" / "run-torchtune.state.json").read_text())
        assert state["rank4/finetune.slurm"]["state"] == "SUBMIT_FAILED"
        log = (tmp_path / "logs" / "run-torchtune.log").read_text()
        assert "Result: failure" in log
        assert "QOS policy" in log


# ---------------------------------------------------------------------------
# (4) End-to-end shape — regression for the eval-collision bug
# ---------------------------------------------------------------------------


class TestEvalCollisionRegression:
    def test_two_runs_with_same_eval_filename_get_distinct_state_entries(
        self, tmp_path, fake_slurm, fast_sleep
    ):
        """Pre-#451 regression: when two runs each had `eval/capitalization.slurm`,
        keying by `work_dir.name == 'eval'` collapsed both entries onto one
        state-file key. Resume after interruption then re-submitted everything
        as duplicates.

        Under the per-cell layout (issue #498), each run now has its own cell
        directory (`{run}/eval/capitalization/cell.slurm`), so state-key
        distinctness is doubly enforced by the {run}/eval/{cell}/cell.slurm path.
        """
        _write_eval_experiment(
            tmp_path,
            {
                "run_a": ["capitalization"],
                "run_b": ["capitalization"],
            },
        )

        original_sbatch = fake_slurm._sbatch

        def sbatch_then_finish(argv, kwargs):
            r = original_sbatch(argv, kwargs)
            fake_slurm.finish_all("COMPLETED")
            return r

        fake_slurm._sbatch = sbatch_then_finish

        submit_inspect.run(tmp_path, user="testuser", max_submit=10)

        state = json.loads((tmp_path / "logs" / "run-inspect.state.json").read_text())
        assert "run_a/eval/capitalization/cell.slurm" in state
        assert "run_b/eval/capitalization/cell.slurm" in state
        assert len(state) == 2


# ---------------------------------------------------------------------------
# (4) Detach mechanisms — sentinel + signal
# ---------------------------------------------------------------------------


class TestDetach:
    """Detach exits cleanly, jobs continue, MONITOR_DETACHED is logged.

    All three trigger paths (sentinel, SIGINT, SIGTERM) converge on the same
    exit. Tests poke at the module-level signal flag directly to avoid
    sending real signals to the test runner.
    """

    def test_sentinel_blocks_all_submissions_when_present_pre_run(
        self, tmp_path, fake_slurm, fast_sleep
    ):
        """Sentinel created before submit_* runs → no sbatch happens."""
        _write_finetune_experiment(tmp_path, ["rank4", "rank8"], include_control=False)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / common.SENTINEL_NAME).touch()

        summary = submit_torchtune.run(tmp_path, user="testuser", max_submit=10)

        assert fake_slurm.submissions == []
        assert summary == {}  # nothing in state to summarize
        log = (logs_dir / "run-torchtune.log").read_text()
        assert "MONITOR_DETACHED" in log
        assert "reason=sentinel" in log

    def test_sentinel_mid_drip_feed_stops_further_submissions(
        self, tmp_path, fake_slurm, fast_sleep
    ):
        """Sentinel appearing after the first submission halts the drip-feed."""
        _write_finetune_experiment(
            tmp_path, ["rank4", "rank8", "rank16"], include_control=False
        )
        original_sbatch = fake_slurm._sbatch
        logs_dir = tmp_path / "logs"

        def sbatch_then_sentinel(argv, kwargs):
            r = original_sbatch(argv, kwargs)
            # After the first submission, drop the sentinel.
            if len(fake_slurm.submissions) == 1:
                logs_dir.mkdir(parents=True, exist_ok=True)
                (logs_dir / common.SENTINEL_NAME).touch()
            return r

        fake_slurm._sbatch = sbatch_then_sentinel

        summary = submit_torchtune.run(tmp_path, user="testuser", max_submit=10)

        # Exactly one job submitted — sentinel caught the stagger sleep.
        assert len(fake_slurm.submissions) == 1
        assert summary == {"PENDING": 1}
        log = (logs_dir / "run-torchtune.log").read_text()
        assert "MONITOR_DETACHED" in log

    def test_sentinel_during_monitor_returns_without_terminal_states(
        self, tmp_path, fake_slurm, fast_sleep
    ):
        """Sentinel during the monitor loop exits with jobs still RUNNING.

        Uses resume_monitor=True with a pre-seeded RUNNING state — this is
        the cleanest way to exercise the monitor-side detach path without
        racing the drip-feed.
        """
        _write_finetune_experiment(tmp_path, ["rank4"], include_control=False)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Pre-seed state as if a previous submitter dispatched and exited.
        fake_slurm.in_queue["1000"] = "RUNNING"
        (logs_dir / "run-torchtune.state.json").write_text(
            json.dumps(
                {
                    "rank4/finetune.slurm": {
                        "job_id": "1000",
                        "submitted_at": "2026-05-11 14:00:00",
                        "state": "RUNNING",
                    }
                }
            )
        )
        # Sentinel is present from the start — monitor bails on first sleep.
        (logs_dir / common.SENTINEL_NAME).touch()

        summary = submit_torchtune.run(
            tmp_path, user="testuser", max_submit=10, resume_monitor=True
        )

        assert summary == {"RUNNING": 1}
        log = (logs_dir / "run-torchtune.log").read_text()
        assert "MONITOR_DETACHED" in log
        assert "ALL_COMPLETE" not in log

    def test_sigterm_during_drip_feed(self, tmp_path, fake_slurm, fast_sleep):
        """Simulating SIGTERM mid-drip-feed halts further submissions."""
        _write_finetune_experiment(
            tmp_path, ["rank4", "rank8", "rank16"], include_control=False
        )
        original_sbatch = fake_slurm._sbatch

        def sbatch_then_sigterm(argv, kwargs):
            r = original_sbatch(argv, kwargs)
            if len(fake_slurm.submissions) == 1:
                # Simulate SIGTERM by calling the handler directly. Avoids
                # actually signalling the test runner.
                common._on_detach_signal(signal.SIGTERM, None)
            return r

        fake_slurm._sbatch = sbatch_then_sigterm

        summary = submit_torchtune.run(tmp_path, user="testuser", max_submit=10)

        assert len(fake_slurm.submissions) == 1
        assert summary == {"PENDING": 1}
        log = (tmp_path / "logs" / "run-torchtune.log").read_text()
        assert "MONITOR_DETACHED" in log
        assert "reason=SIGTERM" in log

    def test_detach_state_resets_between_calls(self, tmp_path, fake_slurm, fast_sleep):
        """A SIGTERM-flagged state from one call is cleared at the start of the next.

        Without _reset_detach_state(), a subsequent submitter call in the
        same process would detach immediately on entry — a real hazard for
        tests and any in-process orchestrator.
        """
        common._on_detach_signal(signal.SIGTERM, None)
        assert common._detach_signal_received == "SIGTERM"

        _write_finetune_experiment(tmp_path, ["rank4"], include_control=False)
        original_sbatch = fake_slurm._sbatch

        def sbatch_then_finish(argv, kwargs):
            r = original_sbatch(argv, kwargs)
            fake_slurm.finish_all("COMPLETED")
            return r

        fake_slurm._sbatch = sbatch_then_finish

        summary = submit_torchtune.run(tmp_path, user="testuser", max_submit=10)

        # The pre-existing signal flag did NOT cause an early exit.
        assert len(fake_slurm.submissions) == 1
        assert summary == {"COMPLETED": 1}


# ---------------------------------------------------------------------------
# (5) --resume-monitor flag
# ---------------------------------------------------------------------------


class TestResumeMonitor:
    def test_resume_monitor_skips_submit_phase(self, tmp_path, fake_slurm, fast_sleep):
        """With existing state, --resume-monitor never calls sbatch."""
        _write_finetune_experiment(tmp_path, ["rank4"], include_control=False)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Pre-seed a state file as if the run was already submitted.
        fake_slurm.in_queue["1000"] = "RUNNING"
        seeded_state = {
            "rank4/finetune.slurm": {
                "job_id": "1000",
                "submitted_at": "2026-05-11 14:00:00",
                "state": "RUNNING",
            }
        }
        (logs_dir / "run-torchtune.state.json").write_text(json.dumps(seeded_state))

        # When monitor refreshes, the job transitions to COMPLETED.
        def transition_to_done():
            fake_slurm.finish_all("COMPLETED")

        original_sleep = common.time.sleep

        def sleep_then_finish(s):
            original_sleep(s)
            transition_to_done()

        # Mock the sleep so the first poll causes the transition.
        common.time.sleep = sleep_then_finish
        try:
            summary = submit_torchtune.run(
                tmp_path, user="testuser", max_submit=10, resume_monitor=True
            )
        finally:
            common.time.sleep = original_sleep

        assert fake_slurm.submissions == [], "resume-monitor must not call sbatch"
        assert summary == {"COMPLETED": 1}
        log = (logs_dir / "run-torchtune.log").read_text()
        assert "ALL_COMPLETE" in log

    def test_resume_monitor_with_no_state_warns_and_returns_empty(
        self, tmp_path, fake_slurm, fast_sleep, capsys
    ):
        """No state file → loud stderr warning, no sbatch, empty summary."""
        (tmp_path / "logs").mkdir()

        summary = submit_inspect.run(
            tmp_path, user="testuser", max_submit=10, resume_monitor=True
        )

        assert summary == {}
        assert fake_slurm.submissions == []
        captured = capsys.readouterr()
        assert "no state" in captured.err.lower()


# ---------------------------------------------------------------------------
# (6) status command — read-only snapshot
# ---------------------------------------------------------------------------


class TestStatus:
    def test_snapshot_reads_both_state_files_when_present(
        self, tmp_path, fake_slurm, fast_sleep
    ):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        (logs_dir / "run-torchtune.state.json").write_text(
            json.dumps(
                {
                    "rank4/finetune.slurm": {
                        "job_id": "1000",
                        "submitted_at": "2026-05-11 14:00:00",
                        "state": "COMPLETED",
                    }
                }
            )
        )
        (logs_dir / "run-inspect.state.json").write_text(
            json.dumps(
                {
                    "rank4/eval/cap/cell.slurm": {
                        "job_id": "1001",
                        "submitted_at": "2026-05-11 14:30:00",
                        "state": "RUNNING",
                    }
                }
            )
        )
        fake_slurm.in_queue["1001"] = "RUNNING"

        snapshots = run_status.snapshot(tmp_path)

        assert set(snapshots.keys()) == {"torchtune", "inspect"}
        assert snapshots["torchtune"]["rank4/finetune.slurm"]["state"] == "COMPLETED"
        assert snapshots["inspect"]["rank4/eval/cap/cell.slurm"]["state"] == "RUNNING"

    def test_snapshot_skips_missing_state_files(self, tmp_path, fake_slurm):
        (tmp_path / "logs").mkdir()
        # Only torchtune state exists.
        (tmp_path / "logs" / "run-torchtune.state.json").write_text(json.dumps({}))

        snapshots = run_status.snapshot(tmp_path)

        assert "torchtune" in snapshots
        assert "inspect" not in snapshots

    def test_snapshot_does_not_call_sbatch(self, tmp_path, fake_slurm):
        """status reads + refreshes; it must never submit anything."""
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "run-torchtune.state.json").write_text(
            json.dumps(
                {
                    "x/finetune.slurm": {
                        "job_id": "1000",
                        "submitted_at": "2026-05-11 14:00:00",
                        "state": "COMPLETED",
                    }
                }
            )
        )

        run_status.snapshot(tmp_path)
        assert fake_slurm.submissions == []

    def test_format_table_empty_returns_helpful_message(self):
        out = run_status.format_table({})
        assert "No state files" in out

    def test_format_table_includes_sections_and_totals(self):
        snapshots = {
            "torchtune": {
                "rank4/finetune.slurm": {
                    "job_id": "1000",
                    "submitted_at": "2026-05-11 14:00:00",
                    "state": "COMPLETED",
                }
            },
            "inspect": {
                "rank4/eval/cap/cell.slurm": {
                    "job_id": "1001",
                    "submitted_at": "2026-05-11 14:30:00",
                    "state": "RUNNING",
                }
            },
        }
        out = run_status.format_table(snapshots)
        assert "[torchtune]" in out
        assert "[inspect]" in out
        assert "COMPLETED" in out
        assert "RUNNING" in out
        assert "Total:" in out
        assert "2 tool(s)" in out

    def test_status_emits_all_complete_when_observing_all_terminal(
        self, tmp_path, fake_slurm
    ):
        """When status refresh sees all entries terminal, ALL_COMPLETE lands."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "run-torchtune.state.json").write_text(
            json.dumps(
                {
                    "rank4/finetune.slurm": {
                        "job_id": "1000",
                        "submitted_at": "2026-05-11 14:00:00",
                        "state": "COMPLETED",
                    }
                }
            )
        )

        run_status.snapshot(tmp_path)

        log = (logs_dir / "run-torchtune.log").read_text()
        assert "ALL_COMPLETE" in log
        assert "COMPLETED=1" in log

    def test_status_does_not_emit_all_complete_when_jobs_in_flight(
        self, tmp_path, fake_slurm
    ):
        """Don't write ALL_COMPLETE while anything is still PENDING/RUNNING."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        fake_slurm.in_queue["1000"] = "RUNNING"
        (logs_dir / "run-torchtune.state.json").write_text(
            json.dumps(
                {
                    "rank4/finetune.slurm": {
                        "job_id": "1000",
                        "submitted_at": "2026-05-11 14:00:00",
                        "state": "RUNNING",
                    }
                }
            )
        )

        run_status.snapshot(tmp_path)

        log_path = logs_dir / "run-torchtune.log"
        # Log may not even exist if no STATE_CHANGE blocks were written.
        if log_path.exists():
            assert "ALL_COMPLETE" not in log_path.read_text()

    def test_status_all_complete_emit_is_idempotent(self, tmp_path, fake_slurm):
        """Repeated status calls on a finished experiment never duplicate ALL_COMPLETE."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "run-torchtune.state.json").write_text(
            json.dumps(
                {
                    "rank4/finetune.slurm": {
                        "job_id": "1000",
                        "submitted_at": "2026-05-11 14:00:00",
                        "state": "COMPLETED",
                    }
                }
            )
        )

        run_status.snapshot(tmp_path)
        run_status.snapshot(tmp_path)
        run_status.snapshot(tmp_path)

        log = (logs_dir / "run-torchtune.log").read_text()
        assert log.count("ALL_COMPLETE") == 1

    def test_resume_monitor_after_status_emitted_all_complete_no_duplicate(
        self, tmp_path, fake_slurm, fast_sleep
    ):
        """The guard inside log_all_complete also protects --resume-monitor."""
        _write_finetune_experiment(tmp_path, ["rank4"], include_control=False)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "run-torchtune.state.json").write_text(
            json.dumps(
                {
                    "rank4/finetune.slurm": {
                        "job_id": "1000",
                        "submitted_at": "2026-05-11 14:00:00",
                        "state": "COMPLETED",
                    }
                }
            )
        )

        # Status writes the first (only) ALL_COMPLETE block.
        run_status.snapshot(tmp_path)
        # --resume-monitor would try to emit on exit; guard must skip it.
        submit_torchtune.run(
            tmp_path, user="testuser", max_submit=10, resume_monitor=True
        )

        log = (logs_dir / "run-torchtune.log").read_text()
        assert log.count("ALL_COMPLETE") == 1


# ---------------------------------------------------------------------------
# (7) Live monitor settings (#480) — logs/monitor.json
# ---------------------------------------------------------------------------


def _mk_cfg(tmp_path: Path, **overrides) -> "common._SubmitConfig":
    """Build a _SubmitConfig pointing at tmp_path's logs/ for refresh tests."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return common._SubmitConfig(
        log_path=logs_dir / "run-torchtune.log",
        state_path=logs_dir / "run-torchtune.state.json",
        action_type="SUBMIT_JOB",
        user="testuser",
        experiment_dir=tmp_path,
        **overrides,
    )


class TestMonitorConfig:
    """Live monitor settings via logs/monitor.json (#480).

    Precedence: monitor.json > CLI flag > env var > default. The watcher
    re-reads the file on every poll iteration so an operator (or agent) can
    tune `poll_sec`, `stagger_sec`, `max_submit` mid-run without detach +
    re-attach. Bad values are warned-and-skipped; removing the file reverts
    each knob to its startup baseline (CLI / config.json / default).
    """

    def test_missing_file_is_silent_noop(self, tmp_path, capsys):
        cfg = _mk_cfg(tmp_path)
        common._refresh_monitor_config(cfg)

        assert cfg.poll_sec == common.DEFAULT_POLL_SEC
        assert cfg.stagger_sec == common.DEFAULT_STAGGER_SEC
        assert cfg.max_submit == common.DEFAULT_MAX_SUBMIT
        # Baseline already equals current → no warning, no log written.
        assert capsys.readouterr().err == ""
        if cfg.log_path.exists():
            assert "MONITOR_CONFIG" not in cfg.log_path.read_text()

    def test_removing_file_reverts_to_baseline(self, tmp_path):
        """Deleting monitor.json after an override reverts each knob."""
        cfg = _mk_cfg(tmp_path)
        config_path = tmp_path / "logs" / "monitor.json"

        config_path.write_text(json.dumps({"poll_sec": 10}))
        common._refresh_monitor_config(cfg)
        assert cfg.poll_sec == 10

        config_path.unlink()
        common._refresh_monitor_config(cfg)

        assert cfg.poll_sec == common.DEFAULT_POLL_SEC
        log = cfg.log_path.read_text()
        assert log.count("MONITOR_CONFIG: applied") == 2
        assert "poll_sec=10 (was 60)" in log
        assert "poll_sec=60 (was 10)" in log

    def test_revert_respects_cli_baseline_not_builtin_default(self, tmp_path):
        """Revert target is the startup baseline, not the built-in default."""
        cfg = _mk_cfg(tmp_path, poll_sec=15)
        config_path = tmp_path / "logs" / "monitor.json"

        config_path.write_text(json.dumps({"poll_sec": 10}))
        common._refresh_monitor_config(cfg)
        assert cfg.poll_sec == 10

        config_path.unlink()
        common._refresh_monitor_config(cfg)

        assert cfg.poll_sec == 15
        assert "poll_sec=15 (was 10)" in cfg.log_path.read_text()

    def test_file_present_applies_overrides_and_logs(self, tmp_path):
        cfg = _mk_cfg(tmp_path)
        (tmp_path / "logs" / "monitor.json").write_text(json.dumps({"poll_sec": 10}))

        common._refresh_monitor_config(cfg)

        assert cfg.poll_sec == 10
        assert cfg.stagger_sec == common.DEFAULT_STAGGER_SEC
        assert cfg.max_submit == common.DEFAULT_MAX_SUBMIT
        log = cfg.log_path.read_text()
        assert "MONITOR_CONFIG: applied" in log
        assert "poll_sec=10 (was 60)" in log
        # Unchanged knobs are still shown so the active picture is complete.
        assert "stagger_sec=5 (unchanged)" in log
        assert "max_submit=25 (unchanged)" in log

    def test_live_update_mid_run(self, tmp_path):
        """Two successive edits each emit their own MONITOR_CONFIG block."""
        cfg = _mk_cfg(tmp_path)
        config_path = tmp_path / "logs" / "monitor.json"

        config_path.write_text(json.dumps({"poll_sec": 20}))
        common._refresh_monitor_config(cfg)
        config_path.write_text(json.dumps({"poll_sec": 5}))
        common._refresh_monitor_config(cfg)

        log = cfg.log_path.read_text()
        assert log.count("MONITOR_CONFIG: applied") == 2
        assert "poll_sec=20 (was 60)" in log
        assert "poll_sec=5 (was 20)" in log
        assert cfg.poll_sec == 5

    def test_unchanged_file_does_not_repeat_log(self, tmp_path):
        """File present but values match current cfg → no repeated logging."""
        cfg = _mk_cfg(tmp_path)
        (tmp_path / "logs" / "monitor.json").write_text(json.dumps({"poll_sec": 30}))

        common._refresh_monitor_config(cfg)
        common._refresh_monitor_config(cfg)
        common._refresh_monitor_config(cfg)

        log = cfg.log_path.read_text()
        assert log.count("MONITOR_CONFIG: applied") == 1

    def test_malformed_json_warns_and_keeps_values(self, tmp_path, capsys):
        cfg = _mk_cfg(tmp_path, poll_sec=42.0)
        (tmp_path / "logs" / "monitor.json").write_text("{not valid json")

        common._refresh_monitor_config(cfg)

        assert cfg.poll_sec == 42.0
        captured = capsys.readouterr()
        assert "not valid JSON" in captured.err
        if cfg.log_path.exists():
            assert "MONITOR_CONFIG" not in cfg.log_path.read_text()

    def test_non_object_json_warns_and_keeps_values(self, tmp_path, capsys):
        """Top-level must be an object; arrays / scalars are rejected."""
        cfg = _mk_cfg(tmp_path)
        (tmp_path / "logs" / "monitor.json").write_text(json.dumps([1, 2, 3]))

        common._refresh_monitor_config(cfg)

        assert cfg.poll_sec == common.DEFAULT_POLL_SEC
        assert "must be a JSON object" in capsys.readouterr().err

    def test_unknown_keys_warned_known_keys_applied(self, tmp_path, capsys):
        cfg = _mk_cfg(tmp_path)
        (tmp_path / "logs" / "monitor.json").write_text(
            json.dumps({"poll_sec": 5, "bogus_key": 99})
        )

        common._refresh_monitor_config(cfg)

        assert cfg.poll_sec == 5
        assert "bogus_key" in capsys.readouterr().err

    def test_invalid_values_warned_and_kept(self, tmp_path, capsys):
        """Out-of-range / wrong-type values: warn, keep last good."""
        cfg = _mk_cfg(tmp_path)
        (tmp_path / "logs" / "monitor.json").write_text(
            json.dumps({"poll_sec": -1, "max_submit": 0, "stagger_sec": "fast"})
        )

        common._refresh_monitor_config(cfg)

        assert cfg.poll_sec == common.DEFAULT_POLL_SEC
        assert cfg.max_submit == common.DEFAULT_MAX_SUBMIT
        assert cfg.stagger_sec == common.DEFAULT_STAGGER_SEC
        err = capsys.readouterr().err
        assert "poll_sec must be > 0" in err
        assert "max_submit must be >= 1" in err
        assert "stagger_sec must be a number" in err

    def test_max_submit_rejects_float(self, tmp_path, capsys):
        """max_submit must be an int, not a float (even 25.0)."""
        cfg = _mk_cfg(tmp_path)
        (tmp_path / "logs" / "monitor.json").write_text(
            json.dumps({"max_submit": 25.0})
        )

        common._refresh_monitor_config(cfg)

        assert cfg.max_submit == common.DEFAULT_MAX_SUBMIT
        assert "max_submit must be an integer" in capsys.readouterr().err

    def test_precedence_monitor_over_cli_over_user_config_over_default(
        self, tmp_path, user_config_path, fake_slurm, fast_sleep
    ):
        """End-to-end precedence: monitor.json > CLI > user config > default."""
        _write_finetune_experiment(tmp_path, ["rank4"], include_control=False)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # User config: poll_sec=30. monitor.json: poll_sec=10. CLI: poll_sec=20.
        # Final effective value through the run = 10 (monitor.json wins).
        user_config_path.write_text(json.dumps({"poll_sec": 30}))
        (logs_dir / "monitor.json").write_text(json.dumps({"poll_sec": 10}))

        original_sbatch = fake_slurm._sbatch

        def sbatch_then_finish(argv, kwargs):
            r = original_sbatch(argv, kwargs)
            fake_slurm.finish_all("COMPLETED")
            return r

        fake_slurm._sbatch = sbatch_then_finish

        submit_torchtune.run(tmp_path, user="testuser", max_submit=10, poll_sec=20)

        log = (logs_dir / "run-torchtune.log").read_text()
        # CLI=20 wins over user-config=30 at startup; monitor.json=10 then
        # drops it from 20 on the first poll iteration.
        assert "poll_sec=10 (was 20)" in log

    def test_config_block_does_not_match_jid_harvest_regex(self, tmp_path):
        """MONITOR_CONFIG must not false-match the analyze-experiment harvesters."""
        log_path = tmp_path / "logs" / "run-torchtune.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        common.log_monitor_config(
            log_path,
            current={"poll_sec": 30, "stagger_sec": 5, "max_submit": 25},
            changes={"poll_sec": 60},
        )

        log = log_path.read_text()
        assert "MONITOR_CONFIG: applied" in log
        assert SUBMIT_JOB_RE.findall(log) == []
        assert SUBMIT_EVAL_RE.findall(log) == []


# ---------------------------------------------------------------------------
# (8) User config — <repo>/.config/config.json
# ---------------------------------------------------------------------------


@pytest.fixture
def user_config_path(tmp_path, monkeypatch):
    """Redirect user-config loading to a per-test temp path with cache reset.

    Yields the (initially-nonexistent) target path. Tests write JSON there
    to exercise user-config behavior in isolation from the real
    <repo>/.config/config.json.
    """
    path = tmp_path / "user_config.json"
    monkeypatch.setattr(common, "_USER_CONFIG_PATH", path)
    common._reset_user_config_cache()
    yield path
    common._reset_user_config_cache()


class TestUserConfig:
    """User-level defaults via <repo>/.config/config.json (power-user knob).

    Precedence layer: CLI > user config > built-in default. Live overrides
    via logs/monitor.json sit above CLI. The file is tracked in the repo
    so the defaults are visible; power users can edit it and add
    `git update-index --skip-worktree` if they don't want changes in git status.
    """

    def test_resolve_max_submit_falls_back_to_user_config(self, user_config_path):
        user_config_path.write_text(json.dumps({"max_submit": 42}))
        assert common.resolve_max_submit(None) == 42
        # CLI flag still wins over user config.
        assert common.resolve_max_submit(5) == 5

    def test_resolve_poll_sec_falls_back_to_user_config(self, user_config_path):
        user_config_path.write_text(json.dumps({"poll_sec": 300}))
        assert common.resolve_poll_sec(None) == 300
        assert common.resolve_poll_sec(15.0) == 15.0

    def test_resolve_stagger_sec_falls_back_to_user_config(self, user_config_path):
        user_config_path.write_text(json.dumps({"stagger_sec": 2}))
        assert common.resolve_stagger_sec(None) == 2
        assert common.resolve_stagger_sec(1.0) == 1.0

    def test_missing_user_config_falls_back_to_built_in_default(self, user_config_path):
        # File deliberately not written -> loader returns empty -> DEFAULT_*.
        assert common.resolve_max_submit(None) == common.DEFAULT_MAX_SUBMIT
        assert common.resolve_poll_sec(None) == common.DEFAULT_POLL_SEC
        assert common.resolve_stagger_sec(None) == common.DEFAULT_STAGGER_SEC

    def test_user_config_bad_value_warns_and_falls_back(self, user_config_path, capsys):
        user_config_path.write_text(json.dumps({"poll_sec": -5}))
        assert common.resolve_poll_sec(None) == common.DEFAULT_POLL_SEC
        assert "poll_sec must be > 0" in capsys.readouterr().err

    def test_user_config_malformed_json_warns_and_falls_back(
        self, user_config_path, capsys
    ):
        user_config_path.write_text("{not valid json")
        assert common.resolve_poll_sec(None) == common.DEFAULT_POLL_SEC
        assert "not valid JSON" in capsys.readouterr().err

    def test_user_config_warnings_emit_once_per_process(self, user_config_path, capsys):
        """Cache ensures unknown-key warnings emit on first load, not every resolver call."""
        user_config_path.write_text(json.dumps({"bogus_key": 99}))
        common.resolve_poll_sec(None)
        common.resolve_max_submit(None)
        common.resolve_stagger_sec(None)
        err = capsys.readouterr().err
        assert err.count("bogus_key") == 1

    def test_real_repo_config_matches_built_in_defaults(self):
        """The shipped <repo>/.config/config.json must match the in-code DEFAULT_*.

        The file is documentation of what the defaults are. If a contributor
        changes a DEFAULT_* without updating the file (or vice versa), this
        test catches the drift.
        """
        # Use the un-monkeypatched real path. Reset cache to avoid carry-over.
        common._reset_user_config_cache()
        path = common._USER_CONFIG_PATH
        assert path.exists(), (
            f"shipped repo config missing at {path} — should be tracked"
        )
        data = json.loads(path.read_text())
        assert data == {
            "poll_sec": common.DEFAULT_POLL_SEC,
            "stagger_sec": common.DEFAULT_STAGGER_SEC,
            "max_submit": common.DEFAULT_MAX_SUBMIT,
        }
        common._reset_user_config_cache()


# ---------------------------------------------------------------------------
# (8) OOM auto-retry (#254 PR A)
# ---------------------------------------------------------------------------


def _make_oom_run(tmp_path: Path, run_name: str, batch_size: int = 128) -> Path:
    """Scaffold a single fine-tune run dir with finetune.slurm + finetune.yaml.

    The yaml has a few unrelated keys plus comments so the regex-based edit
    can be exercised against realistic content (preservation of surrounding
    context, not just the target line).
    """
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "finetune.slurm").write_text("#!/bin/bash\n")
    (run_dir / "finetune.yaml").write_text(
        f"""# Generated by setup_finetune.py
my_wandb_project: ""
batch_size: {batch_size}
epochs: 1
model:
  _component_: torchtune.models.llama
"""
    )
    return run_dir


def _write_slurm_log(
    run_dir: Path,
    job_id: str,
    body: str,
    suffix: str = "out",
) -> Path:
    """Scaffold a realistic <run_dir>/artifacts/slurm-<jid>.<suffix> file."""
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / f"slurm-{job_id}.{suffix}"
    path.write_text(body)
    return path


_REAL_TORCH_OOM_TAIL = (
    '  File "/usr/lib/python3.10/site-packages/torch/utils/checkpoint.py", line 121, in forward\n'
    "    outputs = run_function(*args)\n"
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 864.00 MiB. "
    "GPU 0 has a total capacity of 79.25 GiB of which 233.94 MiB is free.\n"
    "srun: error: della-l01g1: task 0: Exited with exit code 1\n"
)


def _mk_oom_cfg(tmp_path: Path, **overrides) -> "common._SubmitConfig":
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return common._SubmitConfig(
        log_path=logs_dir / "run-torchtune.log",
        state_path=logs_dir / "run-torchtune.state.json",
        action_type="SUBMIT_JOB",
        user="testuser",
        experiment_dir=tmp_path,
        **overrides,
    )


class TestOomRetryHelpers:
    """Small-surface tests for the retry-strategy helpers (#254 PR A)."""

    def test_compute_retry_delta_halves_batch_size(self):
        assert common._compute_retry_delta({"batch_size": 128}) == {"batch_size": 64}
        assert common._compute_retry_delta({"batch_size": 17}) == {"batch_size": 8}

    def test_read_batch_size_finds_top_level(self, tmp_path):
        yaml_path = tmp_path / "finetune.yaml"
        yaml_path.write_text("epochs: 3\nbatch_size: 64\nlora_rank: 8\n")
        assert common._read_batch_size(yaml_path) == 64

    def test_read_batch_size_raises_when_absent(self, tmp_path):
        yaml_path = tmp_path / "finetune.yaml"
        yaml_path.write_text("epochs: 3\nlora_rank: 8\n")
        with pytest.raises(ValueError, match="batch_size"):
            common._read_batch_size(yaml_path)

    def test_apply_retry_preserves_surrounding_content(self, tmp_path):
        yaml_path = tmp_path / "finetune.yaml"
        original = (
            "# Header comment\n"
            'my_wandb_project: "x"\n'
            "batch_size: 128\n"
            "epochs: 1\n"
            "model:\n"
            "  _component_: torchtune.models.llama\n"
        )
        yaml_path.write_text(original)
        prev = common._apply_retry_to_finetune_yaml(yaml_path, {"batch_size": 64})
        assert prev == {"batch_size": 128}
        new_text = yaml_path.read_text()
        assert "batch_size: 64" in new_text
        assert "batch_size: 128" not in new_text
        # Comments + adjacent lines survive.
        assert "# Header comment" in new_text
        assert "epochs: 1" in new_text
        assert "_component_: torchtune.models.llama" in new_text

    def test_apply_retry_rejects_unknown_key(self, tmp_path):
        yaml_path = tmp_path / "finetune.yaml"
        yaml_path.write_text("batch_size: 64\n")
        with pytest.raises(NotImplementedError, match="batch_size"):
            common._apply_retry_to_finetune_yaml(yaml_path, {"num_workers": 4})


class TestCudaOomDetection:
    """Helper-level tests for CUDA-OOM-in-slurm-log detection (#254 PR A).

    SLURM's OUT_OF_MEMORY state covers cgroup-tracked host RAM kills, not
    `torch.OutOfMemoryError` (which exits Python with code 1 → SLURM FAILED).
    These tests pin the secondary detection path that catches realistic
    fine-tune OOMs.
    """

    def test_detects_torch_oom_traceback(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_slurm_log(run_dir, "12345", _REAL_TORCH_OOM_TAIL)
        assert common._failed_due_to_cuda_oom(run_dir, "12345") is True

    def test_detects_bare_cuda_out_of_memory_message(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_slurm_log(run_dir, "12345", "...\nRuntimeError: CUDA out of memory\n")
        assert common._failed_due_to_cuda_oom(run_dir, "12345") is True

    def test_no_log_file_returns_false(self, tmp_path):
        # No artifacts/ at all.
        assert common._failed_due_to_cuda_oom(tmp_path / "run", "12345") is False

    def test_log_without_oom_marker_returns_false(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_slurm_log(run_dir, "12345", "Training started...\nKeyError: 'foo'\n")
        assert common._failed_due_to_cuda_oom(run_dir, "12345") is False

    def test_falls_back_to_err_file_when_out_missing(self, tmp_path):
        run_dir = tmp_path / "run"
        _write_slurm_log(run_dir, "12345", _REAL_TORCH_OOM_TAIL, suffix="err")
        assert common._failed_due_to_cuda_oom(run_dir, "12345") is True

    def test_missing_job_id_returns_false(self, tmp_path):
        assert common._failed_due_to_cuda_oom(tmp_path / "run", None) is False
        assert common._failed_due_to_cuda_oom(tmp_path / "run", "") is False


class TestOomRetryHandler:
    """Tests for `_handle_oom_retries` — the per-iteration retry sweep."""

    def _seed_state(
        self,
        work_dir: Path,
        experiment_dir: Path,
        state: str = "OUT_OF_MEMORY",
        **extra,
    ):
        key = common.state_key(work_dir, "finetune.slurm", experiment_dir)
        entry = {"job_id": "999", "submitted_at": "t0", "state": state, **extra}
        return key, entry

    def test_oom_with_retries_left_resubmits_and_halves(self, tmp_path, fake_slurm):
        run_dir = _make_oom_run(tmp_path, "Llama_rank4", batch_size=128)
        cfg = _mk_oom_cfg(tmp_path)
        key, entry = self._seed_state(run_dir, tmp_path)
        state = {key: entry}

        result = common._handle_oom_retries(state, cfg)

        assert result is True
        # State reflected the resubmit.
        assert state[key]["state"] == "PENDING"
        assert state[key]["retry_history"] == [{"batch_size": 64}]
        assert state[key]["job_id"] != "999"
        # finetune.yaml was halved.
        assert "batch_size: 64" in (run_dir / "finetune.yaml").read_text()
        # Log block shape.
        log = cfg.log_path.read_text()
        assert "OOM_RETRY: Llama_rank4" in log
        assert "batch_size 128 -> 64" in log
        assert "(attempt 1/3)" in log
        # Exactly one sbatch call.
        assert len(fake_slurm.submissions) == 1

    def test_two_retries_grows_history_to_two(self, tmp_path, fake_slurm):
        run_dir = _make_oom_run(tmp_path, "Llama_rank4", batch_size=128)
        cfg = _mk_oom_cfg(tmp_path)
        key, entry = self._seed_state(
            run_dir, tmp_path, retry_history=[{"batch_size": 64}]
        )
        # The on-disk yaml reflects the prior retry's state for realism.
        (run_dir / "finetune.yaml").write_text("batch_size: 64\nepochs: 1\n")
        state = {key: entry}

        common._handle_oom_retries(state, cfg)

        assert state[key]["retry_history"] == [
            {"batch_size": 64},
            {"batch_size": 32},
        ]
        assert "batch_size: 32" in (run_dir / "finetune.yaml").read_text()
        assert "(attempt 2/3)" in cfg.log_path.read_text()

    def test_max_retries_hit_logs_exhausted_once(self, tmp_path, fake_slurm):
        run_dir = _make_oom_run(tmp_path, "Llama_rank4")
        cfg = _mk_oom_cfg(tmp_path)
        history = [
            {"batch_size": 64},
            {"batch_size": 32},
            {"batch_size": 16},
        ]
        key, entry = self._seed_state(run_dir, tmp_path, retry_history=history)
        state = {key: entry}

        result1 = common._handle_oom_retries(state, cfg)
        # Second invocation must not re-emit the block (idempotency on resume).
        result2 = common._handle_oom_retries(state, cfg)

        assert result1 is False  # nothing resubmitted
        assert result2 is False
        assert len(fake_slurm.submissions) == 0
        log = cfg.log_path.read_text()
        assert log.count("OOM_EXHAUSTED: Llama_rank4") == 1
        # State entry got the dedupe marker.
        assert state[key].get("oom_exhausted_logged") is True

    def test_no_retry_flag_skips_resubmission(self, tmp_path, fake_slurm):
        run_dir = _make_oom_run(tmp_path, "Llama_rank4")
        cfg = _mk_oom_cfg(tmp_path, no_retry=True)
        key, entry = self._seed_state(run_dir, tmp_path)
        state = {key: entry}

        result = common._handle_oom_retries(state, cfg)

        assert result is False
        assert state[key]["state"] == "OUT_OF_MEMORY"
        assert "retry_history" not in state[key]
        assert len(fake_slurm.submissions) == 0
        # No log writes at all when retry is gated.
        assert not cfg.log_path.exists() or "OOM" not in cfg.log_path.read_text()

    def test_failed_without_cuda_oom_marker_is_skipped(self, tmp_path, fake_slurm):
        """FAILED + no CUDA OOM in slurm log → ordinary failure, not retried."""
        run_dir = _make_oom_run(tmp_path, "Llama_rank4")
        # Slurm log exists but shows an unrelated traceback.
        _write_slurm_log(run_dir, "999", "KeyError: 'foo'\nExit 1\n")
        cfg = _mk_oom_cfg(tmp_path)
        key, entry = self._seed_state(run_dir, tmp_path, state="FAILED")
        state = {key: entry}

        result = common._handle_oom_retries(state, cfg)

        assert result is False
        assert state[key]["state"] == "FAILED"
        assert len(fake_slurm.submissions) == 0

    def test_failed_with_cuda_oom_in_log_triggers_retry(self, tmp_path, fake_slurm):
        """FAILED + CUDA OOM in slurm log → treated as OOM, retried, oom_detected_via='cuda_log'."""
        run_dir = _make_oom_run(tmp_path, "Llama_rank4", batch_size=1024)
        _write_slurm_log(run_dir, "999", _REAL_TORCH_OOM_TAIL)
        cfg = _mk_oom_cfg(tmp_path)
        key, entry = self._seed_state(run_dir, tmp_path, state="FAILED")
        state = {key: entry}

        result = common._handle_oom_retries(state, cfg)

        assert result is True
        assert state[key]["state"] == "PENDING"
        assert state[key]["retry_history"] == [{"batch_size": 512}]
        assert state[key]["oom_detected_via"] == "cuda_log"
        assert state[key]["job_id"] != "999"
        assert "batch_size: 512" in (run_dir / "finetune.yaml").read_text()
        log = cfg.log_path.read_text()
        assert "OOM_RETRY: Llama_rank4" in log
        assert "batch_size 1024 -> 512" in log
        assert len(fake_slurm.submissions) == 1

    def test_slurm_oom_state_sets_detection_to_slurm_state(self, tmp_path, fake_slurm):
        """OUT_OF_MEMORY retry tags the entry oom_detected_via='slurm_state'."""
        run_dir = _make_oom_run(tmp_path, "Llama_rank4", batch_size=128)
        cfg = _mk_oom_cfg(tmp_path)
        key, entry = self._seed_state(run_dir, tmp_path)  # default: OUT_OF_MEMORY
        state = {key: entry}

        common._handle_oom_retries(state, cfg)

        assert state[key]["oom_detected_via"] == "slurm_state"

    def test_completed_entry_is_skipped(self, tmp_path, fake_slurm):
        run_dir = _make_oom_run(tmp_path, "Llama_rank4")
        cfg = _mk_oom_cfg(tmp_path)
        key, entry = self._seed_state(run_dir, tmp_path, state="COMPLETED")
        state = {key: entry}

        result = common._handle_oom_retries(state, cfg)

        assert result is False
        assert len(fake_slurm.submissions) == 0

    def test_eval_run_without_finetune_yaml_is_silently_skipped(
        self, tmp_path, fake_slurm
    ):
        # Build an eval-style work_dir with no finetune.yaml — only eval slurm.
        eval_dir = tmp_path / "Llama_rank4" / "eval"
        eval_dir.mkdir(parents=True)
        (eval_dir / "eval.slurm").write_text("#!/bin/bash\n")
        cfg = _mk_oom_cfg(tmp_path)
        key = common.state_key(eval_dir, "eval.slurm", tmp_path)
        state = {key: {"job_id": "999", "submitted_at": "t0", "state": "OUT_OF_MEMORY"}}

        result = common._handle_oom_retries(state, cfg)

        assert result is False
        assert state[key]["state"] == "OUT_OF_MEMORY"
        assert len(fake_slurm.submissions) == 0

    def test_sbatch_failure_during_retry_logs_loudly(self, tmp_path, fake_slurm):
        run_dir = _make_oom_run(tmp_path, "Llama_rank4")
        fake_slurm.sbatch_failures.append("finetune.slurm")
        cfg = _mk_oom_cfg(tmp_path)
        key, entry = self._seed_state(run_dir, tmp_path)
        state = {key: entry}

        result = common._handle_oom_retries(state, cfg)

        assert result is False
        # Run stays in OUT_OF_MEMORY — not flipped to SUBMIT_FAILED, because
        # the retry attempt is distinct from an initial submission failure.
        assert state[key]["state"] == "OUT_OF_MEMORY"
        # history was NOT appended (the retry didn't successfully run).
        assert "retry_history" not in state[key] or state[key]["retry_history"] == []
        log = cfg.log_path.read_text()
        assert "OOM_RETRY_SUBMIT_FAILED: Llama_rank4" in log


class TestOomRetryEndToEnd:
    """End-to-end through `submit_and_monitor`: submit → OOM → retry → complete."""

    def test_oom_then_complete_retries_once_and_finishes(
        self, tmp_path, fake_slurm, fast_sleep
    ):
        run_name = "Llama_rank4"
        _make_oom_run(tmp_path, run_name, batch_size=128)
        # Build the experiment_summary.yaml expected by submit_torchtune.
        (tmp_path / "experiment_summary.yaml").write_text(
            yaml.dump(
                {"runs": [{"name": run_name, "type": "fine-tuned", "model": "x"}]}
            )
        )

        # First sbatch lands an OOM; second sbatch (the retry) lands COMPLETED.
        original_sbatch = fake_slurm._sbatch
        attempt = {"n": 0}

        def staged_sbatch(argv, kwargs):
            result = original_sbatch(argv, kwargs)
            attempt["n"] += 1
            if attempt["n"] == 1:
                fake_slurm.finish_all("OUT_OF_MEMORY")
            else:
                fake_slurm.finish_all("COMPLETED")
            return result

        fake_slurm._sbatch = staged_sbatch

        summary = submit_torchtune.run(tmp_path, user="testuser")

        assert summary == {"COMPLETED": 1}
        # Two submissions: initial + one retry.
        assert len(fake_slurm.submissions) == 2
        log = (tmp_path / "logs" / "run-torchtune.log").read_text()
        assert "OOM_RETRY: Llama_rank4" in log
        assert "ALL_COMPLETE" in log
        # finetune.yaml shows the halved value.
        assert "batch_size: 64" in (tmp_path / run_name / "finetune.yaml").read_text()

    def test_oom_three_times_exhausts(self, tmp_path, fake_slurm, fast_sleep):
        run_name = "Llama_rank4"
        _make_oom_run(tmp_path, run_name, batch_size=128)
        (tmp_path / "experiment_summary.yaml").write_text(
            yaml.dump(
                {"runs": [{"name": run_name, "type": "fine-tuned", "model": "x"}]}
            )
        )

        original_sbatch = fake_slurm._sbatch

        def always_oom_sbatch(argv, kwargs):
            result = original_sbatch(argv, kwargs)
            fake_slurm.finish_all("OUT_OF_MEMORY")
            return result

        fake_slurm._sbatch = always_oom_sbatch

        summary = submit_torchtune.run(tmp_path, user="testuser")

        assert summary == {"OUT_OF_MEMORY": 1}
        # Four total: initial + 3 retries.
        assert len(fake_slurm.submissions) == 4
        log = (tmp_path / "logs" / "run-torchtune.log").read_text()
        assert log.count("OOM_RETRY: Llama_rank4") == 3
        assert "(attempt 1/3)" in log
        assert "(attempt 2/3)" in log
        assert "(attempt 3/3)" in log
        assert "OOM_EXHAUSTED: Llama_rank4" in log
        # Final yaml shows the smallest tried value.
        assert "batch_size: 16" in (tmp_path / run_name / "finetune.yaml").read_text()

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
import subprocess
from pathlib import Path

import pytest
import yaml

from cruijff_kit.tools.run import _submit_common as common
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
    """`runs_with_evals` maps run_dir name -> list of eval slurm filenames."""
    for run_name, eval_files in runs_with_evals.items():
        eval_dir = tmp_path / run_name / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        for ef in eval_files:
            (eval_dir / ef).write_text("#!/bin/bash\n")
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
                "Llama_base": ["capitalization.slurm"],
                "Llama_rank4": ["capitalization_epoch0.slurm"],
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

        # Both jobs eventually submitted, but at least one POLL_SEC sleep
        # happened due to back-pressure.
        assert len(fake_slurm.submissions) == 2
        assert any(s == common.DEFAULT_POLL_SEC for s in fast_sleep), (
            f"expected a {common.DEFAULT_POLL_SEC}s back-pressure sleep, "
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

        stagger_sleeps = [s for s in fast_sleep if s == common.DEFAULT_STAGGER_SEC]
        assert len(stagger_sleeps) >= 1, (
            f"expected at least one {common.DEFAULT_STAGGER_SEC}s stagger sleep, "
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
        as duplicates."""
        _write_eval_experiment(
            tmp_path,
            {
                "run_a": ["capitalization.slurm"],
                "run_b": ["capitalization.slurm"],
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
        assert "run_a/eval/capitalization.slurm" in state
        assert "run_b/eval/capitalization.slurm" in state
        assert len(state) == 2

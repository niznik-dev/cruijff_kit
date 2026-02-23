"""Unit tests for tools/slurm/compute_metrics.py

Run with:
    pytest tests/unit/test_compute_metrics.py -v

Tests use fixture data — no cluster or GPU required.
"""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from cruijff_kit.tools.slurm.compute_metrics import (
    check_jobstats_available,
    extract_jobstats_notes,
    format_compute_table,
    parse_jobstats_json,
    parse_sacct_time_limit,
    parse_seff_output,
    run_jobstats,
    summarize_gpu_metrics,
)


# =============================================================================
# parse_seff_output()
# =============================================================================

class TestParseSeffOutput:

    SAMPLE_SEFF = textwrap.dedent("""\
        Job ID: 12345678
        Cluster: della
        User/Group: mjs3/mjs3
        State: COMPLETED (exit code 0)
        Nodes: 1
        Cores per node: 4
        CPU Utilized: 00:08:30
        CPU Efficiency: 56.67% of 00:15:00 core-walltime
        Job Wall-clock time: 00:09:52
        Job Time Limit: 00:15:00
        Memory Utilized: 6.30 GB
        Memory Efficiency: 15.75% of 40.00 GB
    """)

    def test_parses_job_id(self):
        result = parse_seff_output(self.SAMPLE_SEFF)
        assert result["job_id"] == "12345678"

    def test_parses_state(self):
        result = parse_seff_output(self.SAMPLE_SEFF)
        assert result["state"] == "COMPLETED"

    def test_parses_cpu_efficiency(self):
        result = parse_seff_output(self.SAMPLE_SEFF)
        assert result["cpu_efficiency"] == "56.67%"

    def test_parses_wall_time(self):
        result = parse_seff_output(self.SAMPLE_SEFF)
        assert result["wall_time"] == "00:09:52"

    def test_parses_time_limit(self):
        result = parse_seff_output(self.SAMPLE_SEFF)
        assert result["time_limit"] == "00:15:00"

    def test_parses_mem_used_gb(self):
        result = parse_seff_output(self.SAMPLE_SEFF)
        assert result["mem_used_gb"] == 6.30

    def test_parses_mem_requested_gb(self):
        result = parse_seff_output(self.SAMPLE_SEFF)
        assert result["mem_requested_gb"] == 40.00

    def test_parses_mem_efficiency(self):
        result = parse_seff_output(self.SAMPLE_SEFF)
        assert result["mem_efficiency"] == "15.75%"

    def test_handles_mb_memory(self):
        seff_text = textwrap.dedent("""\
            Job ID: 99999
            State: COMPLETED (exit code 0)
            Memory Utilized: 512.00 MB
            Memory Efficiency: 1.25% of 40.00 GB
        """)
        result = parse_seff_output(seff_text)
        assert result["mem_used_gb"] == 0.5

    def test_handles_empty_input(self):
        result = parse_seff_output("")
        assert result["job_id"] is None
        assert result["state"] is None
        assert result["wall_time"] is None


# =============================================================================
# parse_sacct_time_limit()
# =============================================================================

class TestParseSacctTimeLimit:

    def test_basic_time_limit(self):
        assert parse_sacct_time_limit("00:15:00\n") == "00:15:00"

    def test_multiline_takes_first(self):
        """sacct may return lines for job steps; we take the first."""
        assert parse_sacct_time_limit("01:00:00\n01:00:00\n") == "01:00:00"

    def test_empty_returns_none(self):
        assert parse_sacct_time_limit("") is None

    def test_unknown_returns_none(self):
        assert parse_sacct_time_limit("PARTITION_LIMIT\n") is None

    def test_day_format(self):
        """sacct can return D-HH:MM:SS for long jobs."""
        assert parse_sacct_time_limit("1-00:00:00\n") == "1-00:00:00"


# =============================================================================
# summarize_gpu_metrics()
# =============================================================================

class TestSummarizeGpuMetrics:

    def test_basic_csv(self, tmp_path):
        csv_content = textwrap.dedent("""\
            timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W], temperature.gpu
            2026/02/17 10:00:00.000, 80 %, 45 %, 20480 MiB, 40960 MiB, 250.00 W, 65
            2026/02/17 10:00:30.000, 90 %, 50 %, 22528 MiB, 40960 MiB, 270.00 W, 68
            2026/02/17 10:01:00.000, 70 %, 40 %, 18432 MiB, 40960 MiB, 230.00 W, 63
        """)
        csv_path = tmp_path / "gpu_metrics.csv"
        csv_path.write_text(csv_content)

        result = summarize_gpu_metrics(csv_path)
        assert result["gpu_util_mean"] == 80.0
        assert result["gpu_util_max"] == 90.0
        assert result["gpu_mem_total_gb"] == 40.0
        assert result["sample_count"] == 3
        assert result["power_mean_w"] == 250.0
        assert result["gpu_util_unavailable"] is False

    def test_empty_csv(self, tmp_path):
        csv_path = tmp_path / "gpu_metrics.csv"
        csv_path.write_text("timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W], temperature.gpu\n")

        result = summarize_gpu_metrics(csv_path)
        assert result["gpu_util_mean"] is None
        assert result["sample_count"] == 0

    def test_handles_malformed_rows(self, tmp_path):
        csv_content = textwrap.dedent("""\
            timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W], temperature.gpu
            2026/02/17 10:00:00.000, 80 %, 45 %, 20480 MiB, 40960 MiB, 250.00 W, 65
            bad row with insufficient columns
            2026/02/17 10:01:00.000, 70 %, 40 %, 18432 MiB, 40960 MiB, 230.00 W, 63
        """)
        csv_path = tmp_path / "gpu_metrics.csv"
        csv_path.write_text(csv_content)

        result = summarize_gpu_metrics(csv_path)
        assert result["sample_count"] == 2
        assert result["gpu_util_mean"] == 75.0

    def test_mig_gpu_na_utilization(self, tmp_path):
        """MIG GPUs report [N/A] for utilization but have valid memory/power."""
        csv_content = textwrap.dedent("""\
            timestamp, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB], power.draw [W], temperature.gpu
            2026/02/20 14:28:12.659, [N/A], [N/A], 177 MiB, 81920 MiB, 96.01 W, 41
            2026/02/20 14:28:42.685, [N/A], [N/A], 3984 MiB, 81920 MiB, 180.40 W, 44
            2026/02/20 14:29:12.722, [N/A], [N/A], 3984 MiB, 81920 MiB, 188.84 W, 49
        """)
        csv_path = tmp_path / "gpu_metrics.csv"
        csv_path.write_text(csv_content)

        result = summarize_gpu_metrics(csv_path)
        assert result["gpu_util_unavailable"] is True
        assert result["gpu_util_mean"] is None
        assert result["gpu_util_max"] is None
        assert result["gpu_mem_used_mean_gb"] is not None  # memory still parsed
        assert result["gpu_mem_total_gb"] == 80.0
        assert result["power_mean_w"] is not None
        assert result["sample_count"] == 3


# =============================================================================
# format_compute_table()
# =============================================================================

class TestFormatComputeTable:

    def test_basic_table(self):
        jobs = [
            {
                "run_name": "1B_rank4",
                "job_type": "finetune",
                "wall_time": "00:09:52",
                "time_limit": "00:15:00",
                "gpu_util_mean": 80.0,
                "gpu_mem_used_mean_gb": 20.0,
                "power_mean_w": 250.0,
            },
        ]
        table = format_compute_table(jobs)
        assert "1B_rank4" in table
        assert "finetune" in table
        assert "80.0%" in table
        assert "250W" in table
        # Should have header, separator, and one data row
        lines = table.strip().split("\n")
        assert len(lines) == 3

    def test_missing_gpu_metrics(self):
        jobs = [
            {
                "run_name": "1B_eval",
                "job_type": "eval",
                "wall_time": "00:01:13",
                "time_limit": "00:10:00",
            },
        ]
        table = format_compute_table(jobs)
        assert "1B_eval" in table
        # Missing GPU metrics should show as "-"
        lines = table.strip().split("\n")
        data_row = lines[-1]
        # GPU Util, GPU Mem, and Power should all be "-"
        cells = [c.strip() for c in data_row.split("|") if c.strip()]
        assert cells[4] == "-"  # GPU Util
        assert cells[5] == "-"  # GPU Mem
        assert cells[6] == "-"  # Power

    def test_multiple_jobs(self):
        jobs = [
            {"run_name": "job1", "job_type": "finetune", "wall_time": "00:10:00", "time_limit": "00:15:00"},
            {"run_name": "job2", "job_type": "eval", "wall_time": "00:01:00", "time_limit": "00:10:00"},
        ]
        table = format_compute_table(jobs)
        lines = table.strip().split("\n")
        assert len(lines) == 4  # header + separator + 2 data rows

    def test_util_unavailable_footnote(self):
        jobs = [
            {
                "run_name": "1B_bs2",
                "job_type": "finetune",
                "wall_time": "00:05:38",
                "gpu_util_unavailable": True,
                "gpu_util_mean": None,
                "gpu_mem_used_mean_gb": 4.9,
                "gpu_mem_total_gb": 80.0,
                "power_mean_w": 200.0,
            },
        ]
        table = format_compute_table(jobs)
        assert "N/A*" in table
        assert "unavailable" in table
        assert "MIG" in table  # mentioned as common cause

    def test_no_footnote_when_util_available(self):
        jobs = [
            {
                "run_name": "1B_bs2",
                "job_type": "finetune",
                "wall_time": "00:05:38",
                "gpu_util_mean": 80.0,
                "gpu_mem_used_mean_gb": 20.0,
                "power_mean_w": 250.0,
            },
        ]
        table = format_compute_table(jobs)
        assert "N/A*" not in table
        assert "unavailable" not in table

    def test_mem_used_total_format(self):
        """Memory column shows used/total when both values present."""
        jobs = [
            {
                "run_name": "1B_bs2",
                "job_type": "finetune",
                "wall_time": "00:05:00",
                "gpu_util_mean": 65.0,
                "gpu_mem_used_mean_gb": 3.7,
                "gpu_mem_total_gb": 80.0,
                "power_mean_w": 171.0,
            },
        ]
        table = format_compute_table(jobs)
        assert "3.7/80.0" in table

    def test_valid_markdown_format(self):
        jobs = [
            {"run_name": "test", "job_type": "finetune", "wall_time": "00:05:00", "time_limit": "00:15:00"},
        ]
        table = format_compute_table(jobs)
        lines = table.strip().split("\n")
        # Header and separator should start/end with |
        assert lines[0].startswith("|")
        assert lines[0].endswith("|")
        assert lines[1].startswith("|")
        # Separator should contain ---
        assert "---" in lines[1]


# =============================================================================
# check_jobstats_available()
# =============================================================================

class TestCheckJobstatsAvailable:

    @patch("cruijff_kit.tools.slurm.compute_metrics.shutil.which")
    def test_returns_true_when_found(self, mock_which):
        mock_which.return_value = "/usr/bin/jobstats"
        assert check_jobstats_available() is True
        mock_which.assert_called_once_with("jobstats")

    @patch("cruijff_kit.tools.slurm.compute_metrics.shutil.which")
    def test_returns_false_when_missing(self, mock_which):
        mock_which.return_value = None
        assert check_jobstats_available() is False


# =============================================================================
# run_jobstats()
# =============================================================================

class TestRunJobstats:

    SAMPLE_JSON = {
        "nodes": {
            "della-l04g3": {
                "cpus": 4,
                "total_memory": 42949672960,
                "used_memory": 6764573696,
                "total_time": 342.5,
            }
        },
        "total_time": 592,
    }

    @patch("cruijff_kit.tools.slurm.compute_metrics.subprocess.run")
    def test_json_mode_success(self, mock_run):
        mock_run.return_value = type("R", (), {
            "returncode": 0,
            "stdout": json.dumps(self.SAMPLE_JSON),
        })()
        result = run_jobstats("12345", json_mode=True)
        assert result == self.SAMPLE_JSON
        mock_run.assert_called_once_with(
            ["jobstats", "-j", "12345"],
            capture_output=True, text=True, timeout=30,
        )

    @patch("cruijff_kit.tools.slurm.compute_metrics.subprocess.run")
    def test_text_mode_success(self, mock_run):
        mock_run.return_value = type("R", (), {
            "returncode": 0,
            "stdout": "Some formatted output\nNotes:\n* reduce memory",
        })()
        result = run_jobstats("12345", json_mode=False)
        assert "reduce memory" in result
        mock_run.assert_called_once_with(
            ["jobstats", "-n", "12345"],
            capture_output=True, text=True, timeout=30,
        )

    @patch("cruijff_kit.tools.slurm.compute_metrics.subprocess.run")
    def test_timeout_returns_none(self, mock_run):
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd="jobstats", timeout=30)
        assert run_jobstats("12345") is None

    @patch("cruijff_kit.tools.slurm.compute_metrics.subprocess.run")
    def test_nonzero_exit_returns_none(self, mock_run):
        mock_run.return_value = type("R", (), {
            "returncode": 1,
            "stdout": "",
        })()
        assert run_jobstats("12345") is None

    @patch("cruijff_kit.tools.slurm.compute_metrics.subprocess.run")
    def test_empty_nodes_returns_none(self, mock_run):
        mock_run.return_value = type("R", (), {
            "returncode": 0,
            "stdout": json.dumps({"nodes": {}, "total_time": 100}),
        })()
        assert run_jobstats("12345") is None

    @patch("cruijff_kit.tools.slurm.compute_metrics.subprocess.run")
    def test_invalid_json_returns_none(self, mock_run):
        mock_run.return_value = type("R", (), {
            "returncode": 0,
            "stdout": "not valid json {{{",
        })()
        assert run_jobstats("12345") is None


# =============================================================================
# parse_jobstats_json()
# =============================================================================

class TestParseJobstatsJson:

    def test_single_node(self):
        js = {
            "nodes": {
                "della-l04g3": {
                    "cpus": 4,
                    "total_memory": 42949672960,   # 40 GB
                    "used_memory": 6442450944,      # 6 GB
                    "total_time": 1200.0,           # 1200 CPU-seconds
                }
            },
            "total_time": 600,  # 600 wall seconds
        }
        result = parse_jobstats_json(js)
        assert result["cpu_cores"] == 4
        # efficiency = 1200 / (600 * 4) * 100 = 50%
        assert result["cpu_efficiency_pct"] == 50.0
        assert result["cpu_mem_used_gb"] == 6.0
        assert result["cpu_mem_allocated_gb"] == 40.0
        assert result["cpu_mem_efficiency_pct"] == 15.0
        assert result["wall_time_seconds"] == 600
        assert result["nodes"] == ["della-l04g3"]

    def test_multi_node(self):
        js = {
            "nodes": {
                "node1": {
                    "cpus": 4,
                    "total_memory": 21474836480,    # 20 GB
                    "used_memory": 5368709120,       # 5 GB
                    "total_time": 800.0,
                },
                "node2": {
                    "cpus": 4,
                    "total_memory": 21474836480,    # 20 GB
                    "used_memory": 3221225472,       # 3 GB
                    "total_time": 400.0,
                },
            },
            "total_time": 300,
        }
        result = parse_jobstats_json(js)
        assert result["cpu_cores"] == 8
        # efficiency = (800+400) / (300*8) * 100 = 50%
        assert result["cpu_efficiency_pct"] == 50.0
        assert result["cpu_mem_used_gb"] == pytest.approx(8.0, abs=0.1)
        assert result["cpu_mem_allocated_gb"] == pytest.approx(40.0, abs=0.1)
        assert len(result["nodes"]) == 2

    def test_empty_nodes(self):
        result = parse_jobstats_json({"nodes": {}, "total_time": 100})
        assert result["cpu_cores"] is None
        assert result["cpu_efficiency_pct"] is None
        assert result["nodes"] == []

    def test_missing_nodes_key(self):
        result = parse_jobstats_json({"total_time": 100})
        assert result["cpu_cores"] is None

    def test_missing_fields_in_node(self):
        """Nodes with missing fields should not crash."""
        js = {
            "nodes": {"n1": {"cpus": 2}},
            "total_time": 100,
        }
        result = parse_jobstats_json(js)
        assert result["cpu_cores"] == 2
        # No total_time in node → 0 CPU-seconds → 0% efficiency
        assert result["cpu_efficiency_pct"] == 0.0

    def test_zero_wall_time(self):
        js = {
            "nodes": {"n1": {"cpus": 4, "total_time": 100}},
            "total_time": 0,
        }
        result = parse_jobstats_json(js)
        assert result["cpu_efficiency_pct"] is None
        assert result["wall_time_seconds"] is None


# =============================================================================
# extract_jobstats_notes()
# =============================================================================

class TestExtractJobstatsNotes:

    def test_extracts_notes(self):
        text = textwrap.dedent("""\
            Job 12345 ran on della-l04g3
            CPU: 4 cores, 50% efficiency

            Notes:
            * Consider reducing memory allocation from 40 GB to 8 GB
            * Job could run with --time=00:10:00 instead of 01:00:00
        """)
        notes = extract_jobstats_notes(text)
        assert len(notes) == 2
        assert "reducing memory" in notes[0]
        assert "00:10:00" in notes[1]

    def test_no_notes_section(self):
        text = "Job 12345 ran on della-l04g3\nCPU: 4 cores"
        assert extract_jobstats_notes(text) == []

    def test_filters_grafana_urls(self):
        text = textwrap.dedent("""\
            Notes:
            * Reduce memory to 8 GB
            * https://grafana.rc.princeton.edu/d/123/job-stats
            * Lower time limit to 10 minutes
        """)
        notes = extract_jobstats_notes(text)
        assert len(notes) == 2
        assert all("grafana" not in n.lower() for n in notes)

    def test_strips_ansi_codes(self):
        text = (
            "\x1b[1mNotes:\x1b[0m\n"
            "* \x1b[33mReduce memory to 8 GB\x1b[0m\n"
        )
        notes = extract_jobstats_notes(text)
        assert len(notes) == 1
        assert "\x1b" not in notes[0]
        assert "Reduce memory" in notes[0]

    def test_empty_input(self):
        assert extract_jobstats_notes("") == []


# =============================================================================
# format_compute_table() — CPU columns
# =============================================================================

class TestFormatComputeTableCPU:

    def test_cpu_columns_present_when_data_available(self):
        jobs = [
            {
                "run_name": "1B_rank4",
                "job_type": "finetune",
                "wall_time": "00:09:52",
                "time_limit": "00:15:00",
                "cpu_efficiency_pct": 56.7,
                "cpu_mem_used_gb": 6.3,
                "cpu_mem_allocated_gb": 40.0,
                "gpu_util_mean": 80.0,
                "gpu_mem_used_mean_gb": 20.0,
                "power_mean_w": 250.0,
            },
        ]
        table = format_compute_table(jobs)
        assert "CPU Eff" in table
        assert "CPU Mem (GB)" in table
        assert "56.7%" in table
        assert "6.3/40.0" in table

    def test_no_cpu_columns_without_data(self):
        jobs = [
            {
                "run_name": "1B_eval",
                "job_type": "eval",
                "wall_time": "00:01:13",
                "time_limit": "00:10:00",
                "gpu_util_mean": 60.0,
            },
        ]
        table = format_compute_table(jobs)
        assert "CPU Eff" not in table
        assert "CPU Mem (GB)" not in table

    def test_mixed_cpu_availability(self):
        """One job has CPU data, another doesn't — columns shown, missing = '-'."""
        jobs = [
            {
                "run_name": "job1",
                "job_type": "finetune",
                "wall_time": "00:10:00",
                "cpu_efficiency_pct": 45.0,
                "cpu_mem_used_gb": 8.0,
                "cpu_mem_allocated_gb": 40.0,
            },
            {
                "run_name": "job2",
                "job_type": "eval",
                "wall_time": "00:01:00",
            },
        ]
        table = format_compute_table(jobs)
        assert "CPU Eff" in table
        lines = table.strip().split("\n")
        # job2 row should have "-" for CPU columns
        job2_row = [l for l in lines if "job2" in l][0]
        cells = [c.strip() for c in job2_row.split("|") if c.strip()]
        # With CPU: Run, Type, Wall, TimeLimit, CPUEff, CPUMem, GPUUtil, GPUMem, Power
        assert cells[4] == "-"  # CPU Eff
        assert cells[5] == "-"  # CPU Mem

    def test_recommendations_section(self):
        jobs = [
            {"run_name": "1B_rank4", "job_type": "finetune", "wall_time": "00:09:52"},
        ]
        recs = {
            "1B_rank4": [
                "Reduce memory allocation from 40 GB to 8 GB",
                "Lower time limit to 00:10:00",
            ],
        }
        table = format_compute_table(jobs, recommendations=recs)
        assert "### Resource Recommendations" in table
        assert "**1B_rank4:**" in table
        assert "- Reduce memory allocation" in table

    def test_empty_recommendations_not_shown(self):
        jobs = [
            {"run_name": "1B_rank4", "job_type": "finetune", "wall_time": "00:09:52"},
        ]
        recs = {"1B_rank4": []}
        table = format_compute_table(jobs, recommendations=recs)
        assert "Resource Recommendations" not in table

    def test_backward_compatible_without_cpu(self):
        """Existing tests should still pass — no CPU columns when no CPU data."""
        jobs = [
            {
                "run_name": "1B_rank4",
                "job_type": "finetune",
                "wall_time": "00:09:52",
                "time_limit": "00:15:00",
                "gpu_util_mean": 80.0,
                "gpu_mem_used_mean_gb": 20.0,
                "power_mean_w": 250.0,
            },
        ]
        table = format_compute_table(jobs)
        assert "1B_rank4" in table
        assert "80.0%" in table
        assert "250W" in table
        lines = table.strip().split("\n")
        assert len(lines) == 3  # header + separator + 1 row

"""Unit tests for tools/slurm/compute_metrics.py

Run with:
    pytest tests/unit/test_compute_metrics.py -v

Tests use fixture data — no cluster or GPU required.
"""

import textwrap
from pathlib import Path

import pytest

from cruijff_kit.tools.slurm.compute_metrics import (
    format_compute_table,
    parse_seff_output,
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

"""Compute metrics helpers for SLURM job analysis.

Parses seff output and nvidia-smi CSV logs into structured data for
compute utilization reports.
"""

import csv
import re
from pathlib import Path


def parse_seff_output(seff_text: str) -> dict:
    """Parse seff text output into a structured dict.

    Args:
        seff_text: Raw text output from the ``seff`` command.

    Returns:
        Dict with keys: job_id, state, cpu_efficiency, wall_time,
        time_limit, mem_used_gb, mem_requested_gb, mem_efficiency.
        Missing fields default to None.
    """
    result: dict = {
        "job_id": None,
        "state": None,
        "cpu_efficiency": None,
        "wall_time": None,
        "time_limit": None,
        "mem_used_gb": None,
        "mem_requested_gb": None,
        "mem_efficiency": None,
    }

    patterns = {
        "job_id": r"Job ID:\s*(\d+)",
        "state": r"State:\s*(\S+)",
        "cpu_efficiency": r"CPU Efficiency:\s*([\d.]+%)",
        "wall_time": r"Job Wall-clock time:\s*([\d:]+)",
        "time_limit": r"Job Time Limit:\s*([\d:]+)",
        "mem_efficiency": r"Memory Efficiency:\s*([\d.]+%)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, seff_text)
        if m:
            result[key] = m.group(1)

    # Parse memory used (e.g. "6.30 GB", "512.00 MB")
    mem_used_match = re.search(r"Memory Utilized:\s*([\d.]+)\s*(GB|MB|KB)", seff_text)
    if mem_used_match:
        val = float(mem_used_match.group(1))
        unit = mem_used_match.group(2)
        if unit == "MB":
            val /= 1024
        elif unit == "KB":
            val /= 1024 * 1024
        result["mem_used_gb"] = round(val, 2)

    # Parse memory requested from "Memory Efficiency: X% of 40.00 GB"
    mem_req_match = re.search(r"Memory Efficiency:.*?of\s+([\d.]+)\s*(GB|MB|KB)", seff_text)
    if mem_req_match:
        val = float(mem_req_match.group(1))
        unit = mem_req_match.group(2)
        if unit == "MB":
            val /= 1024
        elif unit == "KB":
            val /= 1024 * 1024
        result["mem_requested_gb"] = round(val, 2)

    return result


def summarize_gpu_metrics(csv_path: Path) -> dict:
    """Compute summary statistics from an nvidia-smi CSV log.

    Args:
        csv_path: Path to the CSV file produced by ``nvidia-smi --format=csv``.

    Returns:
        Dict with keys: gpu_util_mean, gpu_util_max, gpu_mem_used_mean_gb,
        gpu_mem_total_gb, power_mean_w, sample_count.
    """
    gpu_utils: list[float] = []
    mem_useds: list[float] = []
    mem_totals: list[float] = []
    powers: list[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip header rows (contain text like "utilization.gpu [%]")
            if not row or any("[" in cell for cell in row):
                continue
            try:
                # Columns: timestamp, gpu_util%, mem_util%, mem_used MiB,
                #          mem_total MiB, power W, temp C
                gpu_util = float(row[1].strip().rstrip(" %"))
                mem_used = float(row[3].strip().split()[0])  # MiB
                mem_total = float(row[4].strip().split()[0])  # MiB
                power = float(row[5].strip().split()[0])  # W
                gpu_utils.append(gpu_util)
                mem_useds.append(mem_used)
                mem_totals.append(mem_total)
                powers.append(power)
            except (ValueError, IndexError):
                continue

    if not gpu_utils:
        return {
            "gpu_util_mean": None,
            "gpu_util_max": None,
            "gpu_mem_used_mean_gb": None,
            "gpu_mem_total_gb": None,
            "power_mean_w": None,
            "sample_count": 0,
        }

    return {
        "gpu_util_mean": round(sum(gpu_utils) / len(gpu_utils), 1),
        "gpu_util_max": round(max(gpu_utils), 1),
        "gpu_mem_used_mean_gb": round(sum(mem_useds) / len(mem_useds) / 1024, 1),
        "gpu_mem_total_gb": round(mem_totals[0] / 1024, 1),
        "power_mean_w": round(sum(powers) / len(powers), 0),
        "sample_count": len(gpu_utils),
    }


def format_compute_table(jobs: list[dict]) -> str:
    """Format a list of job metrics as a markdown table.

    Args:
        jobs: List of dicts, each with keys: run_name, job_type, wall_time,
              time_limit, gpu_util_mean, gpu_mem_used_mean_gb, power_mean_w.
              Missing keys render as "-".

    Returns:
        Markdown-formatted table string.
    """
    headers = ["Run", "Type", "Wall Time", "Time Limit", "GPU Util", "GPU Mem (GB)", "Power (W)"]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"

    lines = [header_line, separator]

    for job in jobs:
        gpu_util = job.get("gpu_util_mean")
        gpu_mem = job.get("gpu_mem_used_mean_gb")
        power = job.get("power_mean_w")

        cells = [
            job.get("run_name", "-"),
            job.get("job_type", "-"),
            job.get("wall_time", "-"),
            job.get("time_limit", "-"),
            f"{gpu_util}%" if gpu_util is not None else "-",
            f"{gpu_mem}" if gpu_mem is not None else "-",
            f"{int(power)}W" if power is not None else "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)

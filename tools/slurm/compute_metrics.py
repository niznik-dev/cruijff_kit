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


def parse_sacct_time_limit(sacct_output: str) -> str | None:
    """Parse time limit from sacct output.

    Expects output from ``sacct -j {id} --format=Timelimit -P -n``,
    which returns a bare time string like ``00:15:00``.  Used as a
    fallback when ``seff`` doesn't report Job Time Limit (e.g. on Della).

    Args:
        sacct_output: Raw text from the sacct command.

    Returns:
        Time limit string (e.g. ``"00:15:00"``) or None if unparseable.
    """
    text = sacct_output.strip()
    if not text or text.lower() in ("", "unknown", "partition_limit"):
        return None
    # sacct may return multiple lines for job steps; take the first
    first_line = text.splitlines()[0].strip()
    if re.match(r"[\d:-]+", first_line):
        return first_line
    return None


def _parse_nvidia_field(value: str) -> float | None:
    """Parse a single nvidia-smi CSV field, returning None for [N/A]."""
    value = value.strip()
    if "[N/A]" in value or not value:
        return None
    # Strip units like " %" or " MiB" or " W"
    return float(value.split()[0].rstrip("%"))


def summarize_gpu_metrics(csv_path: Path) -> dict:
    """Compute summary statistics from an nvidia-smi CSV log.

    Args:
        csv_path: Path to the CSV file produced by ``nvidia-smi --format=csv``.

    Returns:
        Dict with keys: gpu_util_mean, gpu_util_max, gpu_mem_used_mean_gb,
        gpu_mem_total_gb, power_mean_w, sample_count, possibly_mig.
        Fields without data are None. ``possibly_mig`` is True when GPU
        utilization is unavailable (reported as [N/A]) but memory/power
        data exists — a signature of MIG-partitioned GPUs.
    """
    gpu_utils: list[float] = []
    mem_useds: list[float] = []
    mem_totals: list[float] = []
    powers: list[float] = []
    saw_na_util = False
    data_rows = 0

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            # Skip header row (contains column names with units in brackets)
            if "utilization" in row[1].lower() or "timestamp" in row[0].lower():
                continue

            data_rows += 1

            # Parse each field independently so [N/A] in one column
            # doesn't discard the entire row
            gpu_util = _parse_nvidia_field(row[1])
            mem_used = _parse_nvidia_field(row[3])
            mem_total = _parse_nvidia_field(row[4])
            power = _parse_nvidia_field(row[5])

            if gpu_util is not None:
                gpu_utils.append(gpu_util)
            else:
                saw_na_util = True
            if mem_used is not None:
                mem_useds.append(mem_used / 1024)  # MiB -> GiB
            if mem_total is not None:
                mem_totals.append(mem_total / 1024)  # MiB -> GiB
            if power is not None:
                powers.append(power)

    # MIG detection: utilization unknown but other metrics present
    possibly_mig = saw_na_util and data_rows > 0 and bool(mem_useds or powers)

    if not mem_useds and not gpu_utils and not powers:
        return {
            "gpu_util_mean": None,
            "gpu_util_max": None,
            "gpu_mem_used_mean_gb": None,
            "gpu_mem_total_gb": None,
            "power_mean_w": None,
            "sample_count": 0,
            "possibly_mig": possibly_mig,
        }

    return {
        "gpu_util_mean": round(sum(gpu_utils) / len(gpu_utils), 1) if gpu_utils else None,
        "gpu_util_max": round(max(gpu_utils), 1) if gpu_utils else None,
        "gpu_mem_used_mean_gb": round(sum(mem_useds) / len(mem_useds), 1) if mem_useds else None,
        "gpu_mem_total_gb": round(mem_totals[0], 1) if mem_totals else None,
        "power_mean_w": round(sum(powers) / len(powers), 0) if powers else None,
        "sample_count": data_rows,
        "possibly_mig": possibly_mig,
    }


def format_compute_table(jobs: list[dict]) -> str:
    """Format a list of job metrics as a markdown table.

    Args:
        jobs: List of dicts, each with keys: run_name, job_type, wall_time,
              time_limit, gpu_util_mean, gpu_mem_used_mean_gb, power_mean_w,
              possibly_mig. Missing or None values render as "-".

    Returns:
        Markdown-formatted table string, with a footnote if any jobs
        appear to have run on MIG-partitioned GPUs.
    """
    headers = ["Run", "Type", "Wall Time", "Time Limit", "GPU Util", "GPU Mem (GB)", "Power (W)"]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"

    lines = [header_line, separator]
    any_mig = False

    for job in jobs:
        gpu_util = job.get("gpu_util_mean")
        gpu_mem = job.get("gpu_mem_used_mean_gb")
        gpu_total = job.get("gpu_mem_total_gb")
        power = job.get("power_mean_w")
        is_mig = job.get("possibly_mig", False)

        if is_mig:
            any_mig = True

        # Show mem as "used/total" when we have both
        if gpu_mem is not None and gpu_total is not None:
            mem_str = f"{gpu_mem}/{gpu_total}"
        elif gpu_mem is not None:
            mem_str = f"{gpu_mem}"
        else:
            mem_str = "-"

        util_str = f"{gpu_util}%" if gpu_util is not None else ("MIG*" if is_mig else "-")

        cells = [
            job.get("run_name") or "-",
            job.get("job_type") or "-",
            job.get("wall_time") or "-",
            job.get("time_limit") or "-",
            util_str,
            mem_str,
            f"{int(power)}W" if power is not None else "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    if any_mig:
        lines.append("")
        lines.append(
            "*\\*MIG: GPU utilization is unavailable for MIG-partitioned GPUs. "
            "These are half-A100s (40 GB) that appear in the `nomig` partition. "
            "Use `--constraint=gpu80` to guarantee a full A100 with utilization metrics.*"
        )

    return "\n".join(lines)

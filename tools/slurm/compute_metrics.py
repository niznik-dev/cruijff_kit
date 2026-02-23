"""Compute metrics helpers for SLURM job analysis.

Parses seff output, nvidia-smi CSV logs, and jobstats output into
structured data for compute utilization reports.
"""

import csv
import json
import re
import shutil
import subprocess
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
    if re.match(r"^\d[\d:-]+$", first_line):
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
        gpu_mem_total_gb, power_mean_w, sample_count, gpu_util_unavailable.
        Fields without data are None. ``gpu_util_unavailable`` is True when
        GPU utilization is reported as [N/A] but memory/power data exists —
        common with MIG-partitioned GPUs but can have other causes.
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

    # Utilization unavailable but other metrics present (common with MIG)
    gpu_util_unavailable = saw_na_util and data_rows > 0 and bool(mem_useds or powers)

    if not mem_useds and not gpu_utils and not powers:
        return {
            "gpu_util_mean": None,
            "gpu_util_max": None,
            "gpu_mem_used_mean_gb": None,
            "gpu_mem_total_gb": None,
            "power_mean_w": None,
            "sample_count": 0,
            "gpu_util_unavailable": gpu_util_unavailable,
        }

    return {
        "gpu_util_mean": round(sum(gpu_utils) / len(gpu_utils), 1) if gpu_utils else None,
        "gpu_util_max": round(max(gpu_utils), 1) if gpu_utils else None,
        "gpu_mem_used_mean_gb": round(sum(mem_useds) / len(mem_useds), 1) if mem_useds else None,
        "gpu_mem_total_gb": round(mem_totals[0], 1) if mem_totals else None,
        "power_mean_w": round(sum(powers) / len(powers), 0) if powers else None,
        "sample_count": data_rows,
        "gpu_util_unavailable": gpu_util_unavailable,
    }


def check_jobstats_available() -> bool:
    """Check whether the ``jobstats`` command is on PATH."""
    return shutil.which("jobstats") is not None


def run_jobstats(job_id: str, json_mode: bool = True) -> dict | str | None:
    """Run jobstats for a given SLURM job.

    Args:
        job_id: SLURM job ID.
        json_mode: If True, run ``jobstats -j`` for JSON output.
            If False, run ``jobstats -n`` for formatted text (notes).

    Returns:
        Parsed JSON dict (json_mode=True), formatted text string
        (json_mode=False), or None on any failure.
    """
    flag = "-j" if json_mode else "-n"
    try:
        result = subprocess.run(
            ["jobstats", flag, str(job_id)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        output = result.stdout.strip()
        if not output:
            return None
        if json_mode:
            data = json.loads(output)
            # Reject empty nodes dict — means Prometheus has no data
            if not data.get("nodes"):
                return None
            return data
        return output
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None


def parse_jobstats_json(js_data: dict) -> dict:
    """Extract CPU metrics from jobstats JSON output.

    Args:
        js_data: Parsed JSON dict from ``jobstats -j``.

    Returns:
        Dict with cpu_cores, cpu_efficiency_pct, cpu_mem_used_gb,
        cpu_mem_allocated_gb, cpu_mem_efficiency_pct, wall_time_seconds,
        and nodes. Missing fields are None.
    """
    result: dict = {
        "cpu_cores": None,
        "cpu_efficiency_pct": None,
        "cpu_mem_used_gb": None,
        "cpu_mem_allocated_gb": None,
        "cpu_mem_efficiency_pct": None,
        "wall_time_seconds": None,
        "nodes": [],
    }

    nodes = js_data.get("nodes")
    if not nodes or not isinstance(nodes, dict):
        return result

    total_cores = 0
    total_cpu_seconds = 0.0
    total_mem_used_bytes = 0
    total_mem_allocated_bytes = 0
    node_names = []

    for name, info in nodes.items():
        if not isinstance(info, dict):
            continue
        node_names.append(name)
        cores = info.get("cpus", 0)
        total_cores += cores
        total_cpu_seconds += info.get("total_time", 0.0)
        total_mem_used_bytes += info.get("used_memory", 0)
        total_mem_allocated_bytes += info.get("total_memory", 0)

    wall_time = js_data.get("total_time")
    result["nodes"] = node_names

    if total_cores > 0:
        result["cpu_cores"] = total_cores

    if wall_time is not None and wall_time > 0:
        result["wall_time_seconds"] = int(wall_time)
        if total_cores > 0:
            result["cpu_efficiency_pct"] = round(
                total_cpu_seconds / (wall_time * total_cores) * 100, 1
            )

    bytes_per_gb = 1024**3
    if total_mem_used_bytes > 0:
        result["cpu_mem_used_gb"] = round(total_mem_used_bytes / bytes_per_gb, 2)
    if total_mem_allocated_bytes > 0:
        result["cpu_mem_allocated_gb"] = round(
            total_mem_allocated_bytes / bytes_per_gb, 2
        )
    if total_mem_allocated_bytes > 0 and total_mem_used_bytes > 0:
        result["cpu_mem_efficiency_pct"] = round(
            total_mem_used_bytes / total_mem_allocated_bytes * 100, 1
        )

    return result


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def extract_jobstats_notes(formatted_output: str) -> list[str]:
    """Extract actionable recommendations from jobstats text output.

    Parses the "Notes" section from ``jobstats -n`` output, strips ANSI
    colour codes, and filters to resource recommendations (memory, time,
    GPU utilization). Skips Grafana URLs and purely informational lines.

    Args:
        formatted_output: Raw text from ``jobstats -n``.

    Returns:
        List of recommendation strings, or empty list if none found.
    """
    # Strip ANSI escape codes
    clean = _ANSI_RE.sub("", formatted_output)

    # Find the Notes section
    notes_start = None
    for i, line in enumerate(clean.splitlines()):
        if line.strip().lower().startswith("notes"):
            notes_start = i + 1
            break

    if notes_start is None:
        return []

    lines = clean.splitlines()[notes_start:]

    # Skip patterns: Grafana URLs, blank lines, purely informational
    skip_patterns = [
        re.compile(r"https?://", re.IGNORECASE),
        re.compile(r"^\s*$"),
        re.compile(r"^-+$"),  # horizontal rules
    ]

    notes = []
    for line in lines:
        stripped = line.strip().lstrip("*•- ")
        if not stripped:
            continue
        if any(p.search(stripped) for p in skip_patterns):
            continue
        notes.append(stripped)

    return notes


def format_compute_table(
    jobs: list[dict],
    recommendations: dict[str, list[str]] | None = None,
) -> str:
    """Format a list of job metrics as a markdown table.

    Args:
        jobs: List of dicts, each with keys: run_name, job_type, wall_time,
              time_limit, gpu_util_mean, gpu_mem_used_mean_gb, power_mean_w,
              gpu_util_unavailable. Optionally includes cpu_efficiency_pct
              and cpu_mem_used_gb / cpu_mem_allocated_gb for CPU columns.
              Missing or None values render as "-".
        recommendations: Optional dict mapping run names to lists of
              resource recommendation strings (from jobstats). When
              provided, a "Resource Recommendations" subsection is
              appended after the table.

    Returns:
        Markdown-formatted table string, with optional footnotes and
        recommendations.
    """
    # Detect whether any job has CPU data
    has_cpu = any(
        job.get("cpu_efficiency_pct") is not None
        or job.get("cpu_mem_used_gb") is not None
        for job in jobs
    )

    headers = ["Run", "Type", "Wall Time", "Time Limit"]
    if has_cpu:
        headers += ["CPU Eff", "CPU Mem (GB)"]
    headers += ["GPU Util", "GPU Mem (GB)", "Power (W)"]

    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"

    lines = [header_line, separator]
    any_util_unavailable = False

    for job in jobs:
        gpu_util = job.get("gpu_util_mean")
        gpu_mem = job.get("gpu_mem_used_mean_gb")
        gpu_total = job.get("gpu_mem_total_gb")
        power = job.get("power_mean_w")
        util_unavailable = job.get("gpu_util_unavailable", False)

        if util_unavailable:
            any_util_unavailable = True

        # GPU mem: "used/total" when we have both
        if gpu_mem is not None and gpu_total is not None:
            gpu_mem_str = f"{gpu_mem}/{gpu_total}"
        elif gpu_mem is not None:
            gpu_mem_str = f"{gpu_mem}"
        else:
            gpu_mem_str = "-"

        gpu_util_str = (
            f"{gpu_util}%"
            if gpu_util is not None
            else ("N/A*" if util_unavailable else "-")
        )

        cells = [
            job.get("run_name") or "-",
            job.get("job_type") or "-",
            job.get("wall_time") or "-",
            job.get("time_limit") or "-",
        ]

        if has_cpu:
            # CPU efficiency
            cpu_eff = job.get("cpu_efficiency_pct")
            cells.append(f"{cpu_eff}%" if cpu_eff is not None else "-")

            # CPU mem: "used/allocated" when we have both
            cpu_mem_used = job.get("cpu_mem_used_gb")
            cpu_mem_alloc = job.get("cpu_mem_allocated_gb")
            if cpu_mem_used is not None and cpu_mem_alloc is not None:
                cells.append(f"{cpu_mem_used}/{cpu_mem_alloc}")
            elif cpu_mem_used is not None:
                cells.append(f"{cpu_mem_used}")
            else:
                cells.append("-")

        cells += [
            gpu_util_str,
            gpu_mem_str,
            f"{int(power)}W" if power is not None else "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    if any_util_unavailable:
        lines.append("")
        lines.append(
            "*\\*N/A: GPU utilization was unavailable for these jobs. "
            "This is common with MIG-partitioned GPUs, which report memory and power "
            "but not utilization. Check your cluster's GPU allocation policy.*"
        )

    # Append resource recommendations when provided
    if recommendations:
        has_notes = any(notes for notes in recommendations.values())
        if has_notes:
            lines.append("")
            lines.append("### Resource Recommendations")
            lines.append("")
            for run_name, notes in recommendations.items():
                if not notes:
                    continue
                lines.append(f"**{run_name}:**")
                for note in notes:
                    lines.append(f"- {note}")
                lines.append("")

    return "\n".join(lines)

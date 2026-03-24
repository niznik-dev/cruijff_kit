#!/usr/bin/env python3
"""
Archive a completed experiment, preserving all experiment files and deleting
only bulk checkpoint artifacts.

Usage:
    python archive_experiment.py <experiment_dir> [--dry-run] [--force] [--archive-base <path>]

Keeps: entire experiment directory (configs, SLURM scripts, eval logs, analysis, etc.)
Deletes: model checkpoint directories (ck-out-{run_name}/)

Output:
    JSON to stdout with archive results.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def _file_size_bytes(path):
    """Return file size in bytes, or 0 if path doesn't exist."""
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _dir_size_bytes(path):
    """Return total size of all files in a directory tree."""
    total = 0
    p = Path(path)
    if not p.exists():
        return 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _bytes_to_mb(b):
    """Convert bytes to MB, rounded to 1 decimal."""
    return round(b / (1024 * 1024), 1)


def inventory_experiment(experiment_dir, output_dir_base):
    """
    Categorize all experiment files as KEEP or DELETE.

    Args:
        experiment_dir: Path to the experiment directory.
        output_dir_base: Path to the output directory base (contains ck-out-* dirs).

    Returns:
        Dictionary with:
        - experiment_name: str
        - keep_files: list of dicts with path, archive_path, size_bytes
        - delete_paths: list of dicts with path, size_bytes, description
        - keep_total_bytes: int
        - delete_total_bytes: int
        - runs: list of run names
        - incomplete_runs: list of run names missing eval logs
        - findings_source: str or None (path to best findings source)
    """
    exp_dir = Path(experiment_dir)
    summary_path = exp_dir / "experiment_summary.yaml"

    if not summary_path.exists():
        return {
            "status": "error",
            "message": f"experiment_summary.yaml not found in {experiment_dir}",
        }

    try:
        with open(summary_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to parse experiment_summary.yaml: {e}",
        }

    experiment_name = config.get("experiment", {}).get("name", exp_dir.name)
    runs = config.get("runs", [])
    run_names = [r["name"] for r in runs]
    eval_matrix = config.get("evaluation", {}).get("matrix", [])

    # --- KEEP files: entire experiment directory ---
    keep_files = []
    for f in sorted(exp_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(exp_dir)
            keep_files.append(
                {
                    "source": str(f),
                    "archive_path": str(rel),
                    "size_bytes": _file_size_bytes(f),
                }
            )

    # Resolve findings.md source (for reporting which file serves as findings)
    findings_source = None
    if (exp_dir / "findings.md").exists():
        findings_source = str(exp_dir / "findings.md")
    elif (exp_dir / "analysis" / "report.md").exists():
        findings_source = str(exp_dir / "analysis" / "report.md")
    elif (exp_dir / "summary.md").exists():
        findings_source = str(exp_dir / "summary.md")

    # --- DELETE paths: only checkpoint directories ---
    delete_paths = []
    out_base = Path(output_dir_base)
    for run_name in run_names:
        out_dir = out_base / f"ck-out-{run_name}"
        if out_dir.exists():
            delete_paths.append(
                {
                    "path": str(out_dir),
                    "size_bytes": _dir_size_bytes(out_dir),
                    "description": f"Model checkpoints for {run_name}",
                }
            )

    # --- Check completeness ---
    incomplete_runs = []
    for entry in eval_matrix:
        run_name = entry.get("run")
        eval_logs_dir = exp_dir / run_name / "eval" / "logs"
        if not eval_logs_dir.exists() or not list(eval_logs_dir.glob("*.eval")):
            incomplete_runs.append(run_name)

    keep_total = sum(kf["size_bytes"] for kf in keep_files)
    delete_total = sum(dp["size_bytes"] for dp in delete_paths)

    return {
        "status": "success",
        "experiment_name": experiment_name,
        "keep_files": keep_files,
        "delete_paths": delete_paths,
        "keep_total_bytes": keep_total,
        "delete_total_bytes": delete_total,
        "runs": run_names,
        "incomplete_runs": incomplete_runs,
        "findings_source": findings_source,
    }


def create_archive(archive_dir, inventory):
    """
    Copy KEEP files to the archive directory.

    Args:
        archive_dir: Destination archive directory path.
        inventory: Result from inventory_experiment().

    Returns:
        Dictionary with copy results.
    """
    archive = Path(archive_dir)

    if archive.exists():
        return {
            "status": "error",
            "message": f"Archive directory already exists: {archive_dir}",
        }

    copied = []
    errors = []

    for kf in inventory["keep_files"]:
        src = Path(kf["source"])
        dst = archive / kf["archive_path"]

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            copied.append(kf["archive_path"])
        except Exception as e:
            errors.append({"path": kf["archive_path"], "error": str(e)})

    return {
        "status": "success" if not errors else "partial",
        "copied": copied,
        "errors": errors,
        "archive_dir": str(archive),
    }


def verify_archive(archive_dir, inventory):
    """
    Verify all expected files exist in the archive with correct sizes.

    Args:
        archive_dir: Path to the archive directory.
        inventory: Result from inventory_experiment().

    Returns:
        Dictionary with verification results.
    """
    archive = Path(archive_dir)
    missing = []
    size_mismatches = []

    for kf in inventory["keep_files"]:
        dst = archive / kf["archive_path"]
        if not dst.exists():
            missing.append(kf["archive_path"])
        elif dst.stat().st_size != kf["size_bytes"]:
            size_mismatches.append(
                {
                    "path": kf["archive_path"],
                    "expected": kf["size_bytes"],
                    "actual": dst.stat().st_size,
                }
            )

    ok = not missing and not size_mismatches
    return {
        "status": "success" if ok else "error",
        "missing": missing,
        "size_mismatches": size_mismatches,
    }


def delete_originals(experiment_dir, output_dir_base, run_names):
    """
    Remove the experiment directory and checkpoint output directories.

    Args:
        experiment_dir: Path to the experiment directory.
        output_dir_base: Path to the output directory base.
        run_names: List of run names.

    Returns:
        Dictionary with deletion results.
    """
    deleted = []
    errors = []
    freed_bytes = 0

    # Delete checkpoint directories (the big ones)
    out_base = Path(output_dir_base)
    for run_name in run_names:
        out_dir = out_base / f"ck-out-{run_name}"
        if out_dir.exists():
            size = _dir_size_bytes(out_dir)
            try:
                shutil.rmtree(str(out_dir))
                deleted.append(str(out_dir))
                freed_bytes += size
            except Exception as e:
                errors.append({"path": str(out_dir), "error": str(e)})

    # Clean up output base directory if now empty
    if out_base.exists() and not any(out_base.iterdir()):
        try:
            out_base.rmdir()
            deleted.append(str(out_base))
        except Exception as e:
            errors.append({"path": str(out_base), "error": str(e)})

    # Delete experiment directory (now archived)
    exp_dir = Path(experiment_dir)
    if exp_dir.exists():
        size = _dir_size_bytes(exp_dir)
        try:
            shutil.rmtree(str(exp_dir))
            deleted.append(str(exp_dir))
            freed_bytes += size
        except Exception as e:
            errors.append({"path": str(exp_dir), "error": str(e)})

    return {
        "status": "success" if not errors else "partial",
        "deleted": deleted,
        "errors": errors,
        "freed_bytes": freed_bytes,
    }


def archive_experiment(experiment_dir, archive_base, dry_run=False, force=False):
    """
    Main orchestrator: inventory, archive, verify, delete.

    Args:
        experiment_dir: Path to the experiment directory.
        archive_base: Base path for archives (archive lands in {archive_base}/{name}/).
        dry_run: If True, only report what would happen.
        force: If True, archive even if some runs are incomplete.

    Returns:
        JSON-serializable result dictionary.
    """
    if yaml is None:
        return {
            "status": "error",
            "message": "PyYAML not installed. Run: pip install pyyaml",
        }

    exp_dir = Path(experiment_dir)
    if not exp_dir.exists():
        return {
            "status": "error",
            "message": f"Experiment directory not found: {experiment_dir}",
        }

    summary_path = exp_dir / "experiment_summary.yaml"
    if not summary_path.exists():
        return {
            "status": "error",
            "message": f"experiment_summary.yaml not found in {experiment_dir}",
        }

    # Parse to get output directory
    try:
        with open(summary_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to parse experiment_summary.yaml: {e}",
        }

    output_dir_base = config.get("output", {}).get("base_directory", "")
    if not output_dir_base:
        return {
            "status": "error",
            "message": "No output.base_directory found in experiment_summary.yaml",
        }

    # Inventory
    inventory = inventory_experiment(str(exp_dir), output_dir_base)
    if inventory["status"] != "success":
        return inventory

    # Check completeness
    if inventory["incomplete_runs"] and not force:
        return {
            "status": "error",
            "message": "Incomplete runs detected. Use --force to archive anyway.",
            "incomplete_runs": inventory["incomplete_runs"],
        }

    experiment_name = inventory["experiment_name"]
    archive_dir = str(Path(archive_base) / experiment_name)

    if dry_run:
        return {
            "status": "success",
            "mode": "dry-run",
            "experiment": experiment_name,
            "keep": {
                "files": len(inventory["keep_files"]),
                "size_mb": _bytes_to_mb(inventory["keep_total_bytes"]),
            },
            "delete": {
                "checkpoint_dirs": len(inventory["delete_paths"]),
                "size_mb": _bytes_to_mb(inventory["delete_total_bytes"]),
            },
            "archive_path": archive_dir,
            "incomplete_runs": inventory["incomplete_runs"],
            "findings_source": inventory["findings_source"],
        }

    # Create archive
    copy_result = create_archive(archive_dir, inventory)
    if copy_result["status"] == "error":
        return copy_result

    # Verify
    verify_result = verify_archive(archive_dir, inventory)
    if verify_result["status"] != "success":
        return {
            "status": "error",
            "message": "Archive verification failed. Originals NOT deleted.",
            "missing": verify_result["missing"],
            "size_mismatches": verify_result["size_mismatches"],
            "archive_dir": archive_dir,
        }

    # Delete originals (checkpoints + experiment dir, now archived)
    delete_result = delete_originals(str(exp_dir), output_dir_base, inventory["runs"])

    return {
        "status": "success",
        "mode": "archive",
        "experiment": experiment_name,
        "archive_dir": archive_dir,
        "kept": {
            "files": len(copy_result["copied"]),
            "size_mb": _bytes_to_mb(inventory["keep_total_bytes"]),
        },
        "freed": {
            "size_mb": _bytes_to_mb(delete_result["freed_bytes"]),
        },
        "deleted": delete_result["deleted"],
        "incomplete_runs": inventory["incomplete_runs"],
        "findings_source": inventory["findings_source"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Archive a completed experiment, preserving all experiment "
        "files and deleting only checkpoint directories."
    )
    parser.add_argument("experiment_dir", help="Path to the experiment directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be archived/deleted without acting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Archive even if some runs are incomplete",
    )
    parser.add_argument(
        "--archive-base",
        default=None,
        help="Base directory for archives (default: sibling of experiment parent)",
    )
    parser.add_argument(
        "--pretty", "-p", action="store_true", help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    # Default archive base: ck-archive/ as sibling of the experiment's parent
    if args.archive_base is None:
        exp_parent = Path(args.experiment_dir).parent
        archive_base = str(exp_parent.parent / "ck-archive")
    else:
        archive_base = args.archive_base

    result = archive_experiment(
        args.experiment_dir,
        archive_base,
        dry_run=args.dry_run,
        force=args.force,
    )

    if args.pretty:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result))

    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()

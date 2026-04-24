"""Tests for src/tools/experiment/archive_experiment.py"""

import yaml

from cruijff_kit.tools.experiment.archive_experiment import (
    archive_experiment,
    create_archive,
    delete_originals,
    inventory_experiment,
    verify_archive,
)


def _make_experiment(tmp_path, run_names=None, include_eval=True, extras=None):
    """
    Create a minimal experiment directory with expected structure.

    Returns (experiment_dir, output_dir_base) as strings.
    """
    if run_names is None:
        run_names = ["run_rank4", "run_rank8"]

    exp_name = "test_experiment_2026-03-23"
    # Under the unified ck-projects/ layout, experiment dir and output base
    # point to the same location — outputs nest inside each run's dir.
    exp_dir = tmp_path / "ck-projects" / "capitalization" / exp_name
    exp_dir.mkdir(parents=True)
    out_base = exp_dir

    # experiment_summary.yaml
    config = {
        "experiment": {"name": exp_name, "directory": str(exp_dir)},
        "output": {"base_directory": str(out_base)},
        "runs": [
            {"name": rn, "type": "fine-tuned", "model": "test", "parameters": {}}
            for rn in run_names
        ],
        "evaluation": {
            "matrix": [
                {"run": rn, "tasks": ["test_task"], "epochs": [0]} for rn in run_names
            ]
        },
    }
    (exp_dir / "experiment_summary.yaml").write_text(yaml.dump(config))

    # logs/
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / "design-experiment.log").write_text("[log] designed")
    (logs_dir / "run-torchtune.log").write_text("[log] trained")

    # summary.md
    (exp_dir / "summary.md").write_text("# Summary\nResults look good.")

    # Per-run directories
    for rn in run_names:
        run_dir = exp_dir / rn
        run_dir.mkdir()
        (run_dir / "finetune.yaml").write_text("config: test")
        (run_dir / "finetune.slurm").write_text("#!/bin/bash")
        (run_dir / "setup_finetune.yaml").write_text("setup: test")

        if include_eval:
            eval_logs = run_dir / "eval" / "logs"
            eval_logs.mkdir(parents=True)
            (eval_logs / "test_task_epoch0.eval").write_text('{"results": {}}')

        # Output directory with fake checkpoint
        ck_out = out_base / f"ck-out-{rn}"
        ck_out.mkdir(parents=True)
        epoch_dir = ck_out / "epoch_0"
        epoch_dir.mkdir()
        # Write a fake checkpoint (larger than metadata to test size reporting)
        (epoch_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4096)
        (ck_out / "gpu_metrics.csv").write_text("timestamp,gpu_util\n")

    # Optional extras
    if extras:
        if "analysis" in extras:
            analysis = exp_dir / "analysis"
            analysis.mkdir()
            (analysis / "report.md").write_text("# Analysis Report")
            (analysis / "plot.html").write_text("<html>plot</html>")
        if "findings" in extras:
            (exp_dir / "findings.md").write_text("# Findings\nWe learned stuff.")

    return str(exp_dir), str(out_base)


# --- inventory_experiment tests ---


def test_inventory_complete_experiment(tmp_path):
    """Complete experiment → all files categorized correctly."""
    exp_dir, out_base = _make_experiment(tmp_path)
    result = inventory_experiment(exp_dir, out_base)

    assert result["status"] == "success"
    assert len(result["runs"]) == 2
    assert result["incomplete_runs"] == []
    assert result["keep_total_bytes"] > 0
    assert result["delete_total_bytes"] > 0

    archive_paths = [kf["archive_path"] for kf in result["keep_files"]]
    assert "experiment_summary.yaml" in archive_paths
    assert "summary.md" in archive_paths
    assert "logs/design-experiment.log" in archive_paths
    # Entire experiment dir is kept, so eval logs use their original paths
    assert "run_rank4/eval/logs/test_task_epoch0.eval" in archive_paths
    # Configs and SLURM scripts are also kept
    assert "run_rank4/finetune.yaml" in archive_paths
    assert "run_rank4/finetune.slurm" in archive_paths


def test_inventory_incomplete_experiment(tmp_path):
    """Missing eval logs → incomplete_runs populated."""
    exp_dir, out_base = _make_experiment(tmp_path, include_eval=False)
    result = inventory_experiment(exp_dir, out_base)

    assert result["status"] == "success"
    assert set(result["incomplete_runs"]) == {"run_rank4", "run_rank8"}


def test_inventory_missing_summary(tmp_path):
    """No experiment_summary.yaml → error."""
    exp_dir = tmp_path / "empty_experiment"
    exp_dir.mkdir()
    result = inventory_experiment(str(exp_dir), str(tmp_path))

    assert result["status"] == "error"
    assert "experiment_summary.yaml" in result["message"]


def test_inventory_with_analysis(tmp_path):
    """Analysis directory → included in keep files."""
    exp_dir, out_base = _make_experiment(tmp_path, extras=["analysis"])
    result = inventory_experiment(exp_dir, out_base)

    archive_paths = [kf["archive_path"] for kf in result["keep_files"]]
    assert "analysis/report.md" in archive_paths
    assert "analysis/plot.html" in archive_paths


# --- findings.md resolution tests ---


def test_findings_from_explicit_file(tmp_path):
    """findings.md exists → use it directly."""
    exp_dir, out_base = _make_experiment(tmp_path, extras=["findings"])
    result = inventory_experiment(exp_dir, out_base)

    assert result["findings_source"] == str(
        tmp_path
        / "ck-projects"
        / "capitalization"
        / "test_experiment_2026-03-23"
        / "findings.md"
    )


def test_findings_from_report(tmp_path):
    """No findings.md, but analysis/report.md exists → use report."""
    exp_dir, out_base = _make_experiment(tmp_path, extras=["analysis"])
    result = inventory_experiment(exp_dir, out_base)

    assert result["findings_source"].endswith("analysis/report.md")


def test_findings_from_summary(tmp_path):
    """No findings.md or report.md → fall back to summary.md."""
    exp_dir, out_base = _make_experiment(tmp_path)
    result = inventory_experiment(exp_dir, out_base)

    assert result["findings_source"].endswith("summary.md")


def test_findings_none_when_nothing_exists(tmp_path):
    """No findings, report, or summary → None."""
    exp_dir, out_base = _make_experiment(tmp_path)
    # Remove summary.md
    (
        tmp_path
        / "ck-projects"
        / "capitalization"
        / "test_experiment_2026-03-23"
        / "summary.md"
    ).unlink()
    result = inventory_experiment(exp_dir, out_base)

    assert result["findings_source"] is None


# --- create_archive tests ---


def test_create_archive(tmp_path):
    """All KEEP files land in correct archive structure."""
    exp_dir, out_base = _make_experiment(tmp_path)
    inventory = inventory_experiment(exp_dir, out_base)
    archive_dir = str(tmp_path / "ck-archive" / "test_experiment")

    result = create_archive(archive_dir, inventory)

    assert result["status"] == "success"
    assert len(result["errors"]) == 0
    assert "experiment_summary.yaml" in result["copied"]
    assert (
        tmp_path / "ck-archive" / "test_experiment" / "experiment_summary.yaml"
    ).exists()
    # Eval logs preserved at original paths
    assert (
        tmp_path
        / "ck-archive"
        / "test_experiment"
        / "run_rank4"
        / "eval"
        / "logs"
        / "test_task_epoch0.eval"
    ).exists()
    # Configs preserved too
    assert (
        tmp_path / "ck-archive" / "test_experiment" / "run_rank4" / "finetune.yaml"
    ).exists()


def test_create_archive_already_exists(tmp_path):
    """Archive dir already exists → error."""
    exp_dir, out_base = _make_experiment(tmp_path)
    inventory = inventory_experiment(exp_dir, out_base)
    archive_dir = tmp_path / "ck-archive" / "test_experiment"
    archive_dir.mkdir(parents=True)

    result = create_archive(str(archive_dir), inventory)

    assert result["status"] == "error"
    assert "already exists" in result["message"]


# --- verify_archive tests ---


def test_verify_archive(tmp_path):
    """Happy path verification succeeds."""
    exp_dir, out_base = _make_experiment(tmp_path)
    inventory = inventory_experiment(exp_dir, out_base)
    archive_dir = str(tmp_path / "ck-archive" / "test_experiment")
    create_archive(archive_dir, inventory)

    result = verify_archive(archive_dir, inventory)

    assert result["status"] == "success"
    assert result["missing"] == []
    assert result["size_mismatches"] == []


def test_verify_archive_missing_file(tmp_path):
    """Missing file in archive → error."""
    exp_dir, out_base = _make_experiment(tmp_path)
    inventory = inventory_experiment(exp_dir, out_base)
    archive_dir = tmp_path / "ck-archive" / "test_experiment"
    create_archive(str(archive_dir), inventory)

    # Delete one file from archive
    (archive_dir / "experiment_summary.yaml").unlink()

    result = verify_archive(str(archive_dir), inventory)

    assert result["status"] == "error"
    assert "experiment_summary.yaml" in result["missing"]


# --- delete_originals tests ---


def test_delete_originals(tmp_path):
    """Cleanup removes experiment dir and checkpoint dirs."""
    exp_dir, out_base = _make_experiment(tmp_path)
    inventory = inventory_experiment(exp_dir, out_base)
    run_names = inventory["runs"]

    result = delete_originals(exp_dir, out_base, run_names)

    assert result["status"] == "success"
    assert result["freed_bytes"] > 0
    from pathlib import Path

    assert not Path(exp_dir).exists()
    for rn in run_names:
        assert not (Path(out_base) / f"ck-out-{rn}").exists()


# --- dry run tests ---


def test_dry_run_no_side_effects(tmp_path):
    """Dry run reports plan without creating or deleting anything."""
    exp_dir, out_base = _make_experiment(tmp_path)
    archive_base = str(tmp_path / "ck-archive")

    result = archive_experiment(exp_dir, archive_base, dry_run=True)

    assert result["status"] == "success"
    assert result["mode"] == "dry-run"
    assert result["keep"]["files"] > 0
    assert result["delete"]["checkpoint_dirs"] == 2
    assert result["delete"]["size_mb"] >= 0

    # Nothing created or deleted
    from pathlib import Path

    assert not (tmp_path / "ck-archive").exists()
    assert Path(exp_dir).exists()


# --- full archive tests ---


def test_archive_full_workflow(tmp_path):
    """End-to-end: inventory → archive → verify → delete."""
    exp_dir, out_base = _make_experiment(tmp_path, extras=["analysis"])
    archive_base = str(tmp_path / "ck-archive")

    result = archive_experiment(exp_dir, archive_base)

    assert result["status"] == "success"
    assert result["mode"] == "archive"
    assert result["kept"]["files"] > 0
    assert result["freed"]["size_mb"] >= 0

    from pathlib import Path

    # Archive exists with expected files
    archive_dir = Path(result["archive_dir"])
    assert (archive_dir / "experiment_summary.yaml").exists()
    assert (archive_dir / "analysis" / "report.md").exists()
    # Configs preserved in archive
    assert (archive_dir / "run_rank4" / "finetune.yaml").exists()

    # Originals gone
    assert not Path(exp_dir).exists()


def test_archive_incomplete_without_force(tmp_path):
    """Incomplete experiment without --force → error."""
    exp_dir, out_base = _make_experiment(tmp_path, include_eval=False)
    archive_base = str(tmp_path / "ck-archive")

    result = archive_experiment(exp_dir, archive_base)

    assert result["status"] == "error"
    assert "Incomplete" in result["message"]
    assert len(result["incomplete_runs"]) == 2


def test_archive_incomplete_with_force(tmp_path):
    """Incomplete experiment with --force → proceeds with warning."""
    exp_dir, out_base = _make_experiment(tmp_path, include_eval=False)
    archive_base = str(tmp_path / "ck-archive")

    result = archive_experiment(exp_dir, archive_base, force=True)

    assert result["status"] == "success"
    assert len(result["incomplete_runs"]) == 2


def test_archive_nonexistent_dir(tmp_path):
    """Nonexistent experiment dir → error."""
    result = archive_experiment(str(tmp_path / "nope"), str(tmp_path / "archive"))

    assert result["status"] == "error"
    assert "not found" in result["message"]

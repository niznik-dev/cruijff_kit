"""Smoke test: verify setup_finetune.py and setup_inspect.py parse correctly.

Runs both scaffolding tools against minimal configs with fake paths,
checking that they produce the expected output files without errors.

Usage:
    python tests/integration/gpu/smoke_test_scaffolding.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent.parent
SETUP_FINETUNE = REPO_ROOT / "src" / "tools" / "torchtune" / "setup_finetune.py"
SETUP_INSPECT = REPO_ROOT / "src" / "tools" / "inspect" / "setup_inspect.py"


def test_setup_finetune(work_directory: Path):
    """Run setup_finetune.py with minimal args, verify outputs."""
    result = subprocess.run(
        [
            sys.executable,
            str(SETUP_FINETUNE),
            "--torchtune_model_name",
            "Llama-3.2-1B-Instruct",
            "--dataset_label",
            "test_dataset",
            "--dataset_ext",
            ".json",
            "--dataset_type",
            "chat_completion",
            "--input_dir_base",
            "/fake/input/",
            "--project_directory",
            "/fake/output/",
            "--experiment_name",
            "smoke_test",
            "--models_directory",
            "/fake/models/",
            "--my_wandb_run_name",
            "smoke_test",
        ],
        cwd=work_directory,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"setup_finetune.py exited with code {result.returncode}")

    slurm = work_directory / "finetune.slurm"
    yaml_out = work_directory / "finetune.yaml"
    assert slurm.exists(), "finetune.slurm not generated"
    assert yaml_out.exists(), "finetune.yaml not generated"
    print("PASS: setup_finetune.py scaffolding")


def test_setup_inspect(work_directory: Path):
    """Run setup_inspect.py with a minimal eval_config, verify outputs."""
    config = {
        "task_script": "blueprints/capitalization/inspect_task.py@capitalization",
        "task_name": "capitalization",
        "model_path": "/fake/checkpoint/epoch_0",
        "model_hf_name": "hf/smoke_test_epoch_0",
        "output_dir": "/fake/output/",
        "data_path": "tests/fixtures/data/words_5L_80P_10.json",
        "vis_label": "smoke_test",
        "use_chat_template": True,
        "epoch": 0,
        "is_finetuned": True,
        "source_model": "Llama-3.2-1B-Instruct",
    }
    config_path = work_directory / "eval_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = subprocess.run(
        [
            sys.executable,
            str(SETUP_INSPECT),
            "--config",
            str(config_path),
            "--model_name",
            "Llama-3.2-1B-Instruct",
        ],
        cwd=work_directory,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"setup_inspect.py exited with code {result.returncode}")

    slurm = work_directory / "cell.slurm"
    assert slurm.exists(), "cell.slurm not generated"
    print("PASS: setup_inspect.py scaffolding")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        finetune_directory = Path(tmpdir) / "finetune"
        finetune_directory.mkdir()
        test_setup_finetune(finetune_directory)

        inspect_directory = Path(tmpdir) / "inspect"
        inspect_directory.mkdir()
        test_setup_inspect(inspect_directory)


if __name__ == "__main__":
    main()

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


def test_setup_finetune(work_dir: Path):
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
            "--input_dir_base",
            "/fake/input/",
            "--output_dir_base",
            "/fake/output/",
            "--models_dir",
            "/fake/models/",
            "--my_wandb_run_name",
            "smoke_test",
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"setup_finetune.py exited with code {result.returncode}")

    slurm = work_dir / "finetune.slurm"
    yaml_out = work_dir / "finetune.yaml"
    assert slurm.exists(), "finetune.slurm not generated"
    assert yaml_out.exists(), "finetune.yaml not generated"
    print("PASS: setup_finetune.py scaffolding")


def test_setup_inspect(work_dir: Path):
    """Run setup_inspect.py with a minimal eval_config, verify outputs."""
    config = {
        "task_script": "projects/capitalization/inspect_task.py@capitalization",
        "task_name": "capitalization",
        "model_path": "/fake/checkpoint/epoch_0",
        "model_hf_name": "hf/smoke_test_epoch_0",
        "output_dir": "/fake/output/",
        "data_path": "data/green/capitalization/words_5L_80P_1000.json",
        "vis_label": "smoke_test",
        "use_chat_template": True,
        "epoch": 0,
        "finetuned": True,
        "source_model": "Llama-3.2-1B-Instruct",
    }
    config_path = work_dir / "eval_config.yaml"
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
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"setup_inspect.py exited with code {result.returncode}")

    slurm = work_dir / "capitalization_epoch0.slurm"
    assert slurm.exists(), "capitalization_epoch0.slurm not generated"
    print("PASS: setup_inspect.py scaffolding")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        finetune_dir = Path(tmpdir) / "finetune"
        finetune_dir.mkdir()
        test_setup_finetune(finetune_dir)

        inspect_dir = Path(tmpdir) / "inspect"
        inspect_dir.mkdir()
        test_setup_inspect(inspect_dir)


if __name__ == "__main__":
    main()

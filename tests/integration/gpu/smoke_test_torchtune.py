"""Smoke test: scaffold a config, run torchtune for ~10 steps, verify checkpoint.

Catches torchtune dependency breakage, custom recipe issues, checkpoint
writing failures, and dataset loading problems. Does NOT test whether the
model learns anything — just that the pipeline doesn't crash.

Usage:
    python tests/integration/gpu/smoke_test_torchtune.py

Environment variables:
    CK_MODELS_DIR: Path to pretrained models (default: /scratch/gpfs/MSALGANIK/pretrained-llms)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent
SETUP_FINETUNE = REPO_ROOT / "tools" / "torchtune" / "setup_finetune.py"
DATA_PATH = REPO_ROOT / "data" / "green" / "capitalization" / "words_5L_80P_1000.json"
MODELS_DIR = os.environ.get("CK_MODELS_DIR", "/scratch/gpfs/MSALGANIK/pretrained-llms")
MODEL_NAME = "Llama-3.2-1B-Instruct"
MAX_STEPS = 10


def run_command(cmd, cwd=None, label="command"):
    """Run a command and raise on failure."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"{label} exited with code {result.returncode}")
    return result


def main():
    with tempfile.TemporaryDirectory(prefix="ck-torchtune-smoke-") as tmpdir:
        work_dir = Path(tmpdir)
        output_dir = work_dir / "output"
        output_dir.mkdir()

        # Step 1: Scaffold config
        print("Scaffolding torchtune config...")
        run_command(
            [
                sys.executable,
                str(SETUP_FINETUNE),
                "--torchtune_model_name",
                MODEL_NAME,
                "--dataset_label",
                "words_5L_80P_1000",
                "--dataset_ext",
                ".json",
                "--input_dir_base",
                str(REPO_ROOT / "data" / "green" / "capitalization"),
                "--output_dir_base",
                str(output_dir),
                "--models_dir",
                MODELS_DIR,
                "--my_wandb_run_name",
                "torchtune_smoke_test",
                "--batch_size",
                "4",
                "--epochs",
                "1",
                "--max_steps_per_epoch",
                str(MAX_STEPS),
                "--lora_rank",
                "4",
                "--prompt",
                "Capitalize the given word: {input}\n",
                "--system_prompt",
                "You are a helpful assistant.",
            ],
            cwd=work_dir,
            label="setup_finetune.py",
        )

        finetune_yaml = work_dir / "finetune.yaml"
        assert finetune_yaml.exists(), "finetune.yaml not generated"
        print("PASS: scaffolding")

        # Step 2: Run torchtune (~10 steps)
        print(f"Running torchtune for {MAX_STEPS} steps...")
        run_command(
            ["tune", "run", "lora_finetune_single_device", "--config", "finetune.yaml"],
            cwd=work_dir,
            label="tune run",
        )
        print("PASS: torchtune training")

        # Step 3: Verify checkpoint exists and is non-empty
        # setup_finetune.py creates output under output_dir_base/ck-out-{run_name}/
        ck_out_dirs = list(output_dir.glob("ck-out-*"))
        assert len(ck_out_dirs) > 0, f"No ck-out-* directory found in {output_dir}"

        checkpoint_dirs = list(ck_out_dirs[0].glob("epoch_*"))
        assert len(checkpoint_dirs) > 0, (
            f"No epoch_* checkpoint directory found in {ck_out_dirs[0]}"
        )

        # Check for actual checkpoint files (safetensors or bin)
        checkpoint_files = list(checkpoint_dirs[0].glob("*.safetensors")) + list(
            checkpoint_dirs[0].glob("*.bin")
        )
        assert len(checkpoint_files) > 0, (
            f"No checkpoint files found in {checkpoint_dirs[0]}"
        )

        for ckpt in checkpoint_files:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            print(f"  Checkpoint: {ckpt.name} ({size_mb:.1f} MB)")
            assert ckpt.stat().st_size > 0, f"Checkpoint file is empty: {ckpt}"

        print("PASS: checkpoint verification")
        print(f"\nAll torchtune smoke tests passed ({MAX_STEPS} training steps)")


if __name__ == "__main__":
    main()

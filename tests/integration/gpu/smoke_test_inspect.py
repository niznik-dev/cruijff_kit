"""Smoke test: run inspect-ai eval on the base 1B model.

Catches inspect-ai dependency breakage, scorer issues, HF backend
problems, and log generation failures. Completely independent of
torchtune — uses the base model with no fine-tuning.

Usage:
    python tests/integration/gpu/smoke_test_inspect.py

Environment variables:
    CK_MODELS_DIR: Path to pretrained models (default: /scratch/gpfs/MSALGANIK/pretrained-llms)
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent
TASK_SCRIPT = (
    REPO_ROOT / "experiments" / "capitalization" / "inspect_task_capitalization.py"
)
DATA_PATH = REPO_ROOT / "data" / "green" / "capitalization" / "words_5L_80P_1000.json"
MODELS_DIR = os.environ.get("CK_MODELS_DIR", "/scratch/gpfs/MSALGANIK/pretrained-llms")
MODEL_NAME = "Llama-3.2-1B-Instruct"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
SAMPLE_LIMIT = 20


def main():
    with tempfile.TemporaryDirectory(prefix="ck-inspect-smoke-") as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        log_dir.mkdir()

        # Run inspect eval on base model, small sample
        print(f"Running inspect eval ({SAMPLE_LIMIT} samples, base {MODEL_NAME})...")
        cmd = [
            sys.executable,
            "-m",
            "inspect_ai._cli.main",
            "eval",
            f"{TASK_SCRIPT}@capitalization",
            "--model",
            f"hf/{MODEL_NAME}",
            "-M",
            f"model_path={MODEL_PATH}",
            "-T",
            f"data_path={DATA_PATH}",
            "-T",
            "use_chat_template=true",
            "--limit",
            str(SAMPLE_LIMIT),
            "--log-dir",
            str(log_dir),
            "--log-level",
            "info",
        ]

        # HF_HUB_OFFLINE prevents network lookups on compute nodes
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            raise RuntimeError(f"inspect eval exited with code {result.returncode}")
        print("PASS: inspect eval completed")

        # Verify log file exists
        eval_logs = list(log_dir.rglob("*.eval"))
        assert len(eval_logs) > 0, f"No .eval log files found in {log_dir}"
        print(f"  Log file: {eval_logs[0].name}")

        # Verify log file is non-trivial (> 1KB)
        log_size = eval_logs[0].stat().st_size
        assert log_size > 1024, f"Log file suspiciously small: {log_size} bytes"
        print(f"  Log size: {log_size / 1024:.1f} KB")

        # Check for Python tracebacks in output (sign of partial failure)
        combined_output = result.stdout + result.stderr
        if "Traceback (most recent call last)" in combined_output:
            print("WARNING: Python traceback detected in output:")
            print(combined_output)
            raise RuntimeError("inspect eval produced a traceback")

        print(f"\nAll inspect smoke tests passed ({SAMPLE_LIMIT} samples evaluated)")


if __name__ == "__main__":
    main()

"""Integration tests: scaffold tools produce valid output from fixture inputs.

Tests setup_finetune.py and setup_inspect.py using fixture YAML files with
path placeholders resolved to tmp_path. Follows patterns from
test_setup_finetune_main.py and test_setup_inspect.py.

No GPU or SLURM required — runs on CPU.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from cruijff_kit.tools.torchtune.setup_finetune import main as finetune_main
from cruijff_kit.tools.inspect.setup_inspect import main as inspect_main

# Import the fixture resolver
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "fixtures"))
from conftest import FIXTURES_DIR, resolve_placeholders


def _resolve_fixture_to_path(fixture_rel: str, dest: Path, scratch: Path, repo: Path):
    """Copy a fixture file to dest with placeholders resolved."""
    src = FIXTURES_DIR / fixture_rel
    content = src.read_text()
    resolved = resolve_placeholders(content, scratch, repo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resolved)
    return dest


# ---------------------------------------------------------------------------
# setup_finetune tests
# ---------------------------------------------------------------------------


class TestScaffoldFinetune:
    """Test setup_finetune.py with fixture setup_finetune.yaml files."""

    @pytest.fixture(params=["rank4", "rank8"])
    def finetune_env(self, request, tmp_path, monkeypatch):
        """Set up environment for a finetune scaffold test."""
        rank = request.param
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        repo = tmp_path / "repo"
        repo.mkdir()

        run_dir = tmp_path / rank
        run_dir.mkdir()

        config_path = run_dir / "setup_finetune.yaml"
        _resolve_fixture_to_path(
            f"scaffold/torchtune/{rank}/setup_finetune.yaml",
            config_path,
            scratch,
            repo,
        )

        monkeypatch.chdir(run_dir)
        monkeypatch.setenv("USER", "testuser")
        return run_dir, config_path, rank

    def test_generates_finetune_yaml(self, finetune_env):
        run_dir, config_path, _ = finetune_env
        with patch.object(
            sys, "argv", ["setup_finetune.py", "--config_file", str(config_path)]
        ):
            finetune_main()
        assert (run_dir / "finetune.yaml").exists()

    def test_generates_slurm_script(self, finetune_env):
        run_dir, config_path, _ = finetune_env
        with patch.object(
            sys, "argv", ["setup_finetune.py", "--config_file", str(config_path)]
        ):
            finetune_main()
        assert (run_dir / "finetune.slurm").exists()

    def test_lora_rank_matches_fixture(self, finetune_env):
        run_dir, config_path, rank = finetune_env
        expected_rank = int(rank.replace("rank", ""))

        with patch.object(
            sys, "argv", ["setup_finetune.py", "--config_file", str(config_path)]
        ):
            finetune_main()

        with open(run_dir / "finetune.yaml") as f:
            config = yaml.safe_load(f)
        assert config["model"]["lora_rank"] == expected_rank

    def test_lora_alpha_is_double_rank(self, finetune_env):
        run_dir, config_path, rank = finetune_env
        expected_rank = int(rank.replace("rank", ""))

        with patch.object(
            sys, "argv", ["setup_finetune.py", "--config_file", str(config_path)]
        ):
            finetune_main()

        with open(run_dir / "finetune.yaml") as f:
            config = yaml.safe_load(f)
        assert config["model"]["lora_alpha"] == expected_rank * 2

    def test_output_dir_uses_resolved_paths(self, finetune_env):
        run_dir, config_path, _ = finetune_env
        with patch.object(
            sys, "argv", ["setup_finetune.py", "--config_file", str(config_path)]
        ):
            finetune_main()

        with open(run_dir / "finetune.yaml") as f:
            config = yaml.safe_load(f)
        assert "__SCRATCH__" not in config["output_dir"]
        assert "ck-out-" in config["output_dir"]

    def test_batch_size_matches(self, finetune_env):
        run_dir, config_path, _ = finetune_env
        with patch.object(
            sys, "argv", ["setup_finetune.py", "--config_file", str(config_path)]
        ):
            finetune_main()

        with open(run_dir / "finetune.yaml") as f:
            config = yaml.safe_load(f)
        assert config["batch_size"] == 4

    def test_slurm_has_conda_activate(self, finetune_env):
        run_dir, config_path, _ = finetune_env
        with patch.object(
            sys, "argv", ["setup_finetune.py", "--config_file", str(config_path)]
        ):
            finetune_main()

        slurm = (run_dir / "finetune.slurm").read_text()
        assert "conda activate cruijff" in slurm

    def test_slurm_uses_single_device(self, finetune_env):
        run_dir, config_path, _ = finetune_env
        with patch.object(
            sys, "argv", ["setup_finetune.py", "--config_file", str(config_path)]
        ):
            finetune_main()

        slurm = (run_dir / "finetune.slurm").read_text()
        assert "lora_finetune_single_device" in slurm


# ---------------------------------------------------------------------------
# setup_inspect tests
# ---------------------------------------------------------------------------


class TestScaffoldInspect:
    """Test setup_inspect.py with fixture eval_config.yaml files."""

    @pytest.fixture(params=["rank4", "rank8"])
    def inspect_env(self, request, tmp_path, monkeypatch):
        """Set up environment for an inspect scaffold test."""
        rank = request.param
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        repo = tmp_path / "repo"
        repo.mkdir()

        eval_dir = tmp_path / rank / "eval"
        eval_dir.mkdir(parents=True)

        config_path = eval_dir / "eval_config.yaml"
        _resolve_fixture_to_path(
            f"scaffold/inspect/{rank}/eval/eval_config.yaml",
            config_path,
            scratch,
            repo,
        )

        monkeypatch.chdir(eval_dir)
        return eval_dir, config_path, rank

    def test_generates_slurm_script(self, inspect_env, monkeypatch):
        eval_dir, config_path, _ = inspect_env
        monkeypatch.setattr(
            "sys.argv",
            [
                "setup_inspect.py",
                "--config",
                str(config_path),
                "--model_name",
                "Llama-3.2-1B-Instruct",
            ],
        )
        inspect_main()

        output = eval_dir / "capitalization_epoch0.slurm"
        assert output.exists()

    def test_slurm_contains_inspect_eval(self, inspect_env, monkeypatch):
        eval_dir, config_path, _ = inspect_env
        monkeypatch.setattr(
            "sys.argv",
            [
                "setup_inspect.py",
                "--config",
                str(config_path),
                "--model_name",
                "Llama-3.2-1B-Instruct",
            ],
        )
        inspect_main()

        slurm = (eval_dir / "capitalization_epoch0.slurm").read_text()
        assert "inspect eval" in slurm

    def test_slurm_has_correct_model(self, inspect_env, monkeypatch):
        eval_dir, config_path, rank = inspect_env
        monkeypatch.setattr(
            "sys.argv",
            [
                "setup_inspect.py",
                "--config",
                str(config_path),
                "--model_name",
                "Llama-3.2-1B-Instruct",
            ],
        )
        inspect_main()

        slurm = (eval_dir / "capitalization_epoch0.slurm").read_text()
        assert f"hf/Llama-3.2-1B-Instruct_{rank}_epoch_0" in slurm

    def test_slurm_has_vis_label(self, inspect_env, monkeypatch):
        eval_dir, config_path, rank = inspect_env
        monkeypatch.setattr(
            "sys.argv",
            [
                "setup_inspect.py",
                "--config",
                str(config_path),
                "--model_name",
                "Llama-3.2-1B-Instruct",
            ],
        )
        inspect_main()

        slurm = (eval_dir / "capitalization_epoch0.slurm").read_text()
        assert f'vis_label="{rank}"' in slurm

    def test_slurm_has_metadata_epoch(self, inspect_env, monkeypatch):
        eval_dir, config_path, _ = inspect_env
        monkeypatch.setattr(
            "sys.argv",
            [
                "setup_inspect.py",
                "--config",
                str(config_path),
                "--model_name",
                "Llama-3.2-1B-Instruct",
            ],
        )
        inspect_main()

        slurm = (eval_dir / "capitalization_epoch0.slurm").read_text()
        assert '--metadata epoch="0"' in slurm

    def test_no_unresolved_placeholders(self, inspect_env, monkeypatch):
        eval_dir, config_path, _ = inspect_env
        monkeypatch.setattr(
            "sys.argv",
            [
                "setup_inspect.py",
                "--config",
                str(config_path),
                "--model_name",
                "Llama-3.2-1B-Instruct",
            ],
        )
        inspect_main()

        slurm = (eval_dir / "capitalization_epoch0.slurm").read_text()
        assert "__SCRATCH__" not in slurm
        assert "__REPO__" not in slurm

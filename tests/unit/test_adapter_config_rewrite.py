"""Unit tests for adapter_config.json base-path rewrite + uncruijff round-trip."""

import io
import json
import logging
from pathlib import Path


from cruijff_kit.tools.torchtune.custom_recipes.custom_recipe_utils import (
    rewrite_adapter_config_base_path,
)
from cruijff_kit.tools.torchtune import uncruijff_adapter


def _write_epoch(tmp_path: Path, epoch: int, *, with_repo_id: bool = True) -> Path:
    epoch_dir = tmp_path / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "lora_alpha": 16,
                "peft_type": "LORA",
                "base_model_name_or_path": "meta-llama/Llama-3.2-1B-Instruct",
            }
        )
    )
    if with_repo_id:
        (epoch_dir / "original_repo_id.json").write_text(
            json.dumps({"repo_id": "meta-llama/Llama-3.2-1B-Instruct"})
        )
    return epoch_dir


def _read_base(epoch_dir: Path) -> str:
    return json.loads((epoch_dir / "adapter_config.json").read_text())[
        "base_model_name_or_path"
    ]


def _silent_logger() -> logging.Logger:
    logger = logging.getLogger("test_adapter_rewrite")
    logger.addHandler(logging.StreamHandler(io.StringIO()))
    return logger


def test_rewrite_replaces_hub_name_with_local_abs_path(tmp_path):
    _write_epoch(tmp_path, 0)
    base = "/scratch/pretrained-llms/Llama-3.2-1B-Instruct/"

    rewrite_adapter_config_base_path(str(tmp_path), 0, base, _silent_logger())

    assert _read_base(tmp_path / "epoch_0") == base.rstrip("/")


def test_rewrite_is_idempotent(tmp_path):
    _write_epoch(tmp_path, 0)
    base = "/scratch/pretrained-llms/Llama-3.2-1B-Instruct"

    rewrite_adapter_config_base_path(str(tmp_path), 0, base, _silent_logger())
    rewrite_adapter_config_base_path(str(tmp_path), 0, base, _silent_logger())

    assert _read_base(tmp_path / "epoch_0") == base


def test_rewrite_skips_when_no_adapter_config(tmp_path, caplog):
    (tmp_path / "epoch_0").mkdir()
    with caplog.at_level(logging.WARNING):
        rewrite_adapter_config_base_path(str(tmp_path), 0, "/some/path")
    assert any("not found" in r.message for r in caplog.records)


def test_uncruijff_restores_hub_name_single_epoch_dir(tmp_path):
    epoch_dir = _write_epoch(tmp_path, 0)
    # Simulate a cruijff-rewritten config first
    rewrite_adapter_config_base_path(
        str(tmp_path), 0, "/local/abs/path", _silent_logger()
    )
    assert _read_base(epoch_dir) == "/local/abs/path"

    uncruijff_adapter.restore_one(epoch_dir)

    assert _read_base(epoch_dir) == "meta-llama/Llama-3.2-1B-Instruct"


def test_uncruijff_skips_when_no_original_repo_id_and_no_override(tmp_path, capsys):
    epoch_dir = _write_epoch(tmp_path, 0, with_repo_id=False)
    rewrite_adapter_config_base_path(
        str(tmp_path), 0, "/local/abs/path", _silent_logger()
    )

    changed = uncruijff_adapter.restore_one(epoch_dir)

    assert changed is False
    assert _read_base(epoch_dir) == "/local/abs/path"


def test_uncruijff_uses_repo_id_override_when_marker_missing(tmp_path):
    epoch_dir = _write_epoch(tmp_path, 0, with_repo_id=False)
    rewrite_adapter_config_base_path(
        str(tmp_path), 0, "/local/abs/path", _silent_logger()
    )

    changed = uncruijff_adapter.restore_one(
        epoch_dir, repo_id_override="meta-llama/Llama-3.2-1B-Instruct"
    )

    assert changed is True
    assert _read_base(epoch_dir) == "meta-llama/Llama-3.2-1B-Instruct"


def test_uncruijff_repo_id_override_wins_over_marker(tmp_path):
    epoch_dir = _write_epoch(tmp_path, 0, with_repo_id=True)
    rewrite_adapter_config_base_path(
        str(tmp_path), 0, "/local/abs/path", _silent_logger()
    )

    uncruijff_adapter.restore_one(epoch_dir, repo_id_override="my-org/forked-model")

    assert _read_base(epoch_dir) == "my-org/forked-model"


def test_uncruijff_recurses_over_run_dir(tmp_path, monkeypatch, capsys):
    _write_epoch(tmp_path, 0)
    _write_epoch(tmp_path, 1)
    rewrite_adapter_config_base_path(str(tmp_path), 0, "/local/p", _silent_logger())
    rewrite_adapter_config_base_path(str(tmp_path), 1, "/local/p", _silent_logger())

    monkeypatch.setattr("sys.argv", ["uncruijff_adapter", str(tmp_path)])
    exit_code = uncruijff_adapter.main()

    assert exit_code == 0
    assert _read_base(tmp_path / "epoch_0") == "meta-llama/Llama-3.2-1B-Instruct"
    assert _read_base(tmp_path / "epoch_1") == "meta-llama/Llama-3.2-1B-Instruct"


def test_uncruijff_errors_on_missing_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["uncruijff_adapter", str(tmp_path / "nope")])
    exit_code = uncruijff_adapter.main()
    assert exit_code == 1


def test_uncruijff_errors_when_no_epoch_dirs(tmp_path, monkeypatch, capsys):
    # Empty dir, no epoch_*, no adapter_config.json
    monkeypatch.setattr("sys.argv", ["uncruijff_adapter", str(tmp_path)])
    exit_code = uncruijff_adapter.main()
    assert exit_code == 1

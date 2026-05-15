"""Unit tests for adapter_config.json base-path rewrite + port_cruijff_adapter round-trip."""

import io
import json
import logging
from pathlib import Path


from cruijff_kit.tools.torchtune.custom_recipes.custom_recipe_utils import (
    check_adapter_base_path,
    rewrite_adapter_config_base_path,
    stash_adapter_files,
)
from cruijff_kit.tools.torchtune import port_cruijff_adapter


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


def test_port_cruijff_restores_hub_name_single_epoch_dir(tmp_path):
    epoch_dir = _write_epoch(tmp_path, 0)
    # Simulate a cruijff-rewritten config first
    rewrite_adapter_config_base_path(
        str(tmp_path), 0, "/local/abs/path", _silent_logger()
    )
    assert _read_base(epoch_dir) == "/local/abs/path"

    port_cruijff_adapter.restore_one(epoch_dir)

    assert _read_base(epoch_dir) == "meta-llama/Llama-3.2-1B-Instruct"


def test_port_cruijff_skips_when_no_original_repo_id_and_no_override(tmp_path, capsys):
    epoch_dir = _write_epoch(tmp_path, 0, with_repo_id=False)
    rewrite_adapter_config_base_path(
        str(tmp_path), 0, "/local/abs/path", _silent_logger()
    )

    changed = port_cruijff_adapter.restore_one(epoch_dir)

    assert changed is False
    assert _read_base(epoch_dir) == "/local/abs/path"


def test_port_cruijff_uses_repo_id_override_when_marker_missing(tmp_path):
    epoch_dir = _write_epoch(tmp_path, 0, with_repo_id=False)
    rewrite_adapter_config_base_path(
        str(tmp_path), 0, "/local/abs/path", _silent_logger()
    )

    changed = port_cruijff_adapter.restore_one(
        epoch_dir, repo_id_override="meta-llama/Llama-3.2-1B-Instruct"
    )

    assert changed is True
    assert _read_base(epoch_dir) == "meta-llama/Llama-3.2-1B-Instruct"


def test_port_cruijff_repo_id_override_wins_over_marker(tmp_path):
    epoch_dir = _write_epoch(tmp_path, 0, with_repo_id=True)
    rewrite_adapter_config_base_path(
        str(tmp_path), 0, "/local/abs/path", _silent_logger()
    )

    port_cruijff_adapter.restore_one(epoch_dir, repo_id_override="my-org/forked-model")

    assert _read_base(epoch_dir) == "my-org/forked-model"


def test_port_cruijff_recurses_over_run_dir(tmp_path, monkeypatch, capsys):
    _write_epoch(tmp_path, 0)
    _write_epoch(tmp_path, 1)
    rewrite_adapter_config_base_path(str(tmp_path), 0, "/local/p", _silent_logger())
    rewrite_adapter_config_base_path(str(tmp_path), 1, "/local/p", _silent_logger())

    monkeypatch.setattr("sys.argv", ["port_cruijff_adapter", str(tmp_path)])
    exit_code = port_cruijff_adapter.main()

    assert exit_code == 0
    assert _read_base(tmp_path / "epoch_0") == "meta-llama/Llama-3.2-1B-Instruct"
    assert _read_base(tmp_path / "epoch_1") == "meta-llama/Llama-3.2-1B-Instruct"


def test_port_cruijff_errors_on_missing_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["port_cruijff_adapter", str(tmp_path / "nope")])
    exit_code = port_cruijff_adapter.main()
    assert exit_code == 1


def test_port_cruijff_errors_when_no_epoch_dirs(tmp_path, monkeypatch, capsys):
    # Empty dir, no epoch_*, no adapter_config.json
    monkeypatch.setattr("sys.argv", ["port_cruijff_adapter", str(tmp_path)])
    exit_code = port_cruijff_adapter.main()
    assert exit_code == 1


# ---------- check_adapter_base_path ----------


def test_check_returns_none_when_base_path_exists(tmp_path):
    epoch_dir = _write_epoch(tmp_path, 0)
    fake_base = tmp_path / "fake_base"
    fake_base.mkdir()
    rewrite_adapter_config_base_path(str(tmp_path), 0, str(fake_base), _silent_logger())

    assert check_adapter_base_path(epoch_dir) is None


def test_check_flags_missing_local_base_path(tmp_path):
    epoch_dir = _write_epoch(tmp_path, 0)
    rewrite_adapter_config_base_path(
        str(tmp_path), 0, "/scratch/gone/base", _silent_logger()
    )

    problem = check_adapter_base_path(epoch_dir)
    assert problem is not None
    assert "STALE_LOCAL_BASE_PATH" in problem
    assert "/scratch/gone/base" in problem


def test_check_skips_hf_hub_style_names(tmp_path):
    """HF Hub repo names resolve via cache/hub, not filesystem — don't flag."""
    epoch_dir = _write_epoch(tmp_path, 0)
    # adapter_config.json from _write_epoch already has a hub-style name
    assert check_adapter_base_path(epoch_dir) is None


def test_check_returns_none_when_no_adapter_config(tmp_path):
    """Dirs without adapter_config.json (e.g. base/merged model) aren't our concern."""
    assert check_adapter_base_path(tmp_path) is None


# ---------- stash_adapter_files ----------


def _write_merged_save(tmp_path: Path, epoch: int) -> Path:
    """Simulate what torchtune writes when save_adapter_weights_only=False —
    merged model.safetensors + adapter files side-by-side."""
    epoch_dir = tmp_path / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "model.safetensors").write_text("fake-merged-weights")
    (epoch_dir / "adapter_config.json").write_text("{}")
    (epoch_dir / "adapter_model.safetensors").write_text("fake-adapter-weights")
    return epoch_dir


def test_stash_moves_adapter_files_into_subdir(tmp_path):
    epoch_dir = _write_merged_save(tmp_path, 0)
    stash_adapter_files(str(tmp_path), 0, _silent_logger())

    assert (epoch_dir / "model.safetensors").exists()  # merged stays
    assert not (epoch_dir / "adapter_config.json").exists()  # adapter moved
    assert not (epoch_dir / "adapter_model.safetensors").exists()
    assert (epoch_dir / "adapter_weights" / "adapter_config.json").exists()
    assert (epoch_dir / "adapter_weights" / "adapter_model.safetensors").exists()


def test_stash_skips_missing_files_gracefully(tmp_path):
    """Only some adapter files present — stash should move what exists, skip the rest."""
    epoch_dir = tmp_path / "epoch_0"
    epoch_dir.mkdir()
    (epoch_dir / "model.safetensors").write_text("merged")
    (epoch_dir / "adapter_config.json").write_text("{}")
    # adapter_model.safetensors deliberately absent

    stash_adapter_files(str(tmp_path), 0, _silent_logger())

    assert (epoch_dir / "adapter_weights" / "adapter_config.json").exists()
    assert not (epoch_dir / "adapter_weights" / "adapter_model.safetensors").exists()


def test_stash_warns_when_epoch_dir_missing(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        stash_adapter_files(str(tmp_path), 99)
    assert any("not found" in r.message for r in caplog.records)

"""Unit tests for post_finetune.read_run_config.

Regression coverage for the F2 wrapper move (PR #530): finetune.yaml uses
OmegaConf ${...} interpolation (e.g. checkpoint_dir: ${models_directory}/...). The
recipe-internal version resolved these via torchtune's OmegaConf load; an early
wrapper used yaml.safe_load and silently wrote a literal "${models_directory}" into
adapter_config.json. read_run_config must resolve interpolations.
"""

from pathlib import Path

from cruijff_kit.tools.torchtune.post_finetune import read_run_config


def _write_finetune_yaml(tmp_path: Path, **overrides) -> Path:
    """Write a minimal finetune.yaml that uses ${models_directory} interpolation."""
    cfg_path = tmp_path / "finetune.yaml"
    save_only = overrides.get("save_adapter_weights_only", True)
    cfg_path.write_text(
        "models_directory: /scratch/pretrained-llms\n"
        f"output_dir: {tmp_path}/artifacts/\n"
        "checkpointer:\n"
        "  checkpoint_dir: ${models_directory}/Llama-3.2-1B-Instruct/\n"
        f"save_adapter_weights_only: {str(save_only).lower()}\n"
    )
    return cfg_path


def test_resolves_models_dir_interpolation(tmp_path):
    cfg_path = _write_finetune_yaml(tmp_path)

    _output_directory, base_model_path, _save_only = read_run_config(cfg_path)

    # The interpolation must be resolved — not left as a literal "${models_directory}".
    assert "${" not in base_model_path
    assert base_model_path == "/scratch/pretrained-llms/Llama-3.2-1B-Instruct/"


def test_returns_output_dir_and_save_mode(tmp_path):
    cfg_path = _write_finetune_yaml(tmp_path, save_adapter_weights_only=False)

    output_dir, _base, save_only = read_run_config(cfg_path)

    assert output_dir == Path(f"{tmp_path}/artifacts/")
    assert save_only is False


def test_save_adapter_weights_only_defaults_false_when_absent(tmp_path):
    cfg_path = tmp_path / "finetune.yaml"
    cfg_path.write_text(
        "models_directory: /scratch/pretrained-llms\n"
        f"output_dir: {tmp_path}/artifacts/\n"
        "checkpointer:\n"
        "  checkpoint_dir: ${models_directory}/Llama-3.2-1B-Instruct/\n"
    )

    _output_directory, _base, save_only = read_run_config(cfg_path)

    assert save_only is False

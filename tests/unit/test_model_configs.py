"""Unit tests for tools/torchtune/model_configs.py

Run these tests when adding a new model to verify the configuration is correct:
    pytest tests/unit/test_model_configs.py -v

These tests verify:
- All models have required tokenizer configuration
- All tokenizer path_types are valid and handled
- configure_tokenizer() works correctly for all supported model families
"""

import pytest
from cruijff_kit.tools.torchtune.model_configs import (
    MODEL_CONFIGS,
    configure_tokenizer,
    VALID_TOKENIZER_PATH_TYPES,
)


# =============================================================================
# Tests for configure_tokenizer()
# =============================================================================

class TestConfigureTokenizerByFamily:
    """Test tokenizer configuration for each supported model family."""

    def test_llama_tokenizer(self):
        """Test Llama tokenizer configuration (SentencePiece)."""
        config = {"tokenizer": {}}
        model_config = {
            "tokenizer": {
                "component": "torchtune.models.llama3.llama3_tokenizer",
                "path_type": "llama",
            }
        }

        result = configure_tokenizer(config, model_config, "Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct")

        assert result["tokenizer"]["_component_"] == "torchtune.models.llama3.llama3_tokenizer"
        assert result["tokenizer"]["path"] == "${models_dir}/Llama-3.2-1B-Instruct/original/tokenizer.model"
        assert "merges_file" not in result["tokenizer"]

    def test_qwen_tokenizer(self):
        """Test Qwen tokenizer configuration (BPE with vocab.json + merges.txt)."""
        config = {"tokenizer": {}}
        model_config = {
            "tokenizer": {
                "component": "torchtune.models.qwen2_5.qwen2_5_tokenizer",
                "path_type": "qwen",
            }
        }

        result = configure_tokenizer(config, model_config, "Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct")

        assert result["tokenizer"]["_component_"] == "torchtune.models.qwen2_5.qwen2_5_tokenizer"
        assert result["tokenizer"]["path"] == "${models_dir}/Qwen2.5-3B-Instruct/vocab.json"
        assert result["tokenizer"]["merges_file"] == "${models_dir}/Qwen2.5-3B-Instruct/merges.txt"


class TestConfigureTokenizerErrors:
    """Test error handling in configure_tokenizer()."""

    def test_unknown_path_type_raises_error(self):
        """Test that unknown path_type raises ValueError."""
        config = {"tokenizer": {}}
        model_config = {
            "tokenizer": {
                "component": "torchtune.models.some.tokenizer",
                "path_type": "unknown_type",
            }
        }

        with pytest.raises(ValueError) as exc_info:
            configure_tokenizer(config, model_config, "SomeModel", "SomeModel")

        assert "Unknown tokenizer path_type" in str(exc_info.value)
        assert "unknown_type" in str(exc_info.value)
        assert "SomeModel" in str(exc_info.value)

    def test_missing_path_type_raises_error(self):
        """Test that missing path_type raises ValueError."""
        config = {"tokenizer": {}}
        model_config = {
            "tokenizer": {
                "component": "torchtune.models.some.tokenizer",
                # path_type is missing
            }
        }

        with pytest.raises(ValueError) as exc_info:
            configure_tokenizer(config, model_config, "SomeModel", "SomeModel")

        assert "Unknown tokenizer path_type" in str(exc_info.value)
        assert "None" in str(exc_info.value)

    def test_missing_tokenizer_config_raises_error(self):
        """Test that missing tokenizer config raises ValueError."""
        config = {"tokenizer": {}}
        model_config = {}  # No tokenizer config at all

        with pytest.raises(ValueError) as exc_info:
            configure_tokenizer(config, model_config, "SomeModel", "SomeModel")

        assert "Unknown tokenizer path_type" in str(exc_info.value)


class TestConfigureTokenizerBehavior:
    """Test general behavior of configure_tokenizer()."""

    def test_preserves_existing_config(self):
        """Test that existing tokenizer config values are preserved."""
        config = {"tokenizer": {"max_seq_len": 4096, "existing_key": "value"}}
        model_config = {
            "tokenizer": {
                "component": "torchtune.models.llama3.llama3_tokenizer",
                "path_type": "llama",
            }
        }

        result = configure_tokenizer(config, model_config, "Llama-3.2-1B", "Llama-3.2-1B")

        # Original values should still be present
        assert result["tokenizer"]["max_seq_len"] == 4096
        assert result["tokenizer"]["existing_key"] == "value"
        # New values should be added
        assert result["tokenizer"]["_component_"] == "torchtune.models.llama3.llama3_tokenizer"
        assert result["tokenizer"]["path"] == "${models_dir}/Llama-3.2-1B/original/tokenizer.model"


# =============================================================================
# Tests for MODEL_CONFIGS validation
# =============================================================================

class TestModelConfigsStructure:
    """Validate that all MODEL_CONFIGS entries have required fields."""

    def test_all_models_have_tokenizer_config(self):
        """Test that all MODEL_CONFIGS entries have tokenizer configuration."""
        for model_name, config in MODEL_CONFIGS.items():
            assert "tokenizer" in config, f"Model '{model_name}' missing tokenizer config"
            assert "component" in config["tokenizer"], f"Model '{model_name}' missing tokenizer component"
            assert "path_type" in config["tokenizer"], f"Model '{model_name}' missing tokenizer path_type"

    def test_all_models_have_valid_path_type(self):
        """Test that all MODEL_CONFIGS have a valid tokenizer path_type."""
        for model_name, config in MODEL_CONFIGS.items():
            path_type = config["tokenizer"]["path_type"]
            assert path_type in VALID_TOKENIZER_PATH_TYPES, (
                f"Model '{model_name}' has invalid path_type '{path_type}'. "
                f"Valid types: {VALID_TOKENIZER_PATH_TYPES}"
            )

    def test_all_models_have_required_fields(self):
        """Test that all MODEL_CONFIGS entries have all required fields."""
        required_fields = {"component", "checkpoint_files", "model_type", "dataset_type", "tokenizer", "slurm"}

        for model_name, config in MODEL_CONFIGS.items():
            missing = required_fields - set(config.keys())
            assert not missing, f"Model '{model_name}' missing required fields: {missing}"

    def test_all_models_have_slurm_config(self):
        """Test that all MODEL_CONFIGS entries have SLURM configuration."""
        slurm_fields = {"mem", "cpus", "gpus"}

        for model_name, config in MODEL_CONFIGS.items():
            assert "slurm" in config, f"Model '{model_name}' missing slurm config"
            missing = slurm_fields - set(config["slurm"].keys())
            assert not missing, f"Model '{model_name}' missing slurm fields: {missing}"


class TestModelConfigsIntegration:
    """Test that configure_tokenizer works with all actual MODEL_CONFIGS."""

    @pytest.mark.parametrize("model_name", list(MODEL_CONFIGS.keys()))
    def test_configure_tokenizer_works_for_model(self, model_name):
        """Test that configure_tokenizer works for each model in MODEL_CONFIGS."""
        config = {"tokenizer": {}}
        model_config = MODEL_CONFIGS[model_name]

        # Should not raise any exception
        result = configure_tokenizer(config, model_config, model_name, model_name)

        # Should have set component and path
        assert "_component_" in result["tokenizer"], f"Model '{model_name}' missing _component_"
        assert "path" in result["tokenizer"], f"Model '{model_name}' missing path"


# =============================================================================
# Tests for VALID_TOKENIZER_PATH_TYPES
# =============================================================================

class TestValidTokenizerPathTypes:
    """Test the VALID_TOKENIZER_PATH_TYPES constant."""

    def test_contains_expected_types(self):
        """Test that VALID_TOKENIZER_PATH_TYPES contains expected entries."""
        assert "llama" in VALID_TOKENIZER_PATH_TYPES
        assert "qwen" in VALID_TOKENIZER_PATH_TYPES

    def test_all_types_have_handler(self):
        """Test that all valid path_types have a handler in configure_tokenizer."""
        for path_type in VALID_TOKENIZER_PATH_TYPES:
            config = {"tokenizer": {}}
            model_config = {
                "tokenizer": {
                    "component": f"torchtune.models.test.test_tokenizer",
                    "path_type": path_type,
                }
            }

            # Should not raise - each path_type should have a handler
            result = configure_tokenizer(config, model_config, "TestModel", "TestModel")
            assert "_component_" in result["tokenizer"]

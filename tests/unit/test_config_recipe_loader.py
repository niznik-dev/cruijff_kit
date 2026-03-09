"""Unit tests for tools/torchtune/config_recipe_loader.py"""

import pytest
from unittest.mock import patch, MagicMock

from cruijff_kit.tools.torchtune.config_recipe_loader import (
    load_recipe_defaults,
    get_custom_recipe_path,
    extract_recipe_config,
    list_available_recipes,
    validate_recipe_exists,
    get_recipe_config,
    RecipeConfigError,
    RecipeNotFoundError,
    RecipeExtractionError,
)


# =============================================================================
# Tests for load_recipe_defaults()
# =============================================================================


def test_load_recipe_defaults_valid_yaml(tmp_path):
    """Test loading a valid YAML config file."""
    config_file = tmp_path / "recipe.yaml"
    config_file.write_text("""
model:
  lora_rank: 64
  lora_alpha: 128
optimizer:
  lr: 0.0003
batch_size: 4
epochs: 1
""")

    result = load_recipe_defaults(str(config_file))

    assert result["model"]["lora_rank"] == 64
    assert result["model"]["lora_alpha"] == 128
    assert result["optimizer"]["lr"] == 0.0003
    assert result["batch_size"] == 4
    assert result["epochs"] == 1


def test_load_recipe_defaults_empty_file(tmp_path):
    """Test loading an empty YAML file returns empty dict."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")

    result = load_recipe_defaults(str(config_file))

    assert result == {}


def test_load_recipe_defaults_file_not_found():
    """Test that missing file raises RecipeExtractionError."""
    with pytest.raises(RecipeExtractionError) as exc_info:
        load_recipe_defaults("/nonexistent/path/recipe.yaml")

    assert "Config file not found" in str(exc_info.value)


def test_load_recipe_defaults_invalid_yaml(tmp_path):
    """Test that invalid YAML raises RecipeExtractionError."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("""
model:
  lora_rank: 64
  invalid yaml here: [unclosed bracket
""")

    with pytest.raises(RecipeExtractionError) as exc_info:
        load_recipe_defaults(str(config_file))

    assert "Failed to parse YAML" in str(exc_info.value)


def test_load_recipe_defaults_non_dict_yaml(tmp_path):
    """Non-dict YAML (e.g., a bare list) raises RecipeExtractionError."""
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- item1\n- item2\n- item3\n")

    with pytest.raises(RecipeExtractionError):
        load_recipe_defaults(str(config_file))


def test_load_recipe_defaults_nested_config(tmp_path):
    """Test loading a deeply nested config."""
    config_file = tmp_path / "nested.yaml"
    config_file.write_text("""
model:
  _component_: torchtune.models.llama3_2.lora_llama3_2_1b
  lora_attn_modules: ['q_proj', 'v_proj']
  lora_rank: 8
tokenizer:
  path: /models/tokenizer.model
  max_seq_len: 2048
checkpointer:
  checkpoint_dir: /models/
  checkpoint_files:
    - model.safetensors
  model_type: LLAMA3_2
lr_scheduler:
  _component_: torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup
  num_warmup_steps: 100
""")

    result = load_recipe_defaults(str(config_file))

    assert (
        result["model"]["_component_"] == "torchtune.models.llama3_2.lora_llama3_2_1b"
    )
    assert result["model"]["lora_attn_modules"] == ["q_proj", "v_proj"]
    assert result["tokenizer"]["max_seq_len"] == 2048
    assert result["checkpointer"]["checkpoint_files"] == ["model.safetensors"]
    assert result["lr_scheduler"]["num_warmup_steps"] == 100


# =============================================================================
# Tests for get_custom_recipe_path()
# =============================================================================


def test_get_custom_recipe_path_builtin_recipe():
    """Test that built-in recipes (with /) return None."""
    result = get_custom_recipe_path("llama3_2/1B_lora_single_device")

    assert result is None


def test_get_custom_recipe_path_another_builtin():
    """Test another built-in recipe format."""
    result = get_custom_recipe_path("llama3_1/8B_full_single_device")

    assert result is None


def test_get_custom_recipe_path_existing_custom_recipe():
    """Test that existing custom recipes return the module path."""
    result = get_custom_recipe_path("lora_finetune_single_device_stable")
    assert (
        result
        == "cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_stable"
    )


def test_get_custom_recipe_path_nonexistent_custom():
    """Test that nonexistent custom recipes return None."""
    result = get_custom_recipe_path("nonexistent_recipe_name")

    assert result is None


# =============================================================================
# Tests for extract_recipe_config() with mocking
# =============================================================================


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.subprocess.run")
def test_extract_recipe_config_success(mock_run, tmp_path):
    """Test successful recipe extraction."""
    output_file = tmp_path / "recipe.yaml"

    # Mock successful subprocess run
    mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

    # Create the file that would be created by tune cp
    output_file.write_text("model:\n  lora_rank: 64\n")

    result = extract_recipe_config("llama3_2/1B_lora_single_device", str(output_file))

    assert result == str(output_file)
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args[0] == "tune"
    assert call_args[1] == "cp"
    assert call_args[2] == "llama3_2/1B_lora_single_device"


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.subprocess.run")
def test_extract_recipe_config_not_found(mock_run, tmp_path):
    """Test that recipe not found raises RecipeNotFoundError."""
    output_file = tmp_path / "recipe.yaml"

    # Mock subprocess returning error
    mock_run.return_value = MagicMock(
        returncode=1, stderr="Error: Config 'nonexistent/recipe' not found"
    )

    with pytest.raises(RecipeNotFoundError) as exc_info:
        extract_recipe_config("nonexistent/recipe", str(output_file))

    assert "not found" in str(exc_info.value).lower()


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.subprocess.run")
def test_extract_recipe_config_does_not_exist(mock_run, tmp_path):
    """Test recipe does not exist error message variant."""
    output_file = tmp_path / "recipe.yaml"

    mock_run.return_value = MagicMock(
        returncode=1, stderr="Error: Recipe does not exist"
    )

    with pytest.raises(RecipeNotFoundError):
        extract_recipe_config("bad/recipe", str(output_file))


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.subprocess.run")
def test_extract_recipe_config_other_error(mock_run, tmp_path):
    """Test that other errors raise RecipeExtractionError."""
    output_file = tmp_path / "recipe.yaml"

    mock_run.return_value = MagicMock(returncode=1, stderr="Some other error occurred")

    with pytest.raises(RecipeExtractionError) as exc_info:
        extract_recipe_config("some/recipe", str(output_file))

    assert "Failed to extract recipe" in str(exc_info.value)


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.subprocess.run")
def test_extract_recipe_config_tune_not_found(mock_run):
    """Test that missing tune CLI raises RecipeExtractionError."""
    mock_run.side_effect = FileNotFoundError("tune not found")

    with pytest.raises(RecipeExtractionError) as exc_info:
        extract_recipe_config("llama3_2/1B_lora_single_device")

    assert "tune CLI not found" in str(exc_info.value)


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.subprocess.run")
def test_extract_recipe_config_file_not_created(mock_run, tmp_path):
    """Test error when file is not created despite success return code."""
    output_file = tmp_path / "nonexistent" / "recipe.yaml"

    # Mock successful return but don't create the file
    mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

    with pytest.raises(RecipeExtractionError) as exc_info:
        extract_recipe_config("llama3_2/1B_lora_single_device", str(output_file))

    assert "file not found" in str(exc_info.value).lower()


# =============================================================================
# Tests for list_available_recipes() with mocking
# =============================================================================


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.subprocess.run")
def test_list_available_recipes_success(mock_run):
    """Test successful listing of recipes."""
    mock_output = """RECIPE                                   CONFIG
lora_finetune_single_device              llama3_2/1B_lora_single_device
                                         llama3_2/3B_lora_single_device
full_finetune_single_device              llama3_1/8B_full_single_device
"""
    mock_run.return_value = MagicMock(returncode=0, stdout=mock_output, stderr="")

    result = list_available_recipes()

    assert "llama3_2/1B_lora_single_device" in result
    assert "llama3_2/3B_lora_single_device" in result
    assert "llama3_1/8B_full_single_device" in result


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.subprocess.run")
def test_list_available_recipes_empty(mock_run):
    """Test empty recipe list."""
    mock_run.return_value = MagicMock(returncode=0, stdout="RECIPE CONFIG\n", stderr="")

    result = list_available_recipes()

    assert result == {}


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.subprocess.run")
def test_list_available_recipes_tune_not_found(mock_run):
    """Test that missing tune CLI raises RecipeExtractionError."""
    mock_run.side_effect = FileNotFoundError("tune not found")

    with pytest.raises(RecipeExtractionError) as exc_info:
        list_available_recipes()

    assert "tune CLI not found" in str(exc_info.value)


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.subprocess.run")
def test_list_available_recipes_command_fails(mock_run):
    """Test that command failure raises RecipeExtractionError."""
    import subprocess

    mock_run.side_effect = subprocess.CalledProcessError(
        1, "tune ls", stderr="Command failed"
    )

    with pytest.raises(RecipeExtractionError) as exc_info:
        list_available_recipes()

    assert "Failed to list recipes" in str(exc_info.value)


# =============================================================================
# Tests for validate_recipe_exists() with mocking
# =============================================================================


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.get_custom_recipe_path")
def test_validate_recipe_exists_custom(mock_get_custom):
    """Test that custom recipes are validated via get_custom_recipe_path."""
    mock_get_custom.return_value = (
        "cruijff_kit.tools.torchtune.custom_recipes.my_recipe"
    )

    result = validate_recipe_exists("my_custom_recipe")

    assert result is True
    mock_get_custom.assert_called_once_with("my_custom_recipe")


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.get_custom_recipe_path")
@patch("cruijff_kit.tools.torchtune.config_recipe_loader.list_available_recipes")
def test_validate_recipe_exists_builtin(mock_list, mock_get_custom):
    """Test that built-in recipes are validated via list_available_recipes."""
    mock_get_custom.return_value = None
    mock_list.return_value = {"llama3_2/1B_lora_single_device": "Recipe: lora_finetune"}

    result = validate_recipe_exists("llama3_2/1B_lora_single_device")

    assert result is True


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.get_custom_recipe_path")
@patch("cruijff_kit.tools.torchtune.config_recipe_loader.list_available_recipes")
def test_validate_recipe_exists_not_found(mock_list, mock_get_custom):
    """Test that nonexistent recipe returns False."""
    mock_get_custom.return_value = None
    mock_list.return_value = {"other/recipe": "description"}

    result = validate_recipe_exists("nonexistent/recipe")

    assert result is False


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.get_custom_recipe_path")
@patch("cruijff_kit.tools.torchtune.config_recipe_loader.list_available_recipes")
def test_validate_recipe_exists_cli_unavailable(mock_list, mock_get_custom):
    """Test that CLI unavailability returns True (optimistic)."""
    mock_get_custom.return_value = None
    mock_list.side_effect = RecipeExtractionError("tune CLI not found")

    result = validate_recipe_exists("some/recipe")

    # Should return True when CLI is unavailable (optimistic assumption)
    assert result is True


# =============================================================================
# Tests for get_recipe_config() with mocking
# =============================================================================


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.extract_recipe_config")
@patch("cruijff_kit.tools.torchtune.config_recipe_loader.load_recipe_defaults")
def test_get_recipe_config_success(mock_load, mock_extract):
    """Test successful get_recipe_config."""
    mock_extract.return_value = "/tmp/recipe.yaml"
    mock_load.return_value = {"model": {"lora_rank": 64}, "batch_size": 4}

    result = get_recipe_config("llama3_2/1B_lora_single_device")

    assert result["model"]["lora_rank"] == 64
    assert result["batch_size"] == 4
    mock_extract.assert_called_once()
    mock_load.assert_called_once_with("/tmp/recipe.yaml")


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.extract_recipe_config")
@patch("cruijff_kit.tools.torchtune.config_recipe_loader.load_recipe_defaults")
def test_get_recipe_config_with_cache_dir(mock_load, mock_extract, tmp_path):
    """Test get_recipe_config with cache directory."""
    cache_dir = tmp_path / "cache"
    mock_extract.return_value = str(cache_dir / "llama3_2_1B_lora_single_device.yaml")
    mock_load.return_value = {"epochs": 2}

    result = get_recipe_config(
        "llama3_2/1B_lora_single_device", cache_dir=str(cache_dir)
    )

    assert result["epochs"] == 2
    # Verify extract was called with the cache path
    call_args = mock_extract.call_args
    assert "llama3_2_1B_lora_single_device.yaml" in call_args[0][1]


@patch("cruijff_kit.tools.torchtune.config_recipe_loader.extract_recipe_config")
def test_get_recipe_config_extraction_fails(mock_extract):
    """Test that extraction failure propagates."""
    mock_extract.side_effect = RecipeNotFoundError("Recipe not found")

    with pytest.raises(RecipeNotFoundError):
        get_recipe_config("nonexistent/recipe")


# =============================================================================
# Tests for exception classes
# =============================================================================


def test_exception_hierarchy():
    """Test that exception classes have correct inheritance."""
    assert issubclass(RecipeNotFoundError, RecipeConfigError)
    assert issubclass(RecipeExtractionError, RecipeConfigError)
    assert issubclass(RecipeConfigError, Exception)


def test_exception_messages():
    """Test that exceptions can be created with messages."""
    error1 = RecipeConfigError("Base error")
    error2 = RecipeNotFoundError("Recipe xyz not found")
    error3 = RecipeExtractionError("Failed to extract")

    assert str(error1) == "Base error"
    assert str(error2) == "Recipe xyz not found"
    assert str(error3) == "Failed to extract"

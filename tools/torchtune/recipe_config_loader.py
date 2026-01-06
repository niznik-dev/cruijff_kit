"""Recipe configuration loader for torchtune recipes.

This module provides utilities to extract default parameters from torchtune
base recipes and merge them with user-specified overrides. This allows users
to leverage torchtune's built-in recipe defaults instead of maintaining
hardcoded templates.

Key functions:
- list_available_recipes(): List all available torchtune recipes
- extract_recipe_config(): Extract recipe config YAML using tune CLI
- load_recipe_defaults(): Load and parse recipe config
- merge_parameters(): Merge recipe defaults with user overrides
- get_custom_recipe_path(): Resolve custom cruijff_kit recipe paths
"""

import subprocess
import tempfile
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from copy import deepcopy

# Set up logging
logger = logging.getLogger(__name__)


class RecipeConfigError(Exception):
    """Base exception for recipe configuration errors."""
    pass


class RecipeNotFoundError(RecipeConfigError):
    """Raised when a specified recipe cannot be found."""
    pass


class RecipeExtractionError(RecipeConfigError):
    """Raised when recipe config extraction fails."""
    pass


def list_available_recipes() -> Dict[str, str]:
    """
    List available torchtune recipes using tune CLI.

    Returns:
        Dictionary mapping recipe names to descriptions
        Example: {"llama3_2/1B_lora_single_device": "Single GPU LoRA for 1B model"}

    Raises:
        RecipeExtractionError: If tune CLI is not available or command fails
    """
    try:
        result = subprocess.run(
            ["tune", "ls"],
            capture_output=True,
            text=True,
            check=True
        )

        recipes = {}
        lines = result.stdout.strip().split('\n')

        # Parse output format:
        # RECIPE                                   CONFIG
        # recipe_name                              config/path
        #                                          another_config/path

        current_recipe = None

        for line in lines:
            # Skip header line
            if 'RECIPE' in line and 'CONFIG' in line:
                continue

            if not line.strip():
                continue

            # Split line at position 40 (approximate column boundary)
            # Recipe names are left-aligned, configs are right-aligned/indented
            if len(line) >= 40:
                recipe_part = line[:40].strip()
                config_part = line[40:].strip()
            else:
                recipe_part = line.strip()
                config_part = ""

            # If recipe_part has content, it's a new recipe
            if recipe_part:
                current_recipe = recipe_part

            # If config_part has content, add it to recipes dict
            if config_part:
                description = f"Recipe: {current_recipe}" if current_recipe else "Built-in config"
                recipes[config_part] = description

        logger.info(f"Found {len(recipes)} available configs")
        return recipes

    except FileNotFoundError:
        raise RecipeExtractionError(
            "tune CLI not found. Ensure torchtune is installed and accessible."
        )
    except subprocess.CalledProcessError as e:
        raise RecipeExtractionError(
            f"Failed to list recipes: {e.stderr}"
        )


def extract_recipe_config(recipe_name: str, output_path: Optional[str] = None) -> str:
    """
    Extract recipe config YAML using tune CLI.

    Args:
        recipe_name: Name of the recipe (e.g., "llama3_2/1B_lora_single_device")
        output_path: Optional path to save the extracted config. If not provided,
                    a temporary file will be created and returned.

    Returns:
        Path to the extracted config file

    Raises:
        RecipeNotFoundError: If recipe does not exist
        RecipeExtractionError: If extraction fails
    """
    try:
        # Create output path if not provided
        if output_path is None:
            # Create a temp file in system temp directory
            temp_dir = Path(tempfile.gettempdir()) / "cruijff_kit_recipes"
            temp_dir.mkdir(exist_ok=True)
            # Use recipe name as filename (replace / with _)
            safe_name = recipe_name.replace('/', '_')
            output_path = str(temp_dir / f"{safe_name}.yaml")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Run tune cp to extract config
        result = subprocess.run(
            ["tune", "cp", recipe_name, str(output_file)],
            capture_output=True,
            text=True,
            check=False  # Don't raise immediately, check return code manually
        )

        if result.returncode != 0:
            stderr_lower = result.stderr.lower()
            if 'not found' in stderr_lower or 'does not exist' in stderr_lower:
                raise RecipeNotFoundError(
                    f"Recipe '{recipe_name}' not found. Use list_available_recipes() "
                    f"to see available recipes."
                )
            else:
                raise RecipeExtractionError(
                    f"Failed to extract recipe '{recipe_name}': {result.stderr}"
                )

        # Verify file was created
        if not output_file.exists():
            raise RecipeExtractionError(
                f"Recipe extraction appeared to succeed but file not found: {output_file}"
            )

        logger.info(f"Extracted recipe '{recipe_name}' to {output_file}")
        return str(output_file)

    except FileNotFoundError:
        raise RecipeExtractionError(
            "tune CLI not found. Ensure torchtune is installed and accessible."
        )


def load_recipe_defaults(config_path: str) -> Dict[str, Any]:
    """
    Load recipe config YAML and extract default parameters.

    Args:
        config_path: Path to the recipe config YAML file

    Returns:
        Nested dictionary with recipe defaults

    Raises:
        RecipeExtractionError: If file cannot be loaded or parsed
    """
    try:
        config_file = Path(config_path)

        if not config_file.exists():
            raise RecipeExtractionError(f"Config file not found: {config_path}")

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            config = {}

        logger.info(f"Loaded recipe config from {config_path}")
        logger.debug(f"Recipe config keys: {list(config.keys())}")

        return config

    except yaml.YAMLError as e:
        raise RecipeExtractionError(f"Failed to parse YAML from {config_path}: {e}")
    except Exception as e:
        raise RecipeExtractionError(f"Failed to load recipe config: {e}")

def merge_parameters(
    recipe_defaults: Dict[str, Any],
    experiment_controls: Dict[str, Any],
    run_parameters: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:

    tracking: Dict[str, Dict[str, str]] = {}

    # Flatten a nested dict into { "a.b.c": value }
    def flatten(d: Dict[str, Any], parent: str = "") -> Dict[str, Any]:
        flat = {}
        for k, v in d.items():
            p = f"{parent}.{k}" if parent else k
            if isinstance(v, dict):
                flat.update(flatten(v, p))
            else:
                flat[p] = v
        return flat

    # Search tree and override any matching terminal key
    def apply_flat_override(tree: Dict[str, Any], key: str, value: Any, source: str, path=""):
        for k, v in tree.items():
            p = f"{path}.{k}" if path else k

            if isinstance(v, dict):
                apply_flat_override(v, key, value, source, p)
            else:
                if k == key:  # match any nested key name
                    tree[k] = deepcopy(value)
                    tracking[p] = {"source": source, "value": value}

    # Step 1 — apply flat overrides from control and run levels
    # We process the flat overrides BEFORE recursive dict merge
    def apply_flat_overrides(base: Dict[str, Any], params: Dict[str, Any], source: str):
        flat = flatten(params)
        for full_key, value in flat.items():
            # full_key is something like "lora_rank" even if nested
            key = full_key.split(".")[-1]
            apply_flat_override(base, key, value, source)

    # Step 2 — safe recursive merge (same precedence, but flat overrides already applied)
    def merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any], source: str, path=""):
        for k, v in overrides.items():
            p = f"{path}.{k}" if path else k

            # Skip keys not in recipe (only override existing keys)
            if k not in base:
                continue

            # Dict merge
            if isinstance(v, dict) and isinstance(base[k], dict):
                merge_dicts(base[k], v, source, p)
                continue

            # Replace scalar or list
            base[k] = deepcopy(v)
            tracking[p] = {"source": source, "value": v}

        return base

    # ------------------------------------------------------------
    # Begin actual merge
    # ------------------------------------------------------------
    result = deepcopy(recipe_defaults)

    # Initialize tracking with recipe defaults
    def mark_recipe(d: Dict[str, Any], path=""):
        for k, v in d.items():
            p = f"{path}.{k}" if path else k
            if isinstance(v, dict):
                mark_recipe(v, p)
            else:
                tracking[p] = {"source": "recipe", "value": v}

    mark_recipe(result)

    # Apply controls first (both flat and nested), then run parameters
    # This ensures run parameters always override controls for the same key
    apply_flat_overrides(result, experiment_controls, "control")
    merge_dicts(result, experiment_controls, "control")

    # Run parameters override controls
    apply_flat_overrides(result, run_parameters, "run")
    merge_dicts(result, run_parameters, "run")

    return result, tracking

def get_custom_recipe_path(recipe_name: str) -> Optional[str]:
    """
    Get Python module path for custom cruijff_kit recipes.

    Custom recipes are stored in tools/torchtune/custom_recipes/ and should
    be referenced using their module path for torchtune.

    Args:
        recipe_name: Name of the custom recipe
                    (e.g., "lora_finetune_single_device_with_val")

    Returns:
        Full module path (e.g., "cruijff_kit.tools.torchtune.custom_recipes.lora_finetune_single_device_with_val")
        or None if not a custom recipe or file doesn't exist
    """
    # Check if this is a custom recipe (doesn't contain /)
    if '/' in recipe_name:
        # This is a torchtune built-in recipe
        return None

    # Look for custom recipe in our custom_recipes directory
    script_dir = Path(__file__).parent
    custom_recipes_dir = script_dir / "custom_recipes"

    # Check for Python file
    recipe_file = custom_recipes_dir / f"{recipe_name}.py"

    if recipe_file.exists():
        # Return the full module path
        module_path = f"cruijff_kit.tools.torchtune.custom_recipes.{recipe_name}"
        logger.info(f"Found custom recipe: {module_path}")
        return module_path

    # Not a custom recipe or doesn't exist
    return None


def validate_recipe_exists(recipe_name: str) -> bool:
    """
    Validate that a recipe exists (either built-in or custom).

    Args:
        recipe_name: Name of the recipe to validate

    Returns:
        True if recipe exists, False otherwise
    """
    # Check if custom recipe
    if get_custom_recipe_path(recipe_name) is not None:
        return True

    # Check if built-in recipe
    try:
        recipes = list_available_recipes()
        return recipe_name in recipes
    except RecipeExtractionError:
        # If we can't list recipes, assume it might exist
        logger.warning(f"Could not validate recipe '{recipe_name}' - tune CLI unavailable")
        return True


def get_recipe_config(
    recipe_name: str,
    cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get recipe configuration (convenience function combining extract + load).

    Args:
        recipe_name: Name of the recipe
        cache_dir: Optional directory to cache extracted configs

    Returns:
        Recipe configuration dictionary

    Raises:
        RecipeNotFoundError: If recipe doesn't exist
        RecipeExtractionError: If extraction or loading fails
    """
    # Determine output path
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        safe_name = recipe_name.replace('/', '_')
        output_path = str(cache_path / f"{safe_name}.yaml")
    else:
        output_path = None

    # Extract config
    config_path = extract_recipe_config(recipe_name, output_path)

    # Load and return
    return load_recipe_defaults(config_path)


def format_merge_tracking(tracking: Dict[str, Dict[str, str]]) -> str:
    """
    Format merge tracking information for logging.

    Args:
        tracking: Merge tracking dictionary from merge_parameters()

    Returns:
        Formatted string suitable for logging
    """
    lines = ["Parameter merge tracking:"]

    # Group by source
    by_source = {"recipe": [], "control": [], "run": []}

    for path, info in tracking.items():
        source = info["source"]
        value = info["value"]

        # Format value for display
        if isinstance(value, (dict, list)):
            value_str = f"<{type(value).__name__}>"
        else:
            value_str = str(value)

        by_source.setdefault(source, []).append(f"  {path} = {value_str}")

    # Output by source
    for source in ["recipe", "control", "run"]:
        if by_source[source]:
            lines.append(f"\n{source.upper()} parameters:")
            lines.extend(sorted(by_source[source]))

    return "\n".join(lines)


# Example usage and testing
# MUST be run in a conda environment with torchtune installed 
if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)

    print("Recipe Config Loader - Test Mode")
    print("=" * 50)

    # Test 1: List recipes
    print("\nTest 1: Listing available recipes...")
    try:
        recipes = list_available_recipes()
        print(f"Found {len(recipes)} recipes")
        # print first 5 recipes 
        for name, desc in list(recipes.items())[:5]:
            print(f"  - {name}: {desc}")
    except RecipeExtractionError as e:
        print(f"  Error: {e}")

    # Test 2: Extract a recipe config
    print("\nTest 2: Extracting a recipe config...")
    try:
        config_path = extract_recipe_config("llama3_2/1B_lora_single_device")
        print(f"  Extracted to: {config_path}")

        config = load_recipe_defaults(config_path)
        print(f"  Config keys: {list(config.keys())}")
    except (RecipeNotFoundError, RecipeExtractionError) as e:
        print(f"  Error: {e}")

    # Test 3: Parameter merging
    print("\nTest 3: Testing parameter merging...")
    recipe_defaults = {
        "lora_rank": 64,
        "lr": 1e-4,
        "batch_size": 4,
        "epochs": 1,
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4
        }
    }

    controls = {
        "epochs": 2,
        "optimizer": {
            "lr": 2e-4
        }
    }

    run_params = {
        "lora_rank": 8
    }

    merged, tracking = merge_parameters(recipe_defaults, controls, run_params)
    print(f"  Merged config: {json.dumps(merged, indent=2)}")
    print(f"\n{format_merge_tracking(tracking)}")

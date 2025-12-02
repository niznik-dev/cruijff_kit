# Recipe-Based Parameter Extraction Refactoring Plan

## Executive Summary

Refactor cruijff_kit to allow users to specify a torchtune base recipe during experiment design, then extract and use its default parameters during scaffolding instead of using hardcoded defaults from templates.

**Key Requirements:**
- Recipe name specified in design-experiment, used in scaffold-torchtune
- Reference torchtune's built-in recipe configs directly
- Users can override recipe defaults in experiment_summary.yaml
- **Control fields are optional** - only include if overriding defaults
- Defaults don't need to be present in YAML (extracted at scaffold time from recipe or setup_finetune.py)
- Optional feature with fallback to current template approach
- Precedence: Recipe defaults (or setup_finetune.py defaults) → Controls → Run parameters

## Architecture Overview

### Current Flow
```
design-experiment → experiment_summary.yaml
                 ↓
scaffold-torchtune → uses finetune_template.yaml (hardcoded defaults)
                  → merges with user overrides
                  → generates setup_finetune.yaml → finetune.yaml
```

### Proposed Flow
```
design-experiment → user selects base_recipe (optional)
                 → stores in experiment_summary.yaml controls
                 ↓
scaffold-torchtune → if base_recipe specified:
                     → extracts recipe config using tune CLI
                     → merges: recipe → controls → run.parameters
                  → else: fallback to template approach
                  → generates setup_finetune.yaml → finetune.yaml
```

## Implementation Approach

### 1. Recipe Config Extraction Utility

**New file:** `tools/torchtune/recipe_config_loader.py` (~250 lines)

**Core functions:**

```python
def list_available_recipes() -> Dict[str, str]:
    """
    List available torchtune recipes using tune CLI.
    Returns: {recipe_name: description}
    """
    # Run: tune ls recipes
    # Parse output to extract recipe names and descriptions

def extract_recipe_config(recipe_name: str, output_path: str) -> str:
    """
    Extract recipe config YAML using tune CLI.
    Args:
        recipe_name: e.g. "llama3_2/1B_lora_single_device"
        output_path: Where to save the extracted config
    Returns: Path to extracted config file
    """
    # Run: tune cp {recipe_name} {output_path}
    # Returns path to the extracted YAML

def load_recipe_defaults(config_path: str) -> Dict[str, Any]:
    """
    Load recipe config YAML and extract default parameters.
    Returns: Nested dict with recipe defaults
    """
    # Load YAML using yaml.safe_load()
    # Return as nested dictionary

def merge_parameters(
    recipe_defaults: Dict,
    experiment_controls: Dict,
    run_parameters: Dict
) -> Dict:
    """
    Merge parameters with proper precedence.
    Recipe defaults → Controls → Run parameters

    Rules:
    - Scalars: later values override
    - Nested dicts: recursive merge
    - Lists: complete replacement (no append)
    """
    # Implement recursive merge with proper precedence

def get_custom_recipe_path(recipe_name: str) -> Optional[str]:
    """
    Get Python module path for custom cruijff_kit recipes.
    Returns: Full module path or None if not custom recipe
    """
    # Map recipe names to cruijff_kit.tools.torchtune.custom_recipes.*
```

**Implementation notes:**
- Use subprocess to call `tune ls` and `tune cp` commands
- Cache extracted configs to avoid repeated CLI calls
- Handle missing torchtune installation gracefully
- Log all extraction steps for debugging

### 2. Design-Experiment Changes

**File:** `.claude/skills/design-experiment/param_selection.md`

Add new step after "Step 3: Confirm Tool Choices":

```markdown
## Step 3.5: Select Base Recipe (Optional)

Ask user if they want to use a torchtune base recipe for default parameters.

**If yes:**
1. Run `list_available_recipes()` to get options
2. Present recipes with descriptions
3. User selects recipe name (e.g., "llama3_2/1B_lora_single_device")
4. Store in `controls.base_recipe`
5. Inform user they can override any recipe defaults in controls section

**If no/skip:**
- Omit `base_recipe` field from controls
- System uses setup_finetune.py defaults

**Important:** Inform user that **all control fields are optional**:
- With base_recipe: only include fields you want to override
- Without base_recipe: only include fields you want to override from setup_finetune.py defaults
- If controls section is minimal or empty, all defaults are used

**Example recipes:**
- `llama3_2/1B_lora_single_device` - Single GPU LoRA for 1B model
- `llama3_2/3B_lora_single_device` - Single GPU LoRA for 3B model
- `llama3/8B_lora_distributed` - Multi-GPU LoRA for 8B model
```

**File:** `.claude/skills/design-experiment/validation.md`

Add validation:

```markdown
### Recipe Validation (if specified)

If `controls.base_recipe` present:
- Verify recipe exists using `list_available_recipes()`
- Log recipe selection in design-experiment.jsonl
```

**File:** `.claude/skills/design-experiment/templates/experiment_summary.yaml`

Update controls section comment:

```yaml
controls:
  # ALL FIELDS IN THIS SECTION ARE OPTIONAL
  # - With base_recipe: specify only parameters you want to override
  # - Without base_recipe: defaults from setup_finetune.py are used
  # - Any field can be omitted if you want to use the default value

  base_recipe: string              # OPTIONAL: Torchtune recipe name (e.g., "llama3_2/1B_lora_single_device")
                                   # If omitted, uses setup_finetune.py defaults
                                   # Recipe provides defaults for optimizer, scheduler, dataset config
  epochs: int                      # OPTIONAL: Override recipe/default
  batch_size: int                  # OPTIONAL: Override recipe/default
  lora_rank: int                   # OPTIONAL: Override recipe/default
  lr: float                        # OPTIONAL: Override recipe/default
  # ... any other parameters are OPTIONAL overrides
```

### 3. Scaffold-Torchtune Changes

**File:** `.claude/skills/scaffold-experiment/optimizers/torchtune_agent.md`

Update prompt template section to add step:

```markdown
Your tasks:
1. Read experiment_summary.yaml to extract run configurations
2. Read claude.local.md for environment-specific settings
3. **Check if controls.base_recipe is specified:**
   - **If yes:**
     - Extract recipe config using `extract_recipe_config()`
     - Load recipe defaults using `load_recipe_defaults()`
   - **If no:**
     - Use template-based approach (current behavior)
4. Identify which runs are fine-tuned vs control
5. For ONLY the fine-tuned runs:
   - **Merge parameters:** recipe defaults → controls → run.parameters
   - Create run directory
   - Generate setup_finetune.yaml with merged parameters
   - Execute setup_finetune.py
   - Verify outputs created successfully
6. For control runs: Create directory only
7. Create detailed log at scaffold-torchtune.log
8. Verify parameters in finetune.yaml match expected values
```

Update "What scaffold-torchtune Does" section:

```markdown
The subagent performs these operations autonomously:

1. **Loads experiment configuration** from experiment_summary.yaml
2. **If base_recipe specified:**
   - Extracts recipe config using tune CLI
   - Loads recipe defaults
   - Caches config for reuse across runs
3. **Creates run directories** for all runs
4. **For fine-tuned runs:**
   - Merges parameters: recipe defaults → controls → run.parameters
   - Generates `setup_finetune.yaml` with merged config
   - Executes `setup_finetune.py`
   - Validates generated configurations
5. **For control runs:** Creates directory only
6. **Creates detailed log** with parameter merge details
```

### 4. Changes to experiment_summary.yaml Structure

**No breaking changes required.** All control fields are now **optional**:

**Minimal example (with base_recipe):**
```yaml
controls:
  base_recipe: "llama3_2/1B_lora_single_device"  # OPTIONAL - specifies recipe for defaults
  # All other fields omitted → use recipe defaults
```

**Minimal example (without base_recipe):**
```yaml
controls:
  # No fields specified → use setup_finetune.py defaults
  # Or specify only the fields you want to override
  epochs: 2  # Override default
```

**Example with selective overrides:**
```yaml
controls:
  base_recipe: "llama3_2/1B_lora_single_device"  # OPTIONAL - new field
  epochs: 2                                       # OPTIONAL - override recipe default
  batch_size: 8                                   # OPTIONAL - override recipe default
  # Other parameters not specified → use recipe defaults
```

**Parameter precedence examples:**

```yaml
# Example 1: Minimal config with recipe
# Recipe has: lora_rank=64, lr=1e-4, batch_size=4, epochs=1
controls:
  base_recipe: "llama3_2/1B_lora_single_device"
  # No other fields → all use recipe defaults

runs:
  - name: "run1"
    parameters:
      lora_rank: 8     # Override recipe default (64 → 8)
      # lr, batch_size, epochs all use recipe defaults
```

```yaml
# Example 2: Override some controls, vary others per run
# Recipe has: lora_rank=64, lr=1e-4, batch_size=4, epochs=1
controls:
  base_recipe: "llama3_2/1B_lora_single_device"
  epochs: 2            # Override recipe default for all runs
  # lora_rank, lr, batch_size use recipe defaults

runs:
  - name: "run1"
    parameters:
      lora_rank: 8     # Override recipe default for this run
      # epochs=2 (from controls), lr=1e-4 (recipe), batch_size=4 (recipe)
  - name: "run2"
    parameters:
      lora_rank: 16    # Override recipe default for this run
      # epochs=2 (from controls), lr=1e-4 (recipe), batch_size=4 (recipe)
```

```yaml
# Example 3: No recipe, minimal overrides
# Uses setup_finetune.py defaults: lora_rank=64, lr=1e-4, batch_size=4, epochs=1
controls:
  # No base_recipe → use setup_finetune.py defaults
  epochs: 2            # Override only this parameter

runs:
  - name: "run1"
    parameters:
      lora_rank: 8     # Override for this run
      # epochs=2 (from controls), lr=1e-4 (default), batch_size=4 (default)
```

### 5. Backward Compatibility

**Fallback trigger:** Missing `controls.base_recipe` field

**Behavior:**
- scaffold-torchtune detects missing field
- Uses existing template-based approach:
  - Loads `/tools/torchtune/templates/finetune_template.yaml`
  - Uses argparse defaults from `setup_finetune.py`
- Logs fallback decision

**Migration:** Existing experiments continue working without changes.

## Implementation Phases

### Phase 1: Recipe Config Loader (Infrastructure)
**Files:**
- Create `tools/torchtune/recipe_config_loader.py`
- Create unit tests for all functions
- Test with conda environment that has torchtune installed

**Deliverables:**
- Working recipe listing function
- Working config extraction function
- Working parameter merging with precedence
- Test coverage >80%

**Risk:** Low - no workflow changes

### Phase 2: Design-Experiment Integration
**Files:**
- `.claude/skills/design-experiment/param_selection.md`
- `.claude/skills/design-experiment/validation.md`
- `.claude/skills/design-experiment/templates/experiment_summary.yaml`

**Deliverables:**
- Recipe selection step (optional)
- Validation for recipe existence
- Updated template comments

**Risk:** Low - optional feature

### Phase 3: Scaffold-Torchtune Integration
**Files:**
- `.claude/skills/scaffold-experiment/optimizers/torchtune_agent.md`
- Update scaffold-torchtune agent implementation

**Deliverables:**
- Recipe config extraction during scaffolding
- Parameter merging with proper precedence
- Fallback to template if no recipe
- Enhanced logging showing merge results

**Risk:** Medium - changes core scaffolding

### Phase 4: Testing
**Tests:**
- End-to-end with recipe specified
- End-to-end without recipe (fallback)
- Parameter override precedence verification
- Multiple runs with varied parameters

**Deliverables:**
- Integration tests passing
- Update `.claude/workflow_test.yaml` to test recipe approach

**Risk:** Low - testing phase

### Phase 5: Documentation
**Files:**
- `CLAUDE.md` - Add recipe selection to workflow
- `ARCHITECTURE.md` - Document recipe config extraction
- `README.md` - Update feature list
- Create example experiment with recipe

**Deliverables:**
- Complete documentation
- Working example

**Risk:** Low - documentation

## Critical Files to Modify

### New Files (1)
1. `tools/torchtune/recipe_config_loader.py` - Core utility (~250 lines)

### Modified Files (7)
1. `.claude/skills/design-experiment/param_selection.md` - Add recipe selection step
2. `.claude/skills/design-experiment/validation.md` - Add recipe validation
3. `.claude/skills/design-experiment/templates/experiment_summary.yaml` - Update comments
4. `.claude/skills/scaffold-experiment/optimizers/torchtune_agent.md` - Add recipe loading
5. `CLAUDE.md` - Document new workflow step
6. `ARCHITECTURE.md` - Document recipe extraction architecture
7. `.claude/workflow_test.yaml` - Add recipe-based test

### No Changes Required
- `tools/torchtune/setup_finetune.py` - Already supports custom_recipe parameter
- `tools/torchtune/templates/finetune_template.yaml` - Remains as fallback
- Custom recipe Python files - No modifications
- Existing experiment_summary.yaml files - Continue to work

## Implementation Details

### Recipe Config Extraction

**Using tune CLI:**

```python
import subprocess
import tempfile
import yaml
from pathlib import Path

def extract_recipe_config(recipe_name: str) -> Dict[str, Any]:
    """Extract recipe config using tune cp command."""

    # Create temp directory for extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "config.yaml"

        # Run tune cp to extract config
        result = subprocess.run(
            ["tune", "cp", recipe_name, str(output_path)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise ValueError(f"Failed to extract recipe: {result.stderr}")

        # Load and return config
        with open(output_path) as f:
            return yaml.safe_load(f)
```

**Parameter merging:**

```python
def merge_parameters(base: Dict, overrides: Dict) -> Dict:
    """
    Recursively merge overrides into base config.

    Rules:
    - Scalars: override replaces base
    - Dicts: recursive merge
    - Lists: override replaces base (no append)
    """
    result = base.copy()

    for key, override_value in overrides.items():
        if key not in result:
            # New key - add directly
            result[key] = override_value
        elif isinstance(override_value, dict) and isinstance(result[key], dict):
            # Both dicts - recursive merge
            result[key] = merge_parameters(result[key], override_value)
        else:
            # Scalar or list - replace
            result[key] = override_value

    return result
```

### Logging Parameter Merging

Log merge operations for debugging:

```python
# In scaffold-torchtune.log
{
  "timestamp": "2025-12-02T10:30:00Z",
  "action": "MERGE_PARAMETERS",
  "run_name": "Llama-3.2-1B-Instruct_rank8",
  "recipe_name": "llama3_2/1B_lora_single_device",
  "merged_parameters": {
    "lora_rank": {"source": "run", "value": 8},
    "lr": {"source": "recipe", "value": 1e-4},
    "batch_size": {"source": "control", "value": 4}
  }
}
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Recipe config format changes | Document expected structure; add version checking |
| tune CLI not available | Graceful fallback to template approach; clear error message |
| Parameter merge errors | Extensive logging; validation before writing configs |
| Recipe incompatible with GPU config | Validate recipe supports requested GPU count |
| Missing recipe parameters | Use template defaults for missing params; log warnings |

## Success Criteria

1. ✓ Users can select torchtune recipes during design-experiment
2. ✓ Recipe defaults extracted and merged correctly
3. ✓ User overrides work at controls and run levels
4. ✓ Fallback to template works when recipe not specified
5. ✓ All existing experiments continue working
6. ✓ Integration tests pass for recipe and non-recipe paths
7. ✓ Documentation complete and accurate

## Reference Material

- **torchtune_config_writer**: Similar implementation pattern for recipe-based config generation
- **Current implementation**: `tools/torchtune/setup_finetune.py` for parameter handling patterns
- **Torchtune docs**: Recipe structure and tune CLI usage

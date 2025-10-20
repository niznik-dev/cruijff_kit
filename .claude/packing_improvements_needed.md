# Packing Support Improvements

**Created**: 2025-10-20
**Status**: Planning document for future improvements

## Overview

This document lists all changes needed to properly support `packed=True` in torchtune configs. Packing is more efficient (reduces padding, faster training) but requires careful memory management and batch size estimation.

**Design Philosophy**: Enable packing by default, but ensure batch sizes are calculated correctly to avoid OOM errors.

---

## Background: What Is Packing?

**Packing** concatenates multiple training examples into single sequences up to `max_seq_len`:
- **Without packing**: Each example is padded to `max_seq_len` (wastes GPU memory on padding tokens)
- **With packing**: Multiple short examples are concatenated to fill `max_seq_len` (more efficient)

**Memory Impact**:
- Packed sequences have NO padding tokens → more actual tokens per batch
- Effective batch size = `batch_size × avg_examples_per_packed_sequence`
- Example: batch_size=16 with 3 examples/sequence → effective batch of 48 examples
- Memory usage scales with number of actual tokens, not number of sequences

**Torchtune's Default Approach**:
- `packed: False` (safer, uses padding)
- `batch_size: 4` (small to accommodate padding)
- `gradient_accumulation_steps: 8` (increases effective batch without memory cost)
- Effective batch: 32 examples

**Cruijff_kit's Target Approach**:
- `packed: True` (efficient, no padding waste)
- `batch_size: 4-8` (adjusted for packing overhead)
- `gradient_accumulation_steps: 1-2` (less needed with packing efficiency)
- Effective batch: variable based on packing density

---

## 1. Skills That Need Updates

### A. plan-runs Skill

**File**: `.claude/skills/plan-runs/skill.md`

**Current Issue**: Batch size estimation doesn't account for packing at all.

**Changes Needed**:

#### 1. Add new section: "Dataset Packing Considerations"

Insert before the "Batch Size Estimation" section:

```markdown
## Dataset Packing Considerations

**Packing** is a memory optimization that concatenates multiple training examples into single sequences up to `max_seq_len`. It's more efficient but requires careful batch size tuning.

### When to Use Packing

**Use packing (`packed: True`) when**:
- Dataset has short examples (< 50% of max_seq_len on average)
- Training on instruction-following tasks with varied response lengths
- Want to maximize GPU utilization and training speed

**Avoid packing (`packed: False`) when**:
- Examples are already near max_seq_len (little to pack)
- Unsure about memory constraints (safer to start without)
- Debugging training issues (simpler to reason about batch sizes)

### Packing Density Analysis

Before choosing batch size, analyze packing density from prior runs.

#### How to Extract Packing Density from Logs

**Step 1: Locate the training log**
```bash
# Find SLURM output file in your run directory
ls /path/to/run_dir/slurm-*.out

# Example: /scratch/gpfs/username/experiments/run1/slurm-12345678.out
```

**Step 2: Search for packing information**
```bash
# Search for dataset packing messages
grep -i "packing dataset" /path/to/slurm-*.out

# Or look at the start of training
head -100 /path/to/slurm-*.out | grep -A10 "dataset"
```

**Step 3: Interpret the output**

Torchtune may show packing info in several ways:

**Example A: Explicit packing stats** (if recipe logs them)
```
Packing dataset: 100%|██████████| 8000/8000 [00:01<00:00]
INFO: Packed 8000 examples into 2857 sequences
INFO: Average packing density: 2.80 examples/sequence
INFO: Packing efficiency: 87% non-padding tokens
```
→ **Packing density = 2.8** (explicitly stated)

**Example B: Calculate from dataset info**
```
Loading dataset from /path/to/data/train.parquet...
Loaded 8000 examples
Packing dataset with max_seq_len=2048...
Dataset packed: 2857 total sequences
```
→ **Packing density = 8000 ÷ 2857 = 2.80**

**Example C: No explicit output** (older torchtune versions)
```
Dataset loaded: 8000 examples
Training started...
```
→ **Estimate based on task:**
  - Short prompts/responses (< 30% max_seq_len): density ≈ 3-4
  - Medium prompts/responses (30-60% max_seq_len): density ≈ 2-3
  - Long prompts/responses (> 60% max_seq_len): density ≈ 1.5-2

**Step 4: Document for future reference**

Add to your experiment notes or runs_plan.md:
```markdown
## Packing Analysis (from run: bright_horizon)
- Dataset: capitalization_5letter
- Examples: 8000
- Packed sequences: 2857
- **Packing density: 2.8 examples/sequence**
- Implication: batch_size with packing ≈ batch_size_unpacked ÷ 2.8
```

**Key metrics**:
- **Packing density**: Average examples per packed sequence (most important for memory estimation)
- **Packing efficiency**: % of tokens that are actual data vs padding (for throughput estimation)

**Typical values by task type**:
- 5-letter words (capitalization): ~3-4 examples/sequence (high density)
- 13-letter words (capitalization): ~2-3 examples/sequence (medium density)
- Long-form generation: ~1-2 examples/sequence (low density)
- Chat conversations: ~1.5-2.5 examples/sequence (variable density)

### Memory Impact of Packing

**Formula for packed memory estimation**:
```
Memory_packed ≈ Memory_unpacked × packing_density × 0.9
```

Where:
- `packing_density` = avg examples per sequence (e.g., 2.8)
- 0.9 = efficiency factor (packed sequences process more efficiently)

**Example**:
- Unpacked: batch_size=16, 1 example/seq → 16 examples, 2.4 GB
- Packed: batch_size=16, 2.8 examples/seq → 44.8 examples, ~6.0 GB

**This is why batch_size=16 with packed=True caused OOM!**
```

#### 2. Update "Batch Size Estimation" section

Add packing-aware logic after the initial batch size calculation steps:

```markdown
### Accounting for Packing

If using `packed: True`, adjust batch size calculation:

1. **Find packing density from prior runs:**
   ```bash
   # Check logs for packing info
   tail -200 {prior_run_log} | grep -i "packing\|average"

   # If no prior data, estimate based on task:
   # - Short examples (< 512 tokens): density ≈ 3-4
   # - Medium examples (512-1024 tokens): density ≈ 2-3
   # - Long examples (> 1024 tokens): density ≈ 1.5-2
   ```

2. **Adjust batch size for packing:**
   ```
   max_batch_size_packed = max_batch_size_unpacked ÷ packing_density
   ```

3. **Apply safety factor:**
   ```
   recommended_batch_size = max_batch_size_packed × 0.6
   ```

**Example Calculation**:
- Prior run: batch_size=16, packed=False, memory=2.4 GB, GPU=80 GB
- Unpacked headroom: 80 ÷ 2.4 = 33x
- Conservative max unpacked: 33 × 0.7 = 23
- **If enabling packing** with density=2.8:
  - Max packed: 23 ÷ 2.8 ≈ 8
  - Recommended: 8 × 0.6 ≈ 5
  - **Use batch_size=4 or batch_size=8 (test 8 first)**

**If no packing data available**:
- Start with batch_size=4 (1B), batch_size=2 (3B) when using packed=True
- Monitor first run and scale up if memory allows
- Document actual packing density for future reference
```

#### 3. Add packing to Configuration section

In the runs_plan.md template's "Configuration" section, update to include:

```markdown
## Configuration
- **Recipe**: `{recipe_name}`
- **GPUs per job**: {gpu_count}
- **Epochs**: {epochs}
- **Dataset packing**: {True/False} - {explanation}
  - Packing density: {avg_examples_per_sequence} (from prior runs)
  - Impact on memory: {estimated_multiplier}x compared to unpacked
- **Batch sizes**: {batch_size_details} (adjusted for packing)
- **Gradient accumulation**: {steps} (effective batch size: {effective_batch})
- **LoRA alpha**: {alpha_value} (auto-set to 2 × rank)
```

#### 4. Add to "Questions to Ask" section

In the "Training Options" subsection, add:

```markdown
**Should dataset packing be enabled?**
- Default: Yes (packed=True) for efficiency
- Packing concatenates multiple examples into sequences up to max_seq_len
- More efficient (no padding waste) but requires smaller batch sizes
- If unsure: Start with packed=True and batch_size=4 (conservative)
- If have prior runs with packing disabled: Need to recalculate batch sizes
```

---

### B. create-torchtune-config Skill

**File**: `.claude/skills/create-torchtune-config/skill.md`

**Current Issue**: No guidance on packing parameter or its interaction with batch size.

**Changes Needed**:

#### 1. Add new section after "Critical Checks"

```markdown
### 5. Packing and Batch Size Validation

**CRITICAL**: If `packed: True`, batch sizes must be reduced compared to unpacked configs!

**Validation steps**:

1. **Check packing setting** in template or config:
   ```yaml
   dataset:
     packed: True  # or False
   ```

2. **If packed=True, verify batch size is appropriate:**

   **Safe defaults for packed=True**:
   - 1B models: batch_size ≤ 8
   - 3B models: batch_size ≤ 4
   - 7B+ models: batch_size ≤ 2

   **If batch_size is higher**, check for prior run validation:
   - Has a similar run completed successfully with this batch size + packed=True?
   - If no: REDUCE batch size to safe defaults
   - If yes: Document the successful config in validation notes

3. **Cross-check with runs_plan.md**:
   - If runs_plan.md specifies packing density, use those calculations
   - If runs_plan.md batch sizes are from unpacked runs, WARN and recalculate

**Example validation**:
```bash
# Check config for packing
grep "packed:" finetune.yaml

# If packed: True and batch_size > 8 for 1B model:
echo "WARNING: batch_size=16 with packed=True may cause OOM"
echo "Recommended: Reduce to batch_size=4-8"
```

**Add to validation command**:
```bash
# Validate AND check packing/batch size consistency
source ~/.bashrc && conda activate ttenv && \
  cd /path/to/run1 && \
  echo "Checking packing/batch_size..." && \
  grep -E "packed:|batch_size:" finetune.yaml && \
  tune validate finetune.yaml
```
```

#### 2. Update "Customize configs for each run" section

Add to the variables list in this section:

```markdown
- `batch_size`: Based on model size AND packing setting
  - If packed=False: 1B: 16, 3B: 8 (can be larger)
  - If packed=True: 1B: 4-8, 3B: 2-4 (must be smaller)
  - Verify against runs_plan.md estimates
- `packed`: True (efficient) or False (safer, simpler)
  - Check runs_plan.md for packing density estimates
  - If no prior packing data: Use packed=True with conservative batch_size
```

---

### C. setup-experiment-dirs Skill

**File**: `.claude/skills/setup-experiment-dirs/skill.md`

**Changes Needed**:

Add validation step that checks packing + batch_size consistency after configs are generated.

After config generation step, add:

```markdown
### Validate Packing Configuration

After all `finetune.yaml` files are created, validate packing settings:

```bash
# Check all configs for packing + batch_size combinations
for dir in */; do
  if [[ -f "$dir/finetune.yaml" ]]; then
    echo "=== $dir ==="
    packed=$(grep "packed:" "$dir/finetune.yaml" | awk '{print $2}')
    batch_size=$(grep "^batch_size:" "$dir/finetune.yaml" | awk '{print $2}')
    model=$(grep "_component_.*llama3_2" "$dir/finetune.yaml" | grep -o "[0-9]b")

    echo "  packed=$packed, batch_size=$batch_size, model=$model"

    # Warn if risky combination
    if [[ "$packed" == "True" ]] && [[ "$batch_size" -gt 8 ]] && [[ "$model" == "1b" ]]; then
      echo "  ⚠️  WARNING: High batch_size with packing may cause OOM"
    fi
  fi
done
```

**If warnings appear**: Ask user if they want to adjust batch sizes before proceeding.
```

---

## 2. Tools/Scripts That Need Updates

### A. tools/torchtune/templates/finetune_template.yaml

**File**: `tools/torchtune/templates/finetune_template.yaml`

**Current Issue**:
- Template has `packed: True` (default)
- Template has `batch_size: 4` (default, but often overridden to 16)
- No warnings or documentation about interaction

**Changes Needed**:

#### 1. Add comments explaining packing

Update the `dataset.packed` field to include explanatory comments:

```yaml
dataset:
  _component_: torchtune.datasets.instruct_dataset
  source: "parquet"
  data_dir: "${input_dir}/${dataset_label}"
  split: "train"
  # Packing: Concatenates multiple examples into sequences up to max_seq_len
  # Increases efficiency (no padding waste) but requires smaller batch_size
  # Rule of thumb: If packed=True, use batch_size ≤ 8 for 1B, ≤ 4 for 3B
  # If packed=False, can use larger batch_size (16+ for 1B, 8+ for 3B)
  packed: True
```

#### 2. Add comment to batch_size

Update the `batch_size` field to include explanatory comments:

```yaml
# Batch size: Number of sequences per batch
# If packed=True: Use smaller values (4-8 for 1B, 2-4 for 3B)
# If packed=False: Can use larger values (16+ for 1B, 8+ for 3B)
# See docs for memory estimation with packing
batch_size: 4
```

#### 3. Consider adding gradient_accumulation_steps

After `batch_size`, add:

```yaml
batch_size: 4
gradient_accumulation_steps: 1  # Increase to 2-4 if need larger effective batch
```

**Explanation**: Provides users an alternative to increasing batch_size when packed=True.

---

### B. tools/torchtune/setup_finetune.py

**File**: `tools/torchtune/setup_finetune.py`

**Changes Needed**:

#### 1. Add batch size validation when generating configs

After loading/generating the config, before writing `finetune.yaml`, add validation:

```python
def validate_packing_batch_size(config_dict, model_name, skip_validation=False):
    """Validate that batch_size is appropriate for packing setting."""
    import re

    packed = config_dict.get('dataset', {}).get('packed', False)
    batch_size = config_dict.get('batch_size', 4)

    # Determine model size using robust regex extraction
    # Matches patterns like "1b", "3b", "7b", "13b", "70b", etc.
    match = re.search(r'(\d+)b', model_name.lower())
    if not match:
        # Unknown model size, skip validation
        return

    model_size_b = int(match.group(1))

    # Determine safe batch sizes based on model size
    if model_size_b <= 1:
        model_size = '1B'
        max_safe_batch_packed = 8
        max_safe_batch_unpacked = 20
    elif model_size_b <= 3:
        model_size = '3B'
        max_safe_batch_packed = 4
        max_safe_batch_unpacked = 12
    elif model_size_b <= 8:
        model_size = '7B/8B'
        max_safe_batch_packed = 2
        max_safe_batch_unpacked = 6
    elif model_size_b <= 13:
        model_size = '13B'
        max_safe_batch_packed = 1
        max_safe_batch_unpacked = 4
    elif model_size_b <= 70:
        model_size = '70B'
        max_safe_batch_packed = 1
        max_safe_batch_unpacked = 2
    else:
        model_size = f'{model_size_b}B'
        max_safe_batch_packed = 1
        max_safe_batch_unpacked = 1

    # Check for risky combinations
    if packed and batch_size > max_safe_batch_packed:
        print(f"⚠️  WARNING: batch_size={batch_size} with packed=True for {model_size} model")
        print(f"   May cause OOM. Recommended: batch_size ≤ {max_safe_batch_packed}")
        print(f"   Or disable packing: packed=False (allows batch_size ≤ {max_safe_batch_unpacked})")

        if not skip_validation:
            # Ask user if running interactively
            response = input("   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("   Aborting. Please adjust batch_size in setup_finetune.yaml")
                sys.exit(1)
        else:
            print("   WARNING: Continuing despite validation issues (--skip-validation enabled)")

    elif not packed and batch_size <= 4:
        print(f"ℹ️  NOTE: Using packed=False with small batch_size={batch_size}")
        print(f"   Consider packed=True for better efficiency with this batch size")

# Add command-line argument to setup_finetune.py argument parser:
# parser.add_argument('--skip-validation', action='store_true',
#                     help='Skip interactive validation checks')

# Call validation before writing config
validate_packing_batch_size(config, args.model_name, skip_validation=args.skip_validation)
```

---

### C. tools/slurm/generate_slurm_scripts.py

**File**: `tools/slurm/generate_slurm_scripts.py`

**Changes Needed**:

#### 1. Add packing awareness to resource estimation

Currently uses runs_plan.md estimates. Add validation that checks if estimates account for packing:

After reading and parsing the runs_plan.md file, add validation:

```python
def validate_packing_in_plan(runs_plan_path, run_configs, skip_validation=False):
    """Check if runs_plan.md accounts for packing in its estimates."""

    with open(runs_plan_path, 'r') as f:
        plan_content = f.read()

    # Check if plan mentions packing
    mentions_packing = 'packing' in plan_content.lower() or 'packed' in plan_content.lower()

    if not mentions_packing:
        # Check if any configs use packing
        any_packed = False
        for run_dir, config in run_configs.items():
            if config.get('dataset', {}).get('packed', False):
                any_packed = True
                break

        if any_packed:
            print("⚠️  WARNING: Configs use packed=True but runs_plan.md doesn't mention packing")
            print("   Batch size estimates may be incorrect")
            print("   Recommend: Review runs_plan.md and regenerate with packing-aware estimates")

            if not skip_validation:
                response = input("   Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("   Aborting. Please update runs_plan.md")
                    sys.exit(1)
            else:
                print("   WARNING: Continuing despite validation issues (--skip-validation enabled)")

# Add command-line argument to generate_slurm_scripts.py argument parser:
# parser.add_argument('--skip-validation', action='store_true',
#                     help='Skip interactive validation checks')
```

---

## 3. Documentation Improvements

### A. CLAUDE.md

**File**: `CLAUDE.md`

**Changes Needed**:

Add section on packing to "Key Conventions":

```markdown
### Dataset Packing
- **Packing enabled**: `packed: True` (default) - concatenates examples, more efficient
- **Packing disabled**: `packed: False` - each example padded, simpler, safer
- **Memory consideration**: Packed requires smaller batch_size (typically 1/3 to 1/2 of unpacked)
- **Batch size guidelines**:
  - Packed: 1B ≤ 8, 3B ≤ 4, 7B ≤ 2
  - Unpacked: 1B ≤ 16, 3B ≤ 8, 7B ≤ 4
- **When in doubt**: Use packed=True with conservative batch_size=4
```

### B. docs/finetune_and_evaluate.md

**File**: `docs/finetune_and_evaluate.md`

**Changes Needed**:

Add section after "Step: write a fine-tuning config file in .yaml format":

```markdown
## Understanding Dataset Packing

Packing is a memory optimization that can significantly affect training:

**What it does**:
- `packed: False`: Each example padded to max_seq_len (standard approach)
- `packed: True`: Multiple examples concatenated to fill max_seq_len (efficient)

**Why it matters**:
- Packing eliminates wasted padding tokens → faster training
- BUT: Each "batch item" contains multiple examples → higher memory per item
- Effective batch size with packing ≈ batch_size × 2-4 (depends on example length)

**Choosing settings**:
1. If examples are short (< 30% of max_seq_len): Use packed=True with batch_size ≤ 8
2. If examples are long (> 70% of max_seq_len): Use packed=False with batch_size ≤ 16
3. If unsure: Start with packed=True, batch_size=4 (always safe)

**Memory debugging**:
- If OOM with packed=True: Reduce batch_size (try 4, 2, or 1)
- If OOM persists: Try packed=False (allows larger batch_size)
- If still OOM: Enable activation checkpointing or reduce model size
```

---

## 4. New Validation Script

### Create: tools/validate_config.py

**Purpose**: Standalone script to validate torchtune configs for common issues including packing.

```python
#!/usr/bin/env python3
"""
Validate torchtune configuration files for common issues.

Usage:
    python tools/validate_config.py /path/to/finetune.yaml
    python tools/validate_config.py /path/to/experiment_dir/  # validates all subdirs
"""

import sys
import re
import yaml
from pathlib import Path

def extract_model_size(model_component):
    """Extract model size in billions from model component string."""
    # Try to find pattern like "1b", "3b", "7b", "70b", etc.
    match = re.search(r'(\d+)b', model_component.lower())
    if match:
        return int(match.group(1))
    return None

def get_batch_size_limits(model_size_b):
    """Get recommended batch size limits based on model size in billions."""
    if model_size_b is None:
        return None, None, 'Unknown'

    if model_size_b <= 1:
        return 8, 20, '1B'
    elif model_size_b <= 3:
        return 4, 12, '3B'
    elif model_size_b <= 8:
        return 2, 6, '7B/8B'
    elif model_size_b <= 13:
        return 1, 4, '13B'
    elif model_size_b <= 70:
        return 1, 2, '70B'
    else:
        return 1, 1, f'{model_size_b}B'

def validate_single_config(config_path):
    """Validate a single torchtune config file."""

    print(f"\n{'='*60}")
    print(f"Validating: {config_path}")
    print(f"{'='*60}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    issues = []
    warnings = []

    # Extract key parameters
    batch_size = config.get('batch_size', 4)
    packed = config.get('dataset', {}).get('packed', False)
    model_component = config.get('model', {}).get('_component_', '')
    max_seq_len = config.get('tokenizer', {}).get('max_seq_len', 2048)
    gradient_accum = config.get('gradient_accumulation_steps', 1)
    enable_activation_ckpt = config.get('enable_activation_checkpointing', False)

    # Determine model size using robust extraction
    model_size_b = extract_model_size(model_component)
    max_batch_packed, max_batch_unpacked, model_size = get_batch_size_limits(model_size_b)

    print(f"Model: {model_size}")
    print(f"Batch size: {batch_size}")
    print(f"Packed: {packed}")
    print(f"Max seq len: {max_seq_len}")
    print(f"Gradient accumulation: {gradient_accum}")
    print(f"Activation checkpointing: {enable_activation_ckpt}")

    # Check 1: Packing + Batch size combination
    if packed and max_batch_packed:
        if batch_size > max_batch_packed:
            issues.append(
                f"❌ CRITICAL: batch_size={batch_size} too high for packed=True on {model_size} model\n"
                f"   Likely to cause OOM. Recommended: batch_size ≤ {max_batch_packed}"
            )
        elif batch_size == max_batch_packed:
            warnings.append(
                f"⚠️  WARNING: batch_size={batch_size} at maximum for packed=True on {model_size}\n"
                f"   May cause OOM depending on GPU. Consider batch_size={max_batch_packed // 2} for safety"
            )

    # Check 2: Not using packing with small batch
    if not packed and batch_size <= 4:
        warnings.append(
            f"⚠️  INFO: Using packed=False with small batch_size={batch_size}\n"
            f"   Consider packed=True for {2-3}x faster training at this batch size"
        )

    # Check 3: Effective batch size
    effective_batch = batch_size * gradient_accum
    if packed:
        effective_batch_range = f"{effective_batch * 2}-{effective_batch * 4}"
        print(f"Effective batch size: {effective_batch} sequences (~{effective_batch_range} examples with packing)")
    else:
        print(f"Effective batch size: {effective_batch} examples")

    # Check 4: Dataset source and split/field
    dataset_source = config.get('dataset', {}).get('source', 'unknown')
    has_split = 'split' in config.get('dataset', {})
    has_field = 'field' in config.get('dataset', {})

    if dataset_source in ['json', 'parquet']:
        if has_split and not has_field:
            issues.append(
                f"❌ CRITICAL: Using 'split' with local {dataset_source} files\n"
                f"   Should use 'field' for local files, 'split' for HuggingFace datasets"
            )

    # Print results
    print(f"\n{'─'*60}")
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"\n{issue}")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"\n{warning}")

    if not issues and not warnings:
        print("✓ No issues found - config looks good!")

    print(f"{'─'*60}")

    return len(issues) == 0

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        # Validate single file
        success = validate_single_config(path)
        sys.exit(0 if success else 1)

    elif path.is_dir():
        # Validate all finetune.yaml in subdirectories
        configs = list(path.glob("*/finetune.yaml"))

        if not configs:
            print(f"No finetune.yaml files found in {path}")
            sys.exit(1)

        print(f"Found {len(configs)} config files to validate")

        results = []
        for config_path in sorted(configs):
            success = validate_single_config(config_path)
            results.append((config_path.parent.name, success))

        # Summary
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")

        passed = sum(1 for _, success in results if success)
        failed = len(results) - passed

        for run_name, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status}: {run_name}")

        print(f"\nTotal: {passed} passed, {failed} failed")

        sys.exit(0 if failed == 0 else 1)

    else:
        print(f"Error: {path} is not a file or directory")
        sys.exit(1)

if __name__ == '__main__':
    main()
```

---

## 5. Priority Implementation Order

### Phase 1: Critical Fixes (Do First)
1. ✅ **Update plan-runs skill** - Add packing section and batch size adjustment
2. ✅ **Update create-torchtune-config skill** - Add packing validation
3. ✅ **Add comments to template** - Explain packing in finetune_template.yaml

### Phase 2: Enhanced Validation (Do Second)
4. ⏸️ **Create validate_config.py** - Standalone validation script
5. ⏸️ **Update setup_finetune.py** - Add batch size validation function
6. ⏸️ **Update generate_slurm_scripts.py** - Check packing in runs_plan.md

### Phase 3: Documentation (Do Third)
7. ⏸️ **Update CLAUDE.md** - Add packing to Key Conventions
8. ⏸️ **Update docs/finetune_and_evaluate.md** - Add packing explanation
9. ⏸️ **Update setup-experiment-dirs skill** - Add packing validation step

### Phase 4: Metrics Logging (Do Fourth)
10. ⏸️ **Extend runs_status.yaml schema** - Add config and metrics fields
11. ⏸️ **Update update-run-status skill** - Add metric extraction functions
12. ⏸️ **Update monitor-jobs skill** - Auto-extract metrics on completion
13. ⏸️ **Update summarize-experiments skill** - Display and compare packing metrics

---

## 6. Testing Plan

After implementing changes:

### Test Case 1: New Experiment with Packing
1. Run plan-runs skill for 1B + 3B models with packed=True
2. Verify batch sizes are set to ≤ 8 (1B) and ≤ 4 (3B)
3. Verify runs_plan.md documents packing density
4. Run create-torchtune-config skill
5. Run validate_config.py on all generated configs
6. Submit one test job and verify no OOM

### Test Case 2: Unpacked Dataset
1. Run plan-runs with packed=False recommendation
2. Verify batch sizes can be larger (16 for 1B, 8 for 3B)
3. Verify validation passes
4. Submit test job and verify training works

### Test Case 3: Migrating from Unpacked to Packed
1. Start with old runs_plan.md (packed=False, batch_size=16)
2. Update to packed=True
3. Verify validation catches the issue and recommends new batch sizes
4. Adjust batch sizes to 8 → 4
5. Verify no OOM

---

## 7. Related Issues & Context

**Root cause identified**: 2025-10-20
- Experiment `cap_cross_eval_5_9_13L_2025-10-19` used batch_size=16 with packed=True
- This caused 1B model to use 78GB instead of expected 20-25GB
- OOM error on 80GB GPU despite having plenty of theoretical headroom
- Batch size estimates were from prior runs that likely used packed=False

**Key insight**: Torchtune defaults use packed=False with gradient accumulation, but cruijff_kit template defaults to packed=True without adjusting batch sizes accordingly.

**Design decision**: Keep packed=True as default (it's more efficient) but ensure batch sizes are calculated correctly for packing.

---

## 8. Additional Considerations

### A. Packing Density Logging
Consider adding logging to track packing density across runs:
- Log avg examples per packed sequence
- Log packing efficiency (% non-padding tokens)
- Store in runs_status.yaml or separate metrics file
- Use for future batch size estimation

### B. Adaptive Batch Sizing
Future enhancement: Script that automatically determines optimal batch size:
1. Start with batch_size=1
2. Gradually increase until memory reaches 70% of GPU capacity
3. Use that as the recommended batch size
4. Save to runs_plan.md for future reference

### C. Documentation Template
Add to runs_plan.md template:

```markdown
## Packing Analysis (from prior runs)

| Model | Batch Size | Packed | Packing Density | Memory Usage | GPU Type |
|-------|-----------|--------|-----------------|--------------|----------|
| 1B    | 4         | True   | 2.8 ex/seq      | 5.8 GB       | A100 80GB |
| 1B    | 16        | False  | 1.0 ex/seq      | 2.4 GB       | A100 80GB |
| 3B    | 4         | True   | 2.8 ex/seq      | 14.1 GB      | A100 80GB |

**Recommendation**: Use packed=True with batch_size=4 (1B) and batch_size=2 (3B) for this dataset.
```

---

## 9. Logging Packing Metrics

### Why Log Packing Information?

Packing configuration and runtime metrics are **critical** for:
- **Debugging OOM failures**: Understand why memory usage was higher than expected
- **Planning future runs**: Use actual packing density from similar datasets
- **Computational reproducibility**: Document exact training conditions
- **Performance analysis**: Identify bottlenecks and optimization opportunities
- **Resource estimation**: Accurate cost and time predictions for new experiments

### What to Log

#### A. Static Configuration (Known Before Training)

These should be recorded when configs are generated:

```yaml
# In runs_status.yaml or experiment metadata
config:
  batch_size: 4
  packed: true
  max_seq_len: 2048
  gradient_accumulation_steps: 1
  effective_batch_size: 4  # batch_size × gradient_accumulation_steps
```

#### B. Runtime Packing Metrics (Measured During Training)

These should be extracted from SLURM logs after dataset loading:

```yaml
# In runs_status.yaml or separate metrics file
packing_metrics:
  total_examples: 8000
  total_sequences: 2857
  packing_density: 2.80  # examples per sequence
  packing_efficiency: 87  # % non-padding tokens (if available)
```

#### C. Runtime Resource Metrics (Measured During Training)

These should be extracted from SLURM logs or GPU monitoring:

```yaml
# In runs_status.yaml or separate metrics file
resource_metrics:
  peak_gpu_memory_gb: 45.2
  avg_gpu_memory_gb: 38.7
  gpu_utilization_pct: 92
  training_time_minutes: 127
  tokens_per_second: 1834  # if available
```

### Where to Log

**Option 1: Extend runs_status.yaml** (Recommended)

Advantages:
- Centralized tracking in one file
- Already used for experiment status
- Easy to query all runs at once

Structure:
```yaml
# tasks/capitalization/runs_status.yaml
runs:
  - name: bright_horizon
    status: completed
    slurm_job_id: 12345678
    started: "2025-10-20T14:30:00"
    completed: "2025-10-20T16:37:00"

    # Static configuration
    config:
      model: llama-3.2-1b
      batch_size: 4
      packed: true
      max_seq_len: 2048
      gradient_accumulation_steps: 1
      lora_rank: 8
      lora_alpha: 16
      epochs: 5

    # Runtime metrics (extracted from logs)
    metrics:
      # Packing
      total_examples: 8000
      total_sequences: 2857
      packing_density: 2.80
      packing_efficiency: 87

      # Resources
      peak_gpu_memory_gb: 45.2
      avg_gpu_memory_gb: 38.7
      gpu_utilization_pct: 92
      training_time_minutes: 127

      # Performance
      tokens_per_second: 1834
      examples_per_second: 10.5
```

**Option 2: Separate Metrics File per Run**

Advantages:
- More scalable for many runs
- Can include detailed per-epoch metrics
- Doesn't clutter runs_status.yaml

Structure:
```yaml
# tasks/capitalization/run1/run_metrics.yaml
run_name: bright_horizon
timestamp: "2025-10-20T14:30:00"

static_config:
  model: llama-3.2-1b
  batch_size: 4
  packed: true
  max_seq_len: 2048
  gradient_accumulation_steps: 1
  effective_batch_size: 4

packing_metrics:
  total_examples: 8000
  total_sequences: 2857
  packing_density: 2.80
  packing_efficiency: 87
  dataset_path: /scratch/gpfs/user/data/capitalization_5letter

resource_metrics:
  peak_gpu_memory_gb: 45.2
  avg_gpu_memory_gb: 38.7
  gpu_utilization_pct: 92
  training_time_minutes: 127

performance_metrics:
  tokens_per_second: 1834
  examples_per_second: 10.5

per_epoch_metrics:
  - epoch: 0
    training_time_minutes: 25
    avg_gpu_memory_gb: 38.5
  - epoch: 1
    training_time_minutes: 26
    avg_gpu_memory_gb: 38.7
```

### How to Extract and Log

#### Implementation Approach 1: Extend update-run-status Skill

Update the existing `update-run-status` skill to also extract metrics:

```python
# In .claude/skills/update-run-status/update_run_status.py

def extract_packing_metrics(slurm_log_path):
    """Extract packing metrics from SLURM log file."""

    metrics = {
        'total_examples': None,
        'total_sequences': None,
        'packing_density': None,
        'packing_efficiency': None
    }

    with open(slurm_log_path, 'r') as f:
        content = f.read()

    # Pattern 1: Explicit packing stats
    match = re.search(r'Packed (\d+) examples into (\d+) sequences', content)
    if match:
        metrics['total_examples'] = int(match.group(1))
        metrics['total_sequences'] = int(match.group(2))
        metrics['packing_density'] = round(
            metrics['total_examples'] / metrics['total_sequences'], 2
        )

    # Pattern 2: Packing density directly stated
    match = re.search(r'[Pp]acking density[:\s]+([\d.]+)', content)
    if match:
        metrics['packing_density'] = float(match.group(1))

    # Pattern 3: Packing efficiency
    match = re.search(r'[Pp]acking efficiency[:\s]+([\d.]+)%?', content)
    if match:
        metrics['packing_efficiency'] = float(match.group(1))

    return metrics

def extract_resource_metrics(slurm_log_path):
    """Extract GPU and timing metrics from SLURM log."""

    metrics = {
        'peak_gpu_memory_gb': None,
        'training_time_minutes': None
    }

    with open(slurm_log_path, 'r') as f:
        content = f.read()

    # Look for GPU memory usage
    # (patterns depend on whether logging is enabled in the recipe)
    match = re.search(r'Peak GPU memory[:\s]+([\d.]+)\s*GB', content)
    if match:
        metrics['peak_gpu_memory_gb'] = float(match.group(1))

    # Calculate training time from timestamps
    start_match = re.search(r'Training started.*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', content)
    end_match = re.search(r'Training completed.*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', content)

    if start_match and end_match:
        from datetime import datetime
        start = datetime.strptime(start_match.group(1), '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(end_match.group(1), '%Y-%m-%d %H:%M:%S')
        metrics['training_time_minutes'] = round((end - start).total_seconds() / 60, 1)

    return metrics

# Update runs_status.yaml with these metrics when run completes
```

#### Implementation Approach 2: Create New extract-metrics Skill

Create a dedicated skill for metrics extraction:

```bash
# .claude/skills/extract-metrics/skill.md

## Extract Metrics

Extract training metrics from completed runs and add to runs_status.yaml

### Usage
When a user asks to "extract metrics" or "analyze run performance":

1. Ask for the run directory or run name
2. Find the SLURM log: `run_dir/slurm-*.out`
3. Extract metrics using patterns (see implementation above)
4. Update runs_status.yaml with the metrics
5. Display a summary of extracted metrics
```

#### Implementation Approach 3: Automatic Extraction in launch-runs

Modify the `launch-runs` skill to automatically extract metrics when monitoring jobs:

```python
# When job completes, automatically run:
# 1. Update status to "completed"
# 2. Extract metrics from logs
# 3. Add metrics to runs_status.yaml
# 4. Display summary
```

### Integration with Existing Skills

#### plan-runs Skill
- Include expected packing density in the plan based on task analysis
- Template includes packing metrics table (as shown in section 8.C)

#### setup-experiment-dirs Skill
- Initialize runs_status.yaml with static config fields
- Set metrics fields to null (to be filled later)

#### monitor-jobs Skill
- When jobs complete, automatically extract and log metrics
- Display metrics summary alongside completion status

#### summarize-experiments Skill
- Read metrics from runs_status.yaml
- Compare packing densities across runs
- Identify optimal batch size/packing combinations
- Generate recommendations for future runs

---

## Notes

- Priority is enabling packing efficiently, not disabling it
- Focus on education and validation rather than restriction
- Users should understand tradeoffs and make informed choices
